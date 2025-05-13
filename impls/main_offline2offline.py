import os
import platform

import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents
from utils.env_utils import make_env_and_datasets
from utils.wrappers import PositiveRewardShaping, OfflineObservationNormalizer
from utils.datasets import GCDataset, Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_class', 'Dataset', 'Dataset class name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('pretraining_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('finetuning_steps', 500_000, 'Number of online steps.')
flags.DEFINE_integer('finetuning_size', 500_000, 'Size of the dataset for finetuning.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1_500_000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_string('obs_norm_type', 'none',
                    'Type of observation normalization. (none, normal, bounded)')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('num_aug', 1, 'Number of image augmentations.')
flags.DEFINE_integer('inplace_aug', 1, 'Whether to replace the original image after applying augmentations.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb:
        _, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='ogbench', group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    _, _, pretraining_train_dataset, pretraining_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=10_000_000, reward_free=True)
    _, eval_env, finetuning_train_dataset, finetuning_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.finetuning_size, reward_free=False)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    pretraining_train_dataset = Dataset.create(**pretraining_train_dataset)
    finetuning_train_dataset = Dataset.create(**finetuning_train_dataset)
    # if FLAGS.balanced_sampling:
    #     # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
    #     example_transition = {k: v[0] for k, v in train_dataset.items()}
    #     replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    # else:
    #     # Use the training dataset as the replay buffer.
    #     train_dataset = ReplayBuffer.create_from_initial_dataset(
    #         dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    #     )
    #     replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [pretraining_train_dataset, pretraining_val_dataset,
                    finetuning_train_dataset, finetuning_val_dataset]:
        if dataset is not None:
            dataset.obs_norm_type = FLAGS.obs_norm_type
            dataset.p_aug = FLAGS.p_aug
            dataset.num_aug = FLAGS.num_aug
            dataset.inplace_aug = FLAGS.inplace_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['rebrac', 'dino_rebrac', 'td_infonce',
                                        'sarsa_ifac_q', 'sarsa_ifql', 'sarsa_ifql_gpi',
                                        'sarsa_ifql_vib_gpi', 'sarsa_ifql_vfm_gpi',
                                        'fb_repr_fom', 'hilp_fom']:
                dataset.return_next_actions = True
            dataset.normalize_observations()
    if FLAGS.dataset_class == 'GCDataset':
        config['p_aug'] = FLAGS.p_aug
        config['frame_stack'] = FLAGS.frame_stack
        pretraining_train_dataset = GCDataset(pretraining_train_dataset, config)
        finetuning_train_dataset = GCDataset(finetuning_train_dataset, config)
        if pretraining_val_dataset is not None:
            pretraining_val_dataset = GCDataset(pretraining_val_dataset, config)
            finetuning_val_dataset = GCDataset(finetuning_val_dataset, config)

    # Create agent.
    example_batch = pretraining_train_dataset.sample(1)
    # if config['discrete']:
    #     # Fill with the maximum action to let the agent know the action space size.
    #     example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    # if config['agent_name'] in ['mcfac', 'ifac', 'sarsa_ifac', 'sarsa_ifac_q']:
    #     if hasattr(pretraining_train_dataset, 'dataset'):
    #         dataset_observations = pretraining_train_dataset.dataset['observations']
    #     else:
    #         dataset_observations = pretraining_train_dataset['observations']
    #     config['dataset_obs_min'] = jnp.min(dataset_observations, axis=0)
    #     config['dataset_obs_max'] = jnp.max(dataset_observations, axis=0)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # if FLAGS.obs_norm_type in ['normal', 'bounded']:
    #     agent = OfflineObservationNormalizer.create(
    #         agent,
    #         pretraining_train_dataset,
    #         normalizer_type=FLAGS.obs_norm_type
    #     )
    # agent = OfflineObservationNormalizer.create(
    #     agent,
    #     pretraining_train_dataset,
    #     normalizer_type=FLAGS.obs_norm_type
    # )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    pretraining_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_train.csv'))
    pretraining_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_eval.csv'))
    finetuning_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_train.csv'))
    finetuning_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_eval.csv'))
    first_time = time.time()
    last_time = time.time()

    inferred_latent = None  # Only for HILP and FB.
    expl_metrics = dict()
    for i in tqdm.tqdm(range(1, FLAGS.pretraining_steps + FLAGS.finetuning_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.pretraining_steps:
            # Offline pre-training.
            batch = pretraining_train_dataset.sample(config['batch_size'])
            train_logger = pretraining_train_logger
            eval_logger = pretraining_eval_logger

            agent, update_info = agent.pretrain(batch)
        else:
            if (i == (FLAGS.pretraining_steps + 1)) and config['agent_name'] in ['hilp', 'fb_repr']:
                # Infer the latent vector.
                num_samples = 0
                inference_batch = defaultdict(list)
                while num_samples < config['num_latent_inference_samples']:
                    batch = finetuning_train_dataset.sample(config['batch_size'])
                    for k, v in batch.items():
                        inference_batch[k].append(v)
                    num_samples += config['batch_size']
                for k, v in inference_batch.items():
                    if k not in ['observation_min', 'observation_max']:
                        inference_batch[k] = np.concatenate(v, axis=0)[:config['num_latent_inference_samples']]

                inferred_latent = agent.infer_latent(inference_batch)
                inferred_latent = np.array(inferred_latent)

            # for OfflineObservationNormalizer
            # if i == (FLAGS.pretraining_steps + 1):
            #     agent = agent.update_dataset(finetuning_train_dataset)

            # Offline fine-tuning.
            batch = finetuning_train_dataset.sample(config['batch_size'])
            train_logger = finetuning_train_logger
            eval_logger = finetuning_eval_logger

            if config['agent_name'] in ['hilp', 'fb_repr']:
                batch['latents'] = np.tile(inferred_latent, (batch['observations'].shape[0], 1))

            agent, update_info = agent.finetune(batch, full_update=(i % config['actor_freq'] == 0))

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if i <= FLAGS.pretraining_steps:
                val_dataset = pretraining_val_dataset
                loss_fn = agent.pretraining_loss
            else:
                val_dataset = finetuning_val_dataset
                loss_fn = agent.total_loss
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])

                if config['agent_name'] in ['hilp', 'fb_repr'] and (inferred_latent is not None):
                    val_batch['latents'] = np.tile(inferred_latent, (val_batch['observations'].shape[0], 1))

                _, val_info = loss_fn(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if (FLAGS.eval_interval != 0 and (i > FLAGS.pretraining_steps)
            and (i == (FLAGS.pretraining_steps + 1) or i % FLAGS.eval_interval == 0)):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                dataset=finetuning_train_dataset,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                inferred_latent=inferred_latent,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    pretraining_train_logger.close()
    pretraining_eval_logger.close()
    finetuning_train_logger.close()
    finetuning_eval_logger.close()


if __name__ == '__main__':
    app.run(main)
