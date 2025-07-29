import gc
import json
import os
import random
import time

import numpy as np
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from agents.vqvae import get_config as vqvae_get_config
from utils.datasets import augment
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'google_robot_pick_coke_can', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_class', 'Dataset', 'Dataset class name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('vqvae_restore_path', None, 'VQVAE restore path.')
flags.DEFINE_integer('vqvae_restore_epoch', None, 'VQVAE restore epoch.')

flags.DEFINE_integer('pretraining_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('finetuning_steps', 500_000, 'Replay buffer size.')
flags.DEFINE_integer('finetuning_size', 500_000, 'Size of the dataset for finetuning.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1_500_000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

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

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], 'GPU')

    # Make environment and datasets.
    config = FLAGS.agent
    _, _, pretraining_train_dataset, pretraining_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=10_000_000,
        action_clip_eps=None)
    _, eval_env, finetuning_train_dataset, finetuning_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.finetuning_size,
        action_clip_eps=None)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    # Set up datasets.
    # Important:
    #   https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033)
    #   remember to set environment variable LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4 to prevent cpu mem leak.
    pretraining_train_dataset = (
        pretraining_train_dataset
        .shuffle(200_000)
        .repeat()
        .batch(config['batch_size'])
        .prefetch(tf.data.AUTOTUNE)
    )
    pretraining_train_dataset_iter = pretraining_train_dataset.as_numpy_iterator()
    pretraining_val_dataset = (
        pretraining_val_dataset
        .shuffle(20_000)
        .repeat()
        .batch(config['batch_size'])
        .prefetch(tf.data.AUTOTUNE)
    )
    pretraining_val_dataset_iter = pretraining_val_dataset.as_numpy_iterator()

    example_batch = next(pretraining_val_dataset_iter)

    # np.all(example_batch['observations'].reshape(-1, 64, 64, 3, 3)[..., 1:, :] == example_batch['next_observations'].reshape(-1, 64, 64, 3, 3)[..., :2, :]) = True
    # from PIL import Image
    # img = Image.fromarray(example_batch['observations'].reshape(-1, 128, 128, 3, 3)[0, ..., 1, :])
    # img.save(os.path.join(FLAGS.save_dir, 'widowx_spoon_on_towel_obs_img.png'))
    # next_img = Image.fromarray(example_batch['next_observations'].reshape(-1, 128, 128, 3, 3)[0, ..., 1, :])
    # next_img.save(os.path.join(FLAGS.save_dir, 'widowx_spoon_on_towel_next_obs_img.png'))

    # Restore vqvae
    vqvae = None
    if FLAGS.vqvae_restore_path is not None:
        assert 'vqvae' in agents
        vqvae_class = agents['vqvae']
        vqvae_config = vqvae_get_config()
        vqvae_config['encoder'] = 'resnet_34'  # (chongyi): make this configurable
        vqvae_config['decoder'] = 'resnet_34'  # (chongyi): make this configurable
        vqvae = vqvae_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            vqvae_config,
        )

        vqvae = restore_agent(vqvae, FLAGS.vqvae_restore_path, FLAGS.vqvae_restore_epoch)
        example_batch['observations'] = np.asarray(
            vqvae.encode(example_batch['observations'], flatten=True)
        )
        example_batch['next_observations'] = np.asarray(
            vqvae.encode(example_batch['next_observations'], flatten=True)
        )

    # Create agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

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

    # Offline RL.
    for i in tqdm.tqdm(range(1, FLAGS.pretraining_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = next(pretraining_train_dataset_iter)
        train_logger = pretraining_train_logger
        eval_logger = pretraining_eval_logger

        # data augmentation
        if np.random.rand() < FLAGS.p_aug:
            if FLAGS.inplace_aug:
                augment(batch, ['observations', 'next_observations'])
            else:
                for aux_idx in range(FLAGS.num_aug):
                    augment(batch, ['observations', 'next_observations'], f'aug{aux_idx + 1}_')
        if FLAGS.vqvae_restore_path is not None:
            batch['observations'] = np.asarray(
                vqvae.encode(batch['observations'], flatten=True)
            )
            batch['next_observations'] = np.asarray(
                vqvae.encode(batch['next_observations'], flatten=True)
            )
            if 'aug1_observations' in batch:
                for aux_idx in range(FLAGS.num_aug):
                    batch[f'aug{aux_idx + 1}_observations'] = np.asarray(
                        vqvae.encode(batch[f'aug{aux_idx + 1}_observations'], flatten=True)
                    )
                    batch[f'aug{aux_idx + 1}_next_observations'] = np.asarray(
                        vqvae.encode(batch[f'aug{aux_idx + 1}_next_observations'], flatten=True)
                    )
        agent, update_info = agent.pretrain(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            val_dataset_iter = pretraining_val_dataset_iter
            loss_fn = agent.pretraining_loss
            val_batch = next(val_dataset_iter)
            # data augmentation
            if np.random.rand() < FLAGS.p_aug:
                if FLAGS.inplace_aug:
                    augment(val_batch, ['observations', 'next_observations'])
                else:
                    for aux_idx in range(FLAGS.num_aug):
                        augment(val_batch, ['observations', 'next_observations'], f'aug{aux_idx + 1}_')
            if FLAGS.vqvae_restore_path is not None:
                val_batch['observations'] = np.asarray(
                    vqvae.encode(val_batch['observations'], flatten=True)
                )
                val_batch['next_observations'] = np.asarray(
                    vqvae.encode(val_batch['next_observations'], flatten=True)
                )
                if 'aug1_observations' in val_batch:
                    for aux_idx in range(FLAGS.num_aug):
                        val_batch[f'aug{aux_idx + 1}_observations'] = np.asarray(
                            vqvae.encode(val_batch[f'aug{aux_idx + 1}_observations'], flatten=True)
                        )
                        val_batch[f'aug{aux_idx + 1}_next_observations'] = np.asarray(
                            vqvae.encode(val_batch[f'aug{aux_idx + 1}_next_observations'], flatten=True)
                        )
            _, val_info = loss_fn(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                vqvae=vqvae,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
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

    del pretraining_train_dataset_iter, pretraining_train_dataset
    del pretraining_val_dataset_iter, pretraining_val_dataset
    gc.collect()

    finetuning_train_dataset = (
        finetuning_train_dataset
        .shuffle(200_000)
        .repeat()
        .batch(config['batch_size'])
        .prefetch(10)
    )
    finetuning_train_dataset_iter = finetuning_train_dataset.as_numpy_iterator()
    finetuning_val_dataset = (
        finetuning_val_dataset
        .shuffle(20_000)
        .repeat()
        .batch(config['batch_size'])
        .prefetch(10)
    )
    finetuning_val_dataset_iter = finetuning_val_dataset.as_numpy_iterator()

    for i in tqdm.tqdm(range(FLAGS.pretraining_steps, FLAGS.pretraining_steps + FLAGS.finetuning_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = next(finetuning_train_dataset_iter)
        train_logger = finetuning_train_logger
        eval_logger = finetuning_eval_logger

        # data augmentation
        if np.random.rand() < FLAGS.p_aug:
            if FLAGS.inplace_aug:
                augment(batch, ['observations', 'next_observations'])
            else:
                for aux_idx in range(FLAGS.num_aug):
                    augment(batch, ['observations', 'next_observations'], f'aug{aux_idx + 1}_')
        if FLAGS.vqvae_restore_path is not None:
            batch['observations'] = np.asarray(
                vqvae.encode(batch['observations'], flatten=True)
            )
            batch['next_observations'] = np.asarray(
                vqvae.encode(batch['next_observations'], flatten=True)
            )
            if 'aug1_observations' in batch:
                for aux_idx in range(FLAGS.num_aug):
                    batch[f'aug{aux_idx + 1}_observations'] = np.asarray(
                        vqvae.encode(batch[f'aug{aux_idx + 1}_observations'], flatten=True)
                    )
                    batch[f'aug{aux_idx + 1}_next_observations'] = np.asarray(
                        vqvae.encode(batch[f'aug{aux_idx + 1}_next_observations'], flatten=True)
                    )
        agent, update_info = agent.finetune(batch, full_update=(i % config['actor_freq'] == 0))

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            val_dataset_iter = finetuning_val_dataset_iter
            loss_fn = agent.total_loss
            val_batch = next(val_dataset_iter)
            # data augmentation
            if np.random.rand() < FLAGS.p_aug:
                if FLAGS.inplace_aug:
                    augment(val_batch, ['observations', 'next_observations'])
                else:
                    for aux_idx in range(FLAGS.num_aug):
                        augment(val_batch, ['observations', 'next_observations'], f'aug{aux_idx + 1}_')
            if FLAGS.vqvae_restore_path is not None:
                val_batch['observations'] = np.asarray(
                    vqvae.encode(val_batch['observations'], flatten=True)
                )
                val_batch['next_observations'] = np.asarray(
                    vqvae.encode(val_batch['next_observations'], flatten=True)
                )
                if 'aug1_observations' in val_batch:
                    for aux_idx in range(FLAGS.num_aug):
                        val_batch[f'aug{aux_idx + 1}_observations'] = np.asarray(
                            vqvae.encode(val_batch[f'aug{aux_idx + 1}_observations'], flatten=True)
                        )
                        val_batch[f'aug{aux_idx + 1}_next_observations'] = np.asarray(
                            vqvae.encode(val_batch[f'aug{aux_idx + 1}_next_observations'], flatten=True)
                        )
            _, val_info = loss_fn(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        # if (FLAGS.eval_interval != 0 and (i > FLAGS.pretraining_steps)
        #     and (i == (FLAGS.pretraining_steps + 1) or i % FLAGS.eval_interval == 0)):
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                vqvae=vqvae,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
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
