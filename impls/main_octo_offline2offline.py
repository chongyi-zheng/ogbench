import json
import os
import random
import time

import numpy as np
import simpler_env
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    create_validation_dataset
)
from octo.utils.train_utils import (
    filter_eval_datasets,
    process_text,
)

from octo_agents import agents
from octo_utils.wrappers import SimplerOctoWrapper
from utils.env_utils import EpisodeMonitor
from utils.evaluation import evaluate_octo
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

flags.DEFINE_integer('pretraining_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('finetuning_steps', 500_000, 'Number of online steps.')
flags.DEFINE_integer('finetuning_size', 500_000, 'Size of the dataset for finetuning.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1_500_000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file(
    'octo',
    'octo_utils/config.py',
    "File path to the octo hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    'agent',
    'octo_agents/iql.py',
    "File path to the agent hyperparameter configuration.",
    lock_config=False,
)


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

        codebase_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        wandb.run.log_code(codebase_directory)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], 'GPU')

    # Make datasets and environment.
    octo_config = FLAGS.octo
    config = FLAGS.agent
    # _, eval_env, train_dataset, val_dataset = make_env_and_datasets(
    #     FLAGS.env_name, frame_stack=FLAGS.frame_stack, action_clip_eps=None)
    # eval_env = simpler_env_utils.make_env_and_datasets(FLAGS.env_name, env_only=True)

    # Initialize agent.
    octo_config.seed = FLAGS.seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    # set up text tokenization (this needs to happen after batching but before sharding)
    if octo_config.text_processor is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(octo_config.text_processor)()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch['dataset_name']
        return batch

    # load datasets
    assert 'oxe_kwargs' in octo_config.dataset_kwargs
    # create dataset_kwargs_list from oxe_kwargs
    (
        octo_config.dataset_kwargs['dataset_kwargs_list'],
        octo_config.dataset_kwargs['sample_weights'],
    ) = make_oxe_dataset_kwargs_and_weights(
        **octo_config.dataset_kwargs['oxe_kwargs']
    )
    del octo_config.dataset_kwargs['oxe_kwargs']

    # TODO (chongyi): use different datasets for pretraining and finetuning.
    pretraining_train_data = make_interleaved_dataset(**octo_config.dataset_kwargs, train=True)
    pretraining_train_data_iter = map(
        process_batch,
        pretraining_train_data.iterator(prefetch=octo_config.prefetch_num_batches),
    )
    finetuning_train_data = make_interleaved_dataset(**octo_config.dataset_kwargs, train=True)
    finetuning_train_data_iter = map(
        process_batch,
        finetuning_train_data.iterator(prefetch=octo_config.prefetch_num_batches),
    )

    val_datasets_kwargs_list, _ = filter_eval_datasets(
        octo_config.dataset_kwargs['dataset_kwargs_list'],
        octo_config.dataset_kwargs['sample_weights'],
        octo_config.eval_datasets,
    )
    assert len(val_datasets_kwargs_list) == 1

    val_dataset_kwargs = val_datasets_kwargs_list[0]
    pretraining_val_data = create_validation_dataset(
        val_dataset_kwargs,
        octo_config.dataset_kwargs['traj_transform_kwargs'],
        octo_config.dataset_kwargs['frame_transform_kwargs'],
    )
    pretraining_val_iterator = (
        pretraining_val_data.unbatch()
        .shuffle(octo_config.val_kwargs['val_shuffle_buffer_size'])
        .repeat()
        .batch(octo_config.dataset_kwargs['batch_size'])
        .iterator(prefetch=0)
    )
    pretraining_val_data_iter = map(process_batch, pretraining_val_iterator)
    finetuning_val_data = create_validation_dataset(
        val_dataset_kwargs,
        octo_config.dataset_kwargs['traj_transform_kwargs'],
        octo_config.dataset_kwargs['frame_transform_kwargs'],
    )
    finetuning_val_iterator = (
        finetuning_val_data.unbatch()
        .shuffle(octo_config.val_kwargs['val_shuffle_buffer_size'])
        .repeat()
        .batch(octo_config.dataset_kwargs['batch_size'])
        .iterator(prefetch=0)
    )
    finetuning_val_data_iter = map(process_batch, finetuning_val_iterator)

    example_batch = next(pretraining_train_data_iter)

    eval_env = simpler_env.make(FLAGS.env_name)
    eval_env = SimplerOctoWrapper(
        eval_env,
        example_batch=example_batch,
        # TODO (chongyiz): avoid hardcoding the dataset name.
        unnormalization_statistics=pretraining_train_data.dataset_statistics['fractal20220817_data']['action'],
        text_processor=text_processor,
        window_size=octo_config['window_size'],
        pred_action_horizon=4,
    )
    eval_env = EpisodeMonitor(eval_env)

    # Create agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
        octo_config,
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
    expl_metrics = dict()
    for i in tqdm.tqdm(range(1, FLAGS.pretraining_steps + FLAGS.finetuning_steps + 1), smoothing=0.1,
                       dynamic_ncols=True):
        if i <= FLAGS.pretraining_steps:
            # Offline pre-training.
            batch = next(pretraining_train_data_iter)
            train_logger = pretraining_train_logger
            eval_logger = pretraining_eval_logger

            agent, update_info = agent.pretrain(batch)
        else:
            # Offline fine-tuning.
            batch = next(finetuning_train_data_iter)
            train_logger = finetuning_train_logger
            eval_logger = finetuning_eval_logger

            agent, update_info = agent.finetune(batch, full_update=(i % config['actor_freq'] == 0))

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if i <= FLAGS.pretraining_steps:
                val_data_iter = pretraining_val_data_iter
                loss_fn = agent.pretraining_loss
            else:
                val_data_iter = finetuning_val_data_iter
                loss_fn = agent.total_loss
            val_batch = next(val_data_iter)
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
        # if (FLAGS.eval_interval != 0 and (i > FLAGS.pretraining_steps)
        #         and (i == (FLAGS.pretraining_steps + 1) or i % FLAGS.eval_interval == 0)):
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate_octo(
                agent=agent,
                env=eval_env,
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
