import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from policy_evaluation import estimators
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.wrappers import OfflineObservationNormalizer
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate_policy_evaluation, evaluate_heatmaps
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_heatmaps, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_transitions', 50_000, 'Number of transitions for evaluating accuracies.')
flags.DEFINE_integer('visualize_heatmaps', 8, 'Number of estimation heatmaps.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('estimator', 'policy_evaluation/gciql.py', lock_config=False)


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

    # Set up environment and dataset.
    config = FLAGS.estimator
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize estimator.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    estimator_class = estimators[config['estimator_name']]
    estimator = estimator_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if hasattr(config, 'normalize_observation') and config['normalize_observation']:
        estimator = OfflineObservationNormalizer.create(
            estimator,
            train_dataset
        )

    # Restore estimator.
    if FLAGS.restore_path is not None:
        estimator = restore_agent(estimator, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train estimator.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update estimator.
        batch = train_dataset.sample(config['batch_size'])
        estimator, update_info = estimator.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = estimator.total_loss(val_batch, grad_params=None)
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
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_estimator = jax.device_put(estimator, device=jax.devices('cpu')[0])
            else:
                eval_estimator = estimator

            dataset = val_dataset if val_dataset is not None else train_dataset
            metric_names = ['binary_q_acc', 'binary_v_acc']
            eval_metrics = {}
            eval_info = evaluate_policy_evaluation(eval_estimator, dataset)
            eval_metrics.update(
                {f'evaluation/{k}': v for k, v in eval_info.items() if k in metric_names}
            )

            # renders = []
            # eval_metrics = {}
            # overall_metrics = defaultdict(list)
            # task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            # num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            # for task_id in tqdm.trange(1, num_tasks + 1):
            #     task_name = task_infos[task_id - 1]['task_name']
            #     eval_info, trajs, cur_renders = evaluate_policy_evaluation(
            #         estimator=eval_estimator,
            #         env=env,
            #         task_id=task_id,
            #         config=config,
            #         num_eval_episodes=FLAGS.eval_episodes,
            #         num_video_episodes=FLAGS.video_episodes,
            #         video_frame_skip=FLAGS.video_frame_skip,
            #         eval_temperature=FLAGS.eval_temperature,
            #         eval_gaussian=FLAGS.eval_gaussian,
            #     )
            #     renders.extend(cur_renders)
            #     metric_names = ['success']
            #     eval_metrics.update(
            #         {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            #     )
            #     for k, v in eval_info.items():
            #         if k in metric_names:
            #             overall_metrics[k].append(v)
            # for k, v in overall_metrics.items():
            #     eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            # if FLAGS.video_episodes > 0:
            #     video = get_wandb_video(renders=renders, n_cols=num_tasks)
            #     eval_metrics['video'] = video
            if FLAGS.visualize_heatmaps > 0:
                value_info, env_info, = evaluate_heatmaps(
                    eval_estimator,
                    dataset,
                    env,
                    FLAGS.visualize_heatmaps,
                )

                heatmap = get_wandb_heatmaps(
                    value_info, env_info)

                eval_metrics['heatmap'] = heatmap

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(estimator, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)