import json
import os
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from online_env_utils import make_online_env
from utils.datasets import ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from viz_utils import visualize_trajs

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'online-ant-xy-v0', 'Environment name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('seed_steps', 10000, 'Number of seed steps.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('train_interval', 1, 'Train interval.')
flags.DEFINE_integer('num_epochs', 1, 'Number of updates per train interval.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_integer('reset_interval', 0, 'Full parameter reset interval.')
flags.DEFINE_integer('save_final_replay_buffer', 0, 'Whether to save the final replay buffer.')
flags.DEFINE_integer('terminate_at_end', 1, 'Whether to set terminated=True when truncated=True.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', '../impls/agents/sac.py', lock_config=False)


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

    config = FLAGS.agent

    # Set up environments and replay buffer.
    env = make_online_env(FLAGS.env_name, FLAGS.seed)
    eval_env = make_online_env(FLAGS.env_name, FLAGS.seed)

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        terminals=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_transition['observations'],
        example_transition['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    expl_metrics = dict()
    expl_rng = jax.random.PRNGKey(FLAGS.seed)
    ob, _ = env.reset()

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    update_info = None
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample transition.
        if i < FLAGS.seed_steps:
            action = env.action_space.sample()
        else:
            expl_rng, key = jax.random.split(expl_rng)
            action = agent.sample_actions(observations=ob, seed=key)

        action = np.array(action)
        next_ob, reward, terminated, truncated, info = env.step(action)
        if FLAGS.terminate_at_end and truncated:
            terminated = True

        replay_buffer.add_transition(
            dict(
                observations=ob,
                actions=action,
                rewards=reward,
                terminals=float(terminated),
                masks=float(not terminated),
                next_observations=next_ob,
            )
        )
        ob = next_ob

        if terminated or truncated:
            expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}
            ob, _ = env.reset()

        if replay_buffer.size < FLAGS.seed_steps:
            continue

        # Update agent.
        if i % FLAGS.train_interval == 0:
            for _ in range(FLAGS.num_epochs):
                batch = replay_buffer.sample(config['batch_size'])
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0 and update_info is not None:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
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
        if i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                dataset=replay_buffer,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)

            eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            traj_image = visualize_trajs(FLAGS.env_name, trajs)
            if traj_image is not None:
                eval_metrics['traj'] = wandb.Image(traj_image)

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        # Reset agent.
        if FLAGS.reset_interval > 0 and i % FLAGS.reset_interval == 0:
            new_agent = agent_class.create(
                FLAGS.seed + i,
                example_transition['observations'],
                example_transition['actions'],
                config,
            )
            agent = agent.replace(
                network=agent.network.replace(params=new_agent.network.params, opt_state=new_agent.network.opt_state)
            )
            del new_agent

    if FLAGS.save_final_replay_buffer > 0:
        # for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        #     np.savez_compressed(path, **dataset)
        save_path = os.path.join(FLAGS.save_dir, 'final_replay_buffer.npz')
        rb_dataset = replay_buffer._dict
        for k, v in rb_dataset.items():
            rb_dataset[k] = v[:replay_buffer.size]
        del rb_dataset['masks']
        del rb_dataset['rewards']
        np.savez_compressed(save_path, rb_dataset)
        print(f'Saved replay buffer to {save_path}')

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
