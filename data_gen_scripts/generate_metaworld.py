import os
import glob
import json
from collections import defaultdict

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

import numpy as np
from absl import app, flags
from agents import SACAgent
import imageio
from tqdm import trange
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent



FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-v0', 'Environment name.')
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('num_video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('max_episode_steps', 501, 'Maximum number of steps in an episode.')


def main(_):
    # Restore agent config.
    restore_path = FLAGS.restore_path
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    with open(candidates[0] + '/flags.json', 'r') as f:
        config = json.load(f)
        training_seed = config['seed']
        agent_config = config['agent']

    # Create environment.
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env_name + '-goal-observable']
    # Must use the same seed during training.
    env = env_cls(seed=training_seed, render_mode='rgb_array')
    env.max_episode_steps = FLAGS.max_episode_steps
    # set the rendering resolution
    env.width, env.height = 200, 200
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 200, 200
    ob_dim = env.observation_space.shape[0]

    # Load agent.
    agent = SACAgent.create(
        FLAGS.seed,
        np.zeros(ob_dim),
        env.action_space.sample(),
        agent_config,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    # Collect data.
    dataset = defaultdict(list)
    stats = defaultdict(list)
    renders = []
    total_steps = 0
    total_train_steps = 0
    total_val_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    num_video_episodes = FLAGS.num_video_episodes
    for ep_idx in trange(num_train_episodes + num_val_episodes + num_video_episodes):
        should_render = ep_idx >= (num_train_episodes + num_val_episodes)

        ob, _ = env.reset()

        done = False
        step = 0
        success = []
        render = []

        while not done:
            action = actor_fn(ob, temperature=0)
            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if should_render and (step % FLAGS.video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            # masks denotes whether the agent should get a Bellman backup from the next observation. It is 0 only when the task is complete (and 1 otherwise). In this case, the agent should set the target Q-value to 0, instead of using the next observation's target Q-value.
            # terminals simply denotes whether the dataset trajectory is over, regardless of task completion.
            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(truncated)
            dataset['masks'].append(info['success'])
            # dataset['qpos'].append(info['prev_qpos'])
            # dataset['qvel'].append(info['prev_qvel'])

            success.append(info['success'])

            ob = next_ob
            step += 1

        stats['success'].append(success)
        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step
        elif num_train_episodes <= ep_idx < (num_train_episodes + num_val_episodes):
            total_val_steps += step
        else:
            renders.append(np.array(render))

    success_rate = np.mean(np.any(np.array(stats['success']) > 0, axis=-1))

    print('Success rate:', success_rate)
    print('Total steps:', total_steps)

    FLAGS.save_path = os.path.expanduser(FLAGS.save_path)
    if num_video_episodes > 0:
        video_dir = os.path.dirname(FLAGS.save_path)
        os.makedirs(video_dir, exist_ok=True)
        video_path = FLAGS.save_path.replace('.npz', '.mp4')
        renders = np.asarray(renders)
        renders = renders.reshape([-1, *renders.shape[2:]])
        imageio.mimsave(video_path, renders, fps=15)
        print("Save video to {}".format(video_path))

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k in ['terminals', 'masks']:
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:total_train_steps + total_val_steps], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
