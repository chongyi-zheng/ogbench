from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import numpy as np
import jax.numpy as jnp
import flax.linen as nn

import gymnasium as gym
import h5py
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from tqdm import tqdm


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def sinusoidal_pos_embedding(times, emb_dim=64):
    half_dim = emb_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = times * jnp.expand_dims(emb, axis=0)
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

    return emb


def collect_dataset(env_name, dataset_size=200_000, seed=None):
    env = gym.make(env_name)
    dataset = dict()
    size = 0
    if env_name == 'MountainCar-v0':
        options = dict(low=-0.6, high=0.4)
    else:
        options = dict()
    while size < dataset_size:
        observation, info = env.reset(seed=seed, options=options)
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # reward = 0
            # if terminated:
            #     reward = 100.0
            # reward -= np.power(action - 1, 2) * 0.1
            size += 1

            for k in ['observation', 'action', 'reward', 'next_observation', 'done']:
                if k not in dataset:
                    dataset[k] = np.expand_dims(locals()[k], axis=0)
                else:
                    dataset[k] = np.concatenate([
                        dataset[k],
                        np.expand_dims(locals()[k], axis=0)
                    ], axis=0)

            observation = next_observation

    return dataset


def save_dataset_to_h5(dataset, h5_path):
    with h5py.File(h5_path, 'w') as f:
        for k, v in dataset.items():
            f.create_dataset(k, data=v)


def load_dataset_from_h5(h5_path):
    def get_keys(h5file):
        """
        reference: https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/offline_env.py#L20-L28
        """
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset = {}
    with h5py.File(h5_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]

    return dataset


def preprocess_dataset(dataset):
    new_dataset = dict()
    for k in dataset.keys():
        if k != 'done':
            new_k = k + 's'
            new_dataset[new_k] = dataset[k]

    (terminal_locs,) = np.nonzero(dataset['done'] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])

    new_dataset['terminals'] = np.asarray(dataset['done'], dtype=int)
    new_dataset['terminal_locs'] = terminal_locs
    new_dataset['initial_locs'] = initial_locs

    return new_dataset


def plot_metrics(metrics, logyscale_stats=[], suptitle='learning_curve'):
    # learning curves
    nrows = np.ceil(len(metrics) / 4).astype(int)
    ncols = 4
    f, axes = plt.subplots(nrows=nrows, ncols=ncols)
    if nrows == 1:
        axes = np.array([axes])
    f.set_figheight(3 * nrows)
    f.set_figwidth(3 * ncols)

    for idx, (name, val) in enumerate(metrics.items()):
        v = np.array(val)
        if len(v) == 0:
            continue

        x, y = v[:, 0], v[:, 1]
        ax = axes[idx // 4, idx % 4]

        if 'train' in name:
            y = gaussian_filter1d(y, 100)
        ax.plot(x, y)
        if name in logyscale_stats:
            ax.set_yscale('log')
        ax.set_title(name)

        ax.grid()

    f.suptitle(suptitle)

    return f


def explore(actor_fn, env, params, temperature=1.0, num_expl_steps=2000, desc='evaluation'):
    stats, trajs = defaultdict(list), defaultdict(list)

    done, reward_sum, episode_length = False, 0, 0
    observation, info = env.reset()
    for _ in tqdm(range(num_expl_steps), desc=desc, position=0, leave=True):
        action = actor_fn(params=params, observations=observation, temperature=temperature)
        action = np.array(action)
        assert env.action_space.contains(action)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        reward_sum += reward
        episode_length += 1

        for k in ['observation', 'action', 'reward', 'next_observation', 'done']:
            trajs[k].append(locals()[k])

        if done:
            observation, info = env.reset()

            stats['episode_length'].append(episode_length)
            stats['episode_return'].append(reward_sum)

            episode_length = 0
            reward_sum = 0.0
        else:
            observation = next_observation

    for k, v in stats.items():
        stats[k] = np.mean(v)

    episodes = dict()
    for k, v in trajs.items():
        if k != 'done':
            episodes[k] = np.asarray(v)
        else:
            episodes['terminal'] = np.asarray(v, dtype=int)
    episodes['terminal'][-1] = 1.0

    return stats, episodes

    # env = gym.make(env_name)
    # dataset = dict()
    # size = 0
    # if env_name == 'MountainCar-v0':
    #     options = dict(low=-0.6, high=0.4)
    # else:
    #     options = dict()
    # while size < dataset_size:
    #     observation, info = env.reset(seed=seed, options=options)
    #     done = False
    #     while not done:
    #         action = env.action_space.sample()
    #         next_observation, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         # reward = 0
    #         # if terminated:
    #         #     reward = 100.0
    #         # reward -= np.power(action - 1, 2) * 0.1
    #         size += 1
    #
    #         for k in ['observation', 'action', 'reward', 'next_observation', 'done']:
    #             if k not in dataset:
    #                 dataset[k] = jnp.expand_dims(locals()[k], axis=0)
    #             else:
    #                 dataset[k] = np.concatenate([
    #                     dataset[k],
    #                     jnp.expand_dims(locals()[k], axis=0)
    #                 ], axis=0)
    #
    #         observation = next_observation
    #
    # return dataset


def evaluate(actor_fn, env, params, temperature=0.0, num_eval_episodes=20, desc='evaluation'):
    stats = defaultdict(list)
    if isinstance(env.unwrapped, MountainCarEnv):
        options = dict(low=-0.6, high=0.4)
    else:
        options = dict()
    for _ in tqdm(range(num_eval_episodes), desc=desc, position=0, leave=True):
        observation, info = env.reset(options=options)
        done = False
        episode_length = 0
        reward_sum = 0.0

        while not done:
            action = actor_fn(params=params, observations=observation, temperature=temperature)
            action = np.array(action)
            assert env.action_space.contains(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # reward = 0
            # if terminated:
            #     reward = 100.0
            # reward -= np.power(action - 1, 2) * 0.1
            # if terminated:
            #     reward = 0.0
            # done = truncated

            reward_sum += reward
            episode_length += 1

            observation = next_observation

        stats['episode_length'].append(episode_length)
        stats['episode_return'].append(reward_sum)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
