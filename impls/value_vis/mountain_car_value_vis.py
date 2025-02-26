from typing import Any
import copy
import os.path as osp
import functools
from collections import defaultdict

import h5py
from tqdm import tqdm

# import math
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import tqdm
# from IPython import display
# from scipy.ndimage import gaussian_filter1d
import numpy as np
import gymnasium as gym

import jax
import jax.numpy as jnp
import distrax
import flax
import flax.linen as nn
import optax

from impls.value_vis.agents import (
    train_and_eval_q_learning,
    train_and_eval_td3bc,
    train_and_eval_mcfac,
)


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
                    dataset[k] = jnp.expand_dims(locals()[k], axis=0)
                else:
                    dataset[k] = np.concatenate([
                        dataset[k],
                        jnp.expand_dims(locals()[k], axis=0)
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


def get_batch(dataset, batch_size, discount=0.99):
    dataset_size = dataset['observations'].shape[0]
    terminal_locs = dataset['terminal_locs']

    idxs = np.random.randint(dataset_size, size=batch_size)
    batch = jax.tree_util.tree_map(lambda arr: arr[np.minimum(idxs, arr.shape[0] - 1)], dataset)

    # sample future observation from truncated geometric distribution
    final_obs_idxs = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    offsets = np.random.geometric(p=1 - discount, size=(batch_size, ))  # in [1, inf)
    future_obs_idxs = np.minimum(idxs + offsets, final_obs_idxs)
    batch['future_observations'] = jax.tree_util.tree_map(
        lambda arr: arr[future_obs_idxs], dataset['observations'])

    del batch['terminal_locs']
    del batch['initial_locs']

    return batch


def main():
    discount = 0.99
    env_name = 'MountainCarContinuous-v0'
    dataset_path = '~/research/ogbench/impls/value_vis/mountain_car_continous.hdf5'
    dataset_path = osp.expanduser(dataset_path)
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))

    if osp.exists(dataset_path):
        dataset = load_dataset_from_h5(dataset_path)
        print("Load {} dataset from: {}".format(env_name, dataset_path))
    else:
        dataset = collect_dataset(env_name)
        save_dataset_to_h5(dataset, dataset_path)
        print("Save {} dataset to: {}".format(env_name, dataset_path))

    env = gym.make(env_name)
    dataset = preprocess_dataset(dataset)
    get_batch_fn = functools.partial(get_batch, dataset, discount=discount)

    # key, q_learning_key = jax.random.split(key)
    # metrics = train_and_eval_q_learning(env, get_batch_fn, q_learning_key)

    key, td3_key = jax.random.split(key)
    metrics = train_and_eval_td3bc(env, get_batch_fn, td3_key)

    # key, mcfac_key = jax.random.split(key)
    # metrics = train_and_eval_mcfac(env, get_batch_fn, mcfac_key)

    print(metrics['eval/episode_length'])
    print(metrics['eval/episode_return'])


if __name__ == "__main__":
    main()
