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

from impls.value_vis.utils import (
    collect_dataset,
    load_dataset_from_h5,
    save_dataset_to_h5,
    preprocess_dataset
)

from impls.value_vis.agents import (
    train_and_eval_td3bc,
    train_and_eval_mcfac,
)


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
    dataset_path = '~/research/ogbench/impls/value_vis/mountain_car_continuous.hdf5'
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

    # key, td3_key = jax.random.split(key)
    # metrics = train_and_eval_td3bc(env, get_batch_fn, td3_key)

    key, mcfac_key = jax.random.split(key)
    metrics = train_and_eval_mcfac(env, get_batch_fn, mcfac_key)

    print(metrics['eval/episode_length'])
    print(metrics['eval/episode_return'])


if __name__ == "__main__":
    main()
