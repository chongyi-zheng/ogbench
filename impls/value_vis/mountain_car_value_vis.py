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

from impls.utils.evaluation import supply_rng


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def collect_dataset(env_name, dataset_size=200_000, seed=None):
    env = gym.make(env_name)
    dataset = dict()
    size = 0
    while size < dataset_size:
        observation, info = env.reset(seed=seed)
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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


def evaluate(actor_fn, env, params, num_eval_episodes=20, desc='evaluation'):
    stats = defaultdict(list)
    for _ in tqdm(range(num_eval_episodes), desc=desc, position=0, leave=True):
        observation, info = env.reset()
        done = False
        episode_length = 0
        reward_sum = 0.0

        while not done:
            action = actor_fn(params=params, observations=observation)
            action = np.array(action)
            assert env.action_space.contains(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward_sum += reward
            episode_length += 1

            observation = next_observation

        stats['episode_length'].append(episode_length)
        stats['episode_return'].append(reward_sum)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


def train_and_eval_q_learning(env, get_batch_fn, key,
                              discount=0.99, tau=0.005,
                              batch_size=256, learning_rate=3e-4,
                              num_training_steps=100_000, eval_interval=10_000):
    class QValue(nn.Module):
        num_actions: int
        kernel_init: Any = default_init()

        @nn.compact
        def __call__(self, observations):
            q = nn.Sequential([
                nn.Dense(512, kernel_init=self.kernel_init),
                nn.relu,
                nn.Dense(512, kernel_init=self.kernel_init),
                nn.relu,
                nn.Dense(512, kernel_init=self.kernel_init),
                nn.relu,
                nn.Dense(512, kernel_init=self.kernel_init),
                nn.relu,
                nn.Dense(self.num_actions, kernel_init=self.kernel_init),
            ])(observations)

            return q

    def critic_loss_fn(params, target_params, q_value_fn, batch):
        """Compute the Q-learning loss."""
        next_qs = q_value_fn.apply(target_params, batch['next_observations'])
        target_q = batch['rewards'] + discount * (1.0 - batch['terminals']) * jnp.max(next_qs, axis=-1)

        qs = q_value_fn.apply(params, batch['observations'])
        onehot_actions = jax.nn.one_hot(batch['actions'], q_value_fn.num_actions)
        q = jnp.sum(qs * onehot_actions, axis=-1)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    key, q_value_key = jax.random.split(key)
    example_batch = get_batch_fn(2)
    q_value_fn = QValue(env.action_space.n)
    q_params = q_value_fn.init(
        q_value_key, example_batch['observations'])
    target_q_params = copy.deepcopy(q_params)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(q_params)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(params, target_params, opt_state, batch):
        (_, info), grads = critic_grad_fn(params, target_params, q_value_fn, batch)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        target_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            target_params, params)

        return params, target_params, opt_state, info

    @jax.jit
    def sample_actions(params, observations, seed=None):
        q = q_value_fn.apply(params, observations)
        actions = q.argmax(axis=-1)

        return actions

    key, eval_key = jax.random.split(key)
    actor_fn = supply_rng(sample_actions, rng=eval_key)

    metrics = dict()
    for step in tqdm(range(num_training_steps + 1), desc="q_learning training"):
        batch = get_batch_fn(batch_size)
        q_params, target_q_params, opt_state, info = update_fn(
            q_params, target_q_params, opt_state, batch)

        for k, v in info.items():
            critic_k = 'train/critic' + k
            metric = np.array([[step, v]])
            if critic_k not in metrics:
                metrics[critic_k] = metric
            else:
                metrics[critic_k] = np.concatenate([metrics[critic_k], metric], axis=0)

        if step % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_info = evaluate(actor_fn, env, q_params, desc='q_learning evaluation')

            for k, v in eval_info.items():
                eval_k = 'eval/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

    return metrics


def main():
    discount = 0.99
    env_name = 'MountainCar-v0'
    dataset_path = '~/research/ogbench/impls/value_vis/mountain_car.hdf5'
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

    key, q_learning_key = jax.random.split(key)
    metrics = train_and_eval_q_learning(env, get_batch_fn, q_learning_key)

    print()


if __name__ == "__main__":
    main()
