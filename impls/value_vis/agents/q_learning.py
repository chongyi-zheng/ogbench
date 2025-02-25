from typing import Any
import copy
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from impls.utils.evaluation import supply_rng
from impls.value_vis.utils import default_init, evaluate


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
            nn.Dense(self.num_actions, kernel_init=self.kernel_init),
        ])(observations)

        return q


def train_and_eval_q_learning(env, get_batch_fn, key,
                              discount=0.99, tau=0.005,
                              batch_size=256, learning_rate=3e-4,
                              num_training_steps=100_000, eval_interval=10_000):

    def critic_loss_fn(params, target_params, q_value_fn, batch):
        """Compute the Q-learning loss."""
        next_qs = q_value_fn.apply(target_params, batch['next_observations'])
        target_q = batch['rewards'] + discount * (1.0 - batch['terminals']) * jnp.max(next_qs, axis=-1)

        qs = q_value_fn.apply(params, batch['observations'])
        onehot_actions = jax.nn.one_hot(batch['actions'], q_value_fn.num_actions)
        q = jnp.sum(qs * onehot_actions, axis=-1)
        critic_loss = jnp.square(target_q - q).mean()

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
        info = dict()
        (_, critic_info), grads = critic_grad_fn(params, target_params, q_value_fn, batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

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
            train_k = 'train/' + k
            metric = np.array([[step, v]])
            if train_k not in metrics:
                metrics[train_k] = metric
            else:
                metrics[train_k] = np.concatenate([metrics[train_k], metric], axis=0)

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
