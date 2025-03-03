import copy
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from impls.utils.evaluation import supply_rng
from impls.utils.networks import ensemblize
from impls.value_vis.utils import default_init, evaluate


class QValue(nn.Module):
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations, actions):
        q = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(1, kernel_init=self.kernel_init),
        ])(jnp.concatenate([observations, actions], axis=-1))

        q = q.squeeze(axis=-1)

        return q


class Actor(nn.Module):
    action_dim: int
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations):
        actions = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=self.kernel_init),
        ])(observations)

        return actions


def train_and_eval_td3bc(env, get_batch_fn, key,
                         discount=0.99, tau=0.005,
                         noise_clip=0.15, num_ensembles=2, alpha=0.3,
                         batch_size=256, learning_rate=3e-4,
                         num_training_steps=50_000, eval_interval=10_000):
    def actor_loss_fn(actor_params, q_params,
                      actor_fn, q_value_fn, batch):
        """Compute the actor loss."""
        q_actions = actor_fn.apply(actor_params, batch['observations'])
        q_actions = jnp.clip(q_actions, -1.0, 1.0)

        # Compute actor loss
        qs = q_value_fn.apply(q_params, batch['observations'], q_actions)
        q = qs.min(axis=0)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        q_loss = -q.mean()
        bc_loss = alpha * jnp.square(batch['actions'] - q_actions).mean()
        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
        }

    def critic_loss_fn(q_params, target_q_params, actor_params,
                       q_value_fn, actor_fn, batch, key):
        """Compute the Q-learning loss."""
        key, noise_key = jax.random.split(key)

        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']

        noises = jax.random.normal(noise_key, shape=actions.shape, dtype=actions.dtype)
        noises = jnp.clip(noises, -noise_clip, noise_clip)
        next_actions = actor_fn.apply(actor_params, next_observations)
        next_actions += noises
        next_actions = jnp.clip(next_actions, -1.0, 1.0)

        next_qs = q_value_fn.apply(target_q_params, next_observations, next_actions)
        next_q = next_qs.min(axis=0)
        target_q = batch['rewards'] + discount * (1.0 - batch['terminals']) * next_q

        qs = q_value_fn.apply(q_params, observations, actions)
        critic_loss = jnp.square(target_q - qs).mean()

        # for logging
        q = qs.min(axis=0)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    key, q_value_key, actor_key = jax.random.split(key, 3)
    example_batch = get_batch_fn(2)

    if num_ensembles > 1:
        q_value_fn = ensemblize(QValue, num_ensembles)()
    else:
        q_value_fn = QValue()
    actor_fn = Actor(env.action_space.shape[0])

    q_params = q_value_fn.init(
        q_value_key, example_batch['observations'], example_batch['actions'])
    target_q_params = copy.deepcopy(q_params)
    actor_params = actor_fn.init(
        actor_key, example_batch['observations'])

    q_optimizer = optax.adam(learning_rate=learning_rate)
    q_opt_state = q_optimizer.init(q_params)
    actor_optimizer = optax.adam(learning_rate=learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(q_params, target_q_params, actor_params,
                  q_opt_state, actor_opt_state, batch, key):
        key, critic_key = jax.random.split(key)
        info = dict()
        (_, critic_info), critic_grads = critic_grad_fn(
            q_params, target_q_params, actor_params, q_value_fn, actor_fn, batch, critic_key)

        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        (_, actor_info), actor_grads = actor_grad_fn(
            actor_params, q_params, actor_fn, q_value_fn, batch)

        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        q_updates, q_opt_state = q_optimizer.update(
            critic_grads, q_opt_state)
        q_params = optax.apply_updates(q_params, q_updates)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            target_q_params, q_params)

        return (q_params, target_q_params, actor_params,
                q_opt_state, actor_opt_state, info)

    @jax.jit
    def sample_actions(params, observations, seed=None, temperature=1.0):
        actions = actor_fn.apply(params, observations)
        actions = jnp.clip(actions, -1, 1)

        return actions

    key, eval_key = jax.random.split(key)
    sample_action_fn = supply_rng(sample_actions, rng=eval_key)

    metrics = dict()
    for step in tqdm(range(num_training_steps + 1), desc="td3 + bc training"):
        key, train_key = jax.random.split(key)

        batch = get_batch_fn(batch_size)
        (q_params, target_q_params, actor_params,
         q_opt_state, actor_opt_state, info) = update_fn(
            q_params, target_q_params, actor_params,
            q_opt_state, actor_opt_state, batch, train_key)

        for k, v in info.items():
            train_k = 'train/' + k
            metric = np.array([[step, v]])
            if train_k not in metrics:
                metrics[train_k] = metric
            else:
                metrics[train_k] = np.concatenate([metrics[train_k], metric], axis=0)

        if step % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_info = evaluate(sample_action_fn, env, actor_params, desc='td3 + bc evaluation')

            for k, v in eval_info.items():
                eval_k = 'eval/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

    return metrics
