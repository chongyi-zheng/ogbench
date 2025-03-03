import copy
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

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


class Reward(nn.Module):
    num_actions: int
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations):
        rewards = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(self.num_actions, kernel_init=self.kernel_init),
        ])(observations)

        return rewards


class VectorField(nn.Module):
    vector_dim: int
    num_actions: int
    time_dim: int = 64
    kernel_init: Any = default_init()

    @nn.compact
    def sinusoidal_pos_embedding(self, times):
        half_dim = self.time_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = times * jnp.expand_dims(emb, axis=0)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        return emb

    @nn.compact
    def __call__(self, noisy_goals, times, observations, actions):
        onehot_actions = jax.nn.one_hot(actions, self.num_actions)
        time_embs = self.sinusoidal_pos_embedding(times)
        # time_embs = times

        vector_field = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(self.vector_dim * self.num_actions, kernel_init=self.kernel_init),
        ])(jnp.concatenate([noisy_goals, time_embs, observations], axis=-1))
        vector_field = jnp.reshape(vector_field, [-1, self.vector_dim, self.num_actions])
        vector_field = jnp.einsum('ijk,ik->ij', vector_field, onehot_actions)

        return vector_field


def train_and_eval_mcfac(env, get_batch_fn, key,
                         num_flow_steps=10, expectile=0.95,
                         discount=0.99, batch_size=256, learning_rate=3e-4,
                         num_training_steps=100_000, eval_interval=10_000):

    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def reward_loss_fn(params, reward_fn, batch):
        """Compute the reward loss."""
        reward_preds = reward_fn.apply(params, batch['observations'])
        onehot_actions = jax.nn.one_hot(batch['actions'], reward_fn.num_actions)
        reward_preds = jnp.sum(reward_preds * onehot_actions, axis=-1)

        reward_loss = jnp.square(batch['rewards'] - reward_preds).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def flow_matching_loss_fn(params, vector_field_fn, batch, key):
        """Compute the flow matching loss."""

        observations = batch['observations']
        actions = batch['actions']
        goals = batch['future_observations']
        # onehot_actions = jax.nn.one_hot(batch['actions'], vector_field_fn.num_actions)

        key, noise_key, time_key = jax.random.split(key, 3)
        noises = jax.random.normal(noise_key, shape=goals.shape, dtype=goals.dtype)
        times = jax.random.uniform(time_key, shape=(batch_size, 1))
        noisy_goals = times * goals + (1 - times) * noises
        vector_field_targets = goals - noises
        vector_field_preds = vector_field_fn.apply(
            params, noisy_goals, times, observations, actions)
        # vector_field_preds = jnp.einsum('ijk,ik->ij', vector_field_preds, onehot_actions)
        flow_matching_loss = jnp.square(vector_field_targets - vector_field_preds).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss
        }

    def compute_fwd_flow_goals(params, vector_field_fn, noises, observations, actions):
        """Compute the forward flow goals using Euler method."""
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, step):
            """
            carry: (noisy_goals, )
            step: current step index
            """
            (noisy_goals, ) = carry

            times = jnp.full((*noisy_goals.shape[:-1], 1), step * step_size)
            vector_field = vector_field_fn.apply(
                params, noisy_goals, times, observations, actions)
            # vector_field = jnp.einsum('ijk,ik->ij', vector_field, onehot_actions)
            new_noisy_goals = noisy_goals + vector_field * step_size

            return (new_noisy_goals, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, ), _ = jax.lax.scan(
            body_fn, (noises, ), jnp.arange(num_flow_steps))

        return noisy_goals

    def critic_loss_fn(q_params, reward_params, vf_params,
                       q_value_fn, reward_fn, vector_field_fn,
                       batch, key):
        """Compute the critic loss."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        onehot_actions = jax.nn.one_hot(batch['actions'], q_value_fn.num_actions)

        qs = q_value_fn.apply(q_params, observations)
        q = jnp.sum(qs * onehot_actions, axis=-1)

        key, noise_key = jax.random.split(key)
        noises = jax.random.normal(noise_key, shape=observations.shape, dtype=observations.dtype)
        flow_goals = compute_fwd_flow_goals(vf_params, vector_field_fn, noises, observations, actions)

        future_actions = q_value_fn.apply(q_params, flow_goals).argmax(axis=-1)
        future_rewards = reward_fn.apply(reward_params, flow_goals)
        onehot_future_actions = jax.nn.one_hot(future_actions, reward_fn.num_actions)
        future_rewards = jnp.sum(onehot_future_actions * future_rewards, axis=-1)

        target_q = rewards + discount / (1 - discount) * future_rewards
        critic_loss = expectile_loss(
            target_q - q, target_q - q, expectile).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    key, time_key, q_value_key, reward_key, vector_field_key = jax.random.split(key, 5)
    example_batch = get_batch_fn(2)
    ex_times = jax.random.uniform(time_key, shape=(2, 1))

    q_value_fn = QValue(env.action_space.n)
    reward_fn = Reward(env.action_space.n)
    vector_field_fn = VectorField(env.observation_space.shape[0],
                                  env.action_space.n)

    q_params = q_value_fn.init(
        q_value_key, example_batch['observations'])
    # target_q_params = copy.deepcopy(q_params)
    reward_params = reward_fn.init(
        reward_key, example_batch['observations']
    )
    vf_params = vector_field_fn.init(
        vector_field_key,
        example_batch['future_observations'], ex_times,
        example_batch['observations'], example_batch['actions']
    )

    q_optimizer = optax.adam(learning_rate=learning_rate)
    reward_optimizer = optax.adam(learning_rate=learning_rate)
    vf_optimizer = optax.adam(learning_rate=learning_rate)
    q_opt_state = q_optimizer.init(q_params)
    reward_opt_state = reward_optimizer.init(reward_params)
    vf_opt_state = vf_optimizer.init(vf_params)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    reward_grad_fn = jax.value_and_grad(reward_loss_fn, has_aux=True)
    flow_matching_grad_fn = jax.value_and_grad(flow_matching_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(q_params, reward_params, vf_params,
                  q_opt_state, reward_opt_state, vf_opt_state,
                  batch, key):
        key, flow_matching_key, critic_key = jax.random.split(key, 3)
        info = dict()

        (_, reward_info), reward_grads = reward_grad_fn(
            reward_params, reward_fn, batch
        )
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        (_, flow_matching_info), flow_matching_grads = flow_matching_grad_fn(
            vf_params, vector_field_fn, batch, flow_matching_key
        )
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        (_, critic_info), critic_grads = critic_grad_fn(
            q_params, reward_params, vf_params,
            q_value_fn, reward_fn, vector_field_fn,
            batch, critic_key
        )
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        reward_updates, reward_opt_state = reward_optimizer.update(
            reward_grads, reward_opt_state)
        reward_params = optax.apply_updates(reward_params, reward_updates)
        vf_updates, vf_opt_state = vf_optimizer.update(
            flow_matching_grads, vf_opt_state)
        vf_params = optax.apply_updates(vf_params, vf_updates)
        q_updates, q_opt_state = q_optimizer.update(
            critic_grads, q_opt_state)
        q_params = optax.apply_updates(q_params, q_updates)

        return (q_params, reward_params, vf_params,
                q_opt_state, reward_opt_state, vf_opt_state, info)

    @jax.jit
    def sample_actions(params, observations, seed=None):
        q = q_value_fn.apply(params, observations)
        actions = q.argmax(axis=-1)

        return actions

    key, eval_key = jax.random.split(key)
    actor_fn = supply_rng(sample_actions, rng=eval_key)

    metrics = dict()
    for step in tqdm(range(num_training_steps + 1), desc="mcfac training"):
        key, train_key = jax.random.split(key)

        batch = get_batch_fn(batch_size)
        (q_params, reward_params, vf_params,
         q_opt_state, reward_opt_state, vf_opt_state, info) = update_fn(
            q_params, reward_params, vf_params,
            q_opt_state, reward_opt_state, vf_opt_state,
            batch, train_key
        )

        for k, v in info.items():
            train_k = 'train/' + k
            metric = np.array([[step, v]])
            if train_k not in metrics:
                metrics[train_k] = metric
            else:
                metrics[train_k] = np.concatenate([metrics[train_k], metric], axis=0)

        if step % eval_interval == 0:
            eval_info = evaluate(actor_fn, env, q_params, desc='mcfac evaluation')

            for k, v in eval_info.items():
                eval_k = 'eval/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

    return metrics
