import copy
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from impls.utils.networks import ensemblize
from impls.utils.evaluation import supply_rng
from impls.value_vis.utils import (
    default_init,
    evaluate,
    sinusoidal_pos_embedding,
)


class Reward(nn.Module):
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations, actions):
        reward = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(1, kernel_init=self.kernel_init),
        ])(jnp.concatenate([observations, actions], axis=-1))

        reward = reward.squeeze(-1)

        return reward


class Critic(nn.Module):
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations, actions):
        q = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(1, kernel_init=self.kernel_init),
        ])(jnp.concatenate([observations, actions], axis=-1))

        q = q.squeeze(-1)

        return q


class CriticVectorField(nn.Module):
    vector_dim: int
    time_dim: int = 64
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, noisy_goals, times, observations, actions):
        time_embs = sinusoidal_pos_embedding(times, self.time_dim)
        # time_embs = times

        vector_field = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(self.vector_dim, kernel_init=self.kernel_init),
        ])(jnp.concatenate([noisy_goals, time_embs, observations, actions], axis=-1))

        return vector_field


class Actor(nn.Module):
    action_dim: int
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, noises, observations):
        actions = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(1, kernel_init=self.kernel_init),
        ])(jnp.concatenate([noises, observations], axis=-1))

        return actions


class ActorVectorField(nn.Module):
    vector_dim: int
    time_dim: int = 64
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, noisy_actions, times, observations):
        time_embs = sinusoidal_pos_embedding(times, self.time_dim)
        # time_embs = times

        vector_field = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.gelu,
            nn.Dense(self.vector_dim, kernel_init=self.kernel_init),
        ])(jnp.concatenate([noisy_actions, time_embs, observations], axis=-1))

        return vector_field


def train_and_eval_mcfac(env, get_batch_fn, key,
                         num_flow_steps=20, expectile=0.9,
                         alpha=1.0, tau=0.005, num_ensembles=2,
                         discount=0.99, batch_size=256, learning_rate=3e-4,
                         num_training_steps=100_000, eval_interval=10_000):

    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

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
            new_noisy_goals = noisy_goals + vector_field * step_size

            return (new_noisy_goals, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, ), _ = jax.lax.scan(
            body_fn, (noises, ), jnp.arange(num_flow_steps))

        return noisy_goals

    def compute_fwd_flow_actions(params, vector_field_fn, noises, observations):
        """Compute the forward flow actions using Euler method."""
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, step):
            """
            carry: (noisy_goals, )
            step: current step index
            """
            (noisy_actions, ) = carry

            times = jnp.full((*noisy_actions.shape[:-1], 1), step * step_size)
            vector_field = vector_field_fn.apply(
                params, noisy_actions, times, observations)
            new_noisy_actions = noisy_actions + vector_field * step_size

            return (new_noisy_actions, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_actions, ), _ = jax.lax.scan(
            body_fn, (noises, ), jnp.arange(num_flow_steps))

        return noisy_actions

    def reward_loss_fn(params, reward_fn, batch):
        """Compute the reward loss."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        reward_preds = reward_fn.apply(params, observations, actions)
        reward_loss = jnp.square(rewards - reward_preds).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_flow_matching_loss_fn(params, vector_field_fn, batch, key):
        """Compute the critic flow matching loss."""

        observations = batch['observations']
        actions = batch['actions']
        goals = batch['future_observations']

        key, noise_key, time_key = jax.random.split(key, 3)
        noises = jax.random.normal(noise_key, shape=goals.shape, dtype=goals.dtype)
        times = jax.random.uniform(time_key, shape=(batch_size, 1))
        noisy_goals = times * goals + (1 - times) * noises
        vector_field_targets = goals - noises
        vector_field_preds = vector_field_fn.apply(
            params, noisy_goals, times, observations, actions)
        flow_matching_loss = jnp.square(vector_field_targets - vector_field_preds).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss
        }

    def actor_flow_matching_loss_fn(params, vector_field_fn, batch, key):
        """Compute the critic flow matching loss."""
        observations = batch['observations']
        actions = batch['actions']

        key, noise_key, time_key = jax.random.split(key, 3)
        noises = jax.random.normal(noise_key, shape=actions.shape, dtype=actions.dtype)
        times = jax.random.uniform(time_key, shape=(batch_size, 1))
        noisy_actions = times * actions + (1 - times) * noises
        vector_field_targets = actions - noises
        vector_field_preds = vector_field_fn.apply(
            params, noisy_actions, times, observations)
        flow_matching_loss = jnp.square(vector_field_targets - vector_field_preds).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss
        }

    def critic_loss_fn(critic_params, target_reward_params, critic_vf_params, actor_params,
                       critic_fn, reward_fn, critic_vf_fn, actor_fn,
                       batch, key):
        """Compute the critic loss."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        key, g_noise_key, a_noise_key = jax.random.split(key, 3)
        noises = jax.random.normal(g_noise_key, shape=observations.shape, dtype=observations.dtype)
        flow_goals = compute_fwd_flow_goals(
            critic_vf_params, critic_vf_fn, noises, observations, actions)

        a_noises = jax.random.normal(
          a_noise_key, shape=actions.shape, dtype=actions.dtype)
        goal_actions = actor_fn.apply(actor_params, a_noises, flow_goals)
        goal_actions = jnp.clip(goal_actions, -1.0, 1.0)
        future_rewards = reward_fn.apply(target_reward_params, flow_goals, goal_actions)

        target_q = rewards + discount / (1 - discount) * future_rewards

        qs = critic_fn.apply(critic_params, observations, actions)
        critic_loss = expectile_loss(target_q - qs, target_q - qs, expectile).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }

    def actor_loss_fn(actor_params, actor_vf_params, critic_params,
                      actor_fn, actor_vf_fn, critic_fn,
                      batch, key):
        observations = batch['observations']
        actions = batch['actions']

        key, noise_key = jax.random.split(key)
        noises = jax.random.normal(noise_key, shape=actions.shape, dtype=actions.dtype)
        q_actions = actor_fn.apply(actor_params, noises, observations)
        q_actions = jnp.clip(q_actions, -1, 1)

        flow_actions = compute_fwd_flow_actions(actor_vf_params, actor_vf_fn, noises, observations)
        flow_actions = jnp.clip(flow_actions, -1.0, 1.0)
        flow_actions = jax.lax.stop_gradient(flow_actions)

        qs = critic_fn.apply(critic_params, observations, q_actions)
        if num_ensembles > 1:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
        distill_mse = jnp.square(flow_actions - q_actions).mean()
        distill_loss = alpha * distill_mse

        actor_loss = q_loss + distill_loss

        # logging
        key, noise_key = jax.random.split(key)
        noises = jax.random.normal(noise_key, shape=actions.shape, dtype=actions.dtype)
        action_preds = actor_fn.apply(actor_params, noises, observations)
        action_preds = jnp.clip(action_preds, -1, 1)
        mse = jnp.mean((actions - action_preds) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'distill_mse': distill_mse,
            'distill_loss': distill_loss,
            'mse': mse,
        }

    key, time_key, reward_key, critic_key, critic_vf_key, actor_key, actor_vf_key = jax.random.split(key, 7)
    example_batch = get_batch_fn(2)
    example_times = jax.random.uniform(time_key, shape=(2, 1))

    reward_fn = Reward()
    if num_ensembles > 1:
        critic_fn = ensemblize(Critic, num_ensembles)()
    else:
        critic_fn = Critic()
    critic_vf_fn = CriticVectorField(env.observation_space.shape[0])
    actor_fn = Actor(env.action_space.shape[0])
    actor_vf_fn = ActorVectorField(env.action_space.shape[0])

    reward_params = reward_fn.init(
        reward_key, example_batch['observations'], example_batch['actions']
    )
    target_reward_params = copy.deepcopy(reward_params)
    critic_params = critic_fn.init(
        critic_key, example_batch['observations'], example_batch['actions'])
    critic_vf_params = critic_vf_fn.init(
        critic_vf_key,
        example_batch['future_observations'],
        example_times,
        example_batch['observations'],
        example_batch['actions'],
    )
    actor_params = actor_fn.init(
        actor_key, example_batch['actions'], example_batch['observations'])
    actor_vf_params = actor_vf_fn.init(
        actor_vf_key,
        example_batch['actions'],
        example_times,
        example_batch['observations'],
    )

    reward_optimizer = optax.adam(learning_rate=learning_rate)
    critic_optimizer = optax.adam(learning_rate=learning_rate)
    critic_vf_optimizer = optax.adam(learning_rate=learning_rate)
    actor_optimizer = optax.adam(learning_rate=learning_rate)
    actor_vf_optimizer = optax.adam(learning_rate=learning_rate)

    reward_opt_state = reward_optimizer.init(reward_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    critic_vf_opt_state = critic_vf_optimizer.init(critic_vf_params)
    actor_opt_state = actor_optimizer.init(actor_params)
    actor_vf_opt_state = actor_vf_optimizer.init(actor_vf_params)

    reward_grad_fn = jax.value_and_grad(reward_loss_fn, has_aux=True)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    critic_flow_matching_grad_fn = jax.value_and_grad(critic_flow_matching_loss_fn, has_aux=True)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    actor_flow_matching_grad_fn = jax.value_and_grad(actor_flow_matching_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(reward_params, target_reward_params, critic_params, critic_vf_params, actor_params, actor_vf_params,
                  reward_opt_state, critic_opt_state, critic_vf_opt_state, actor_opt_state, actor_vf_opt_state,
                  batch, key):
        key, critic_key, critic_flow_matching_key, actor_key, actor_flow_matching_key = jax.random.split(key, 5)
        info = dict()

        (_, reward_info), reward_grads = reward_grad_fn(
            reward_params, reward_fn, batch
        )
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        (_, critic_info), critic_grads = critic_grad_fn(
            critic_params, target_reward_params, critic_vf_params, actor_params,
            critic_fn, reward_fn, critic_vf_fn, actor_fn,
            batch, critic_key
        )
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        (_, critic_flow_matching_info), critic_flow_matching_grads = critic_flow_matching_grad_fn(
            critic_vf_params, critic_vf_fn, batch, critic_flow_matching_key
        )
        for k, v in critic_flow_matching_info.items():
            info[f'critic_flow_matching/{k}'] = v

        (_, actor_info), actor_grads = actor_grad_fn(
            actor_params, actor_vf_params, critic_params,
            actor_fn, actor_vf_fn, critic_fn,
            batch, actor_key
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        (_, actor_flow_matching_info), actor_flow_matching_grads = actor_flow_matching_grad_fn(
            actor_vf_params, actor_vf_fn, batch, actor_flow_matching_key
        )
        for k, v in actor_flow_matching_info.items():
            info[f'actor_flow_matching/{k}'] = v

        reward_updates, reward_opt_state = reward_optimizer.update(
            reward_grads, reward_opt_state)
        reward_params = optax.apply_updates(reward_params, reward_updates)
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state)
        critic_vf_updates, critic_vf_opt_state = critic_vf_optimizer.update(
            critic_flow_matching_grads, critic_vf_opt_state)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state)
        actor_vf_updates, actor_vf_opt_state = actor_vf_optimizer.update(
            actor_flow_matching_grads, actor_vf_opt_state)

        target_reward_params = jax.tree.map(
            lambda x, y: x * (1 - tau) + y * tau,
            target_reward_params, reward_params)

        return (reward_params, target_reward_params, critic_params, critic_vf_params, actor_params, actor_vf_params,
                reward_opt_state, critic_opt_state, critic_vf_opt_state, actor_opt_state, actor_vf_opt_state,
                info)

    @jax.jit
    def sample_actions(params, observations, seed=None, temperature=1.0):
        seed, noise_seed = jax.random.split(seed)

        noises = jax.random.normal(
            noise_seed,
            shape=(*observations.shape[:-1], env.action_space.shape[0]),
            dtype=observations.dtype
        )
        actions = actor_fn.apply(params, noises, observations)
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions

    key, eval_key = jax.random.split(key)
    eval_actor_fn = supply_rng(sample_actions, rng=eval_key)

    metrics = dict()
    for step in tqdm(range(num_training_steps + 1), desc="mcfac training"):
        key, train_key = jax.random.split(key)

        batch = get_batch_fn(batch_size)
        (reward_params, target_reward_params, critic_params, critic_vf_params, actor_params, actor_vf_params,
         reward_opt_state, critic_opt_state, critic_vf_opt_state, actor_opt_state, actor_vf_opt_state,
         info) = update_fn(
            reward_params, target_reward_params, critic_params, critic_vf_params, actor_params, actor_vf_params,
            reward_opt_state, critic_opt_state, critic_vf_opt_state, actor_opt_state, actor_vf_opt_state,
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
            eval_info = evaluate(eval_actor_fn, env, actor_params, desc='mcfac evaluation')

            for k, v in eval_info.items():
                eval_k = 'eval/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

    return metrics
