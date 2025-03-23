import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from diffrax import (
    diffeqsolve, ODETerm,
    Euler, Dopri5,
)

from utils.env_utils import compute_reward
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCFMValue, Value
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class SARSAIFACQAgent(flax.struct.PyTreeNode):
    """SARSA Implicit Flow Actor Critic (Q) agent."""

    rng: Any
    network: Any
    cond_prob_path: Any
    ode_solver: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        if self.config['reward_type'] == 'state_action':
            actions = batch['actions']
        else:
            actions = None
        rewards = batch['rewards']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)
        reward_preds = self.network.select('reward')(
            observations, actions=actions,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the value loss."""
        observations = batch['observations']
        actions = batch['actions']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)

        rng, noise_rng = jax.random.split(rng)
        if self.config['critic_noise_type'] == 'normal':
            noises = jax.random.normal(
                noise_rng, shape=(self.config['num_flow_goals'], *observations.shape), dtype=observations.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            # noises = jax.random.permutation(noise_rng, observations, axis=0)
            noise_rngs = jax.random.split(noise_rng, self.config['num_flow_goals'])
            noises = jax.vmap(jax.random.permutation, in_axes=(0, None), out_axes=0)(
                noise_rngs, observations)
        # elif self.config['critic_noise_type'] == 'marginal_goal':
        #     noises = jax.random.permutation(noise_rng, goals, axis=0)
        flow_goals = self.compute_fwd_flow_goals(
            noises,
            jnp.repeat(jnp.expand_dims(observations, axis=0), self.config['num_flow_goals'], axis=0),
            jnp.repeat(jnp.expand_dims(actions, axis=0), self.config['num_flow_goals'], axis=0)
        )
        if self.config['clip_flow_goals']:
            flow_goals = jnp.clip(flow_goals,
                                  batch['observation_min'] + 1e-5,
                                  batch['observation_max'] - 1e-5)

        if self.config['reward_type'] == 'state_action':
            # raise NotImplementedError("doesn't work for image observation now")
            rng, noise_rng = jax.random.split(rng)
            a_noises = jax.random.normal(
                noise_rng, shape=(self.config['num_flow_goals'], *actions.shape), dtype=actions.dtype)
            if self.config['distill_type'] == 'fwd_sample':
                goal_actions = self.network.select('actor')(a_noises, flow_goals)
            elif self.config['distill_type'] == 'fwd_int':
                goal_action_vfs = self.network.select('actor')(a_noises, flow_goals)
                goal_actions = a_noises + goal_action_vfs
            goal_actions = jnp.clip(goal_actions, -1, 1)
        else:
            assert self.config['reward_type'] == 'state'
            goal_actions = None

        if self.config['use_reward_func']:
            assert self.config['reward_type'] == 'state'
            future_rewards = compute_reward(self.config['reward_env_info'], flow_goals)
        else:
            # if self.config['use_target_reward']:
            #     future_rewards = self.network.select('target_reward')(flow_goals, actions=goal_actions)
            # else:
            #     future_rewards = self.network.select('reward')(flow_goals, actions=goal_actions)
            future_rewards = self.network.select('reward')(flow_goals, actions=goal_actions)

        future_rewards = future_rewards.mean(axis=0)  # MC estimations
        target_q = 1.0 / (1 - self.config['discount']) * future_rewards
        qs = self.network.select('critic')(observations, actions, params=grad_params)
        critic_loss = self.expectile_loss(target_q - qs, target_q - qs, self.config['expectile']).mean()

        # For logging
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def flow_matching_loss(self, batch, grad_params, rng):
        """Compute the flow matching loss."""

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']
        masks = batch['masks']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(
                batch['observations'], params=grad_params)
            next_observations = self.network.select('target_critic_vf_encoder')(
                batch['next_observations'])

        # if self.config['critic_fm_loss_type'] == 'mc':
        #     # MC value flow matching
        #     rng, value_noise_rng, value_time_rng = jax.random.split(rng, 3)
        #     assert self.config['critic_noise_type'] == 'normal'
        #     if self.config['critic_noise_type'] == 'normal':
        #         value_noises = jax.random.normal(value_noise_rng, shape=goals.shape, dtype=goals.dtype)
        #     elif self.config['critic_noise_type'] == 'marginal_state':
        #         value_noises = jax.random.permutation(value_noise_rng, observations, axis=0)
        #     value_times = jax.random.uniform(value_time_rng, shape=(batch_size, ))
        #     value_path_sample = self.cond_prob_path(x_0=value_noises, x_1=goals, t=value_times)
        #     critic_vf_pred = self.network.select('critic_vf')(
        #         value_path_sample.x_t,
        #         value_times,
        #         observations,
        #         actions,
        #         params=grad_params,
        #     )
        #     critic_flow_matching_loss = jnp.square(critic_vf_pred - value_path_sample.dx_t).mean()
        info = dict()
        if self.config['critic_fm_loss_type'] == 'naive_sarsa':
            # naive SARSA value flow matching
            rng, current_time_rng, current_noise_rng = jax.random.split(rng, 3)
            current_times = jax.random.uniform(current_time_rng, shape=(batch_size,), dtype=observations.dtype)
            if self.config['critic_noise_type'] == 'normal':
                current_noises = jax.random.normal(
                    current_noise_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                current_noises = jax.random.permutation(current_noise_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     current_noises = jax.random.permutation(current_noise_rng, goals, axis=0)
            current_path_sample = self.cond_prob_path(
                x_0=current_noises, x_1=observations, t=current_times)
            current_vf_pred = self.network.select('critic_vf')(
                current_path_sample.x_t,
                current_times,
                observations,
                actions,
                params=grad_params,
            )
            current_loss = jnp.square(current_vf_pred - current_path_sample.dx_t).mean(axis=-1)

            rng, future_goal_rng = jax.random.split(rng)
            if self.config['critic_noise_type'] == 'normal':
                future_goal_noises = jax.random.normal(
                    future_goal_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                future_goal_noises = jax.random.permutation(future_goal_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     future_goal_noises = jax.random.permutation(current_noise_rng, goals, axis=0)
            future_flow_goals = self.compute_fwd_flow_goals(
                future_goal_noises, next_observations, next_actions,
                use_target_network=True
            )
            if self.config['clip_flow_goals']:
                future_flow_goals = jnp.clip(future_flow_goals,
                                             batch['observation_min'] + 1e-5,
                                             batch['observation_max'] - 1e-5)
            future_flow_goals = jax.lax.stop_gradient(future_flow_goals)

            rng, future_time_rng, future_noise_rng = jax.random.split(rng, 3)
            future_times = jax.random.uniform(future_time_rng, shape=(batch_size,), dtype=observations.dtype)
            if self.config['critic_noise_type'] == 'normal':
                future_noises = jax.random.normal(
                    future_noise_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                future_noises = jax.random.permutation(future_noise_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     future_noises = jax.random.permutation(future_noise_rng, goals, axis=0)
            future_path_sample = self.cond_prob_path(
                x_0=future_noises, x_1=future_flow_goals, t=future_times)
            future_vf_pred = self.network.select('critic_vf')(
                future_path_sample.x_t,
                future_times,
                observations,
                actions,
                params=grad_params,
            )
            future_loss = jnp.square(future_vf_pred - future_path_sample.dx_t).mean(axis=-1)

            if self.config['use_terminal_masks']:
                critic_flow_matching_loss = ((1 - self.config['discount']) * current_loss +
                                             self.config['discount'] * masks * future_loss).mean()
            else:
                critic_flow_matching_loss = ((1 - self.config['discount']) * current_loss +
                                             self.config['discount'] * future_loss).mean()
        elif self.config['critic_fm_loss_type'] == 'coupled_sarsa':
            # coupled SARSA value flow matching
            rng, noise_rng, bern_rng, time_rng = jax.random.split(rng, 4)
            if self.config['critic_noise_type'] == 'normal':
                noises = jax.random.normal(noise_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                noises = jax.random.permutation(noise_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     noises = jax.random.permutation(noise_rng, goals, axis=0)
            flow_observations = self.compute_fwd_flow_goals(
                noises, next_observations, next_actions, use_target_network=True)
            flow_observations = jax.lax.stop_gradient(flow_observations)
            if self.config['clip_flow_goals']:
                flow_observations = jnp.clip(flow_observations,
                                             batch['observation_min'] + 1e-5,
                                             batch['observation_max'] - 1e-5)

            bern = jax.random.bernoulli(
                bern_rng, p=self.config['discount'], shape=(batch_size, 1)).astype(observations.dtype)
            if self.config['use_terminal_masks']:
                future_observations = (1 - bern) * observations + bern * jnp.expand_dims(masks,
                                                                                         axis=-1) * flow_observations
            else:
                future_observations = (1 - bern) * observations + bern * flow_observations

            times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
            path_sample = self.cond_prob_path(
                x_0=noises, x_1=future_observations, t=times)
            vf_pred = self.network.select('critic_vf')(
                path_sample.x_t,
                times,
                observations, actions,
                params=grad_params,
            )
            critic_flow_matching_loss = jnp.square(vf_pred - path_sample.dx_t).mean()
        elif self.config['critic_fm_loss_type'] == 'sarsa_squared':
            # SARSA^2 value flow matching
            rng, time_rng, current_noise_rng, future_noise_rng = jax.random.split(rng, 4)
            times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
            if self.config['critic_noise_type'] == 'normal':
                current_noises = jax.random.normal(
                    current_noise_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                current_noises = jax.random.permutation(
                    current_noise_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     current_noises = jax.random.permutation(
            #         current_noise_rng, goals, axis=0)
            current_path_sample = self.cond_prob_path(
                x_0=current_noises, x_1=observations, t=times)
            current_vf_pred = self.network.select('critic_vf')(
                current_path_sample.x_t,
                times,
                observations, actions,
                params=grad_params,
            )
            # no gradient through the target for the encoder
            current_flow_matching_loss = jnp.square(
                jax.lax.stop_gradient(current_path_sample.dx_t) - current_vf_pred).mean(axis=-1)

            if self.config['critic_noise_type'] == 'normal':
                future_noises = jax.random.normal(
                    future_noise_rng, shape=observations.shape, dtype=observations.dtype)
            elif self.config['critic_noise_type'] == 'marginal_state':
                future_noises = jax.random.permutation(
                    future_noise_rng, observations, axis=0)
            # elif self.config['critic_noise_type'] == 'marginal_goal':
            #     future_noises = jax.random.permutation(
            #         future_noise_rng, goals, axis=0)
            flow_future_observations = self.compute_fwd_flow_goals(
                future_noises, next_observations, next_actions, use_target_network=True)
            if self.config['clip_flow_goals']:
                flow_future_observations = jnp.clip(flow_future_observations,
                                                    batch['observation_min'] + 1e-5,
                                                    batch['observation_max'] - 1e-5)
            future_path_sample = self.cond_prob_path(
                x_0=future_noises, x_1=flow_future_observations, t=times)
            future_vf_target = self.network.select('target_critic_vf')(
                future_path_sample.x_t,
                times,
                next_observations, next_actions,
            )
            future_vf_pred = self.network.select('critic_vf')(
                future_path_sample.x_t,
                times,
                observations, actions,
                params=grad_params,
            )
            future_flow_matching_loss = jnp.square(future_vf_target - future_vf_pred).mean(axis=-1)

            if self.config['use_terminal_masks']:
                critic_flow_matching_loss = ((1 - self.config['discount']) * current_flow_matching_loss
                                             + self.config['discount'] * masks * future_flow_matching_loss).mean()
            else:
                critic_flow_matching_loss = ((1 - self.config['discount']) * current_flow_matching_loss
                                             + self.config['discount'] * future_flow_matching_loss).mean()

            info.update(
                flow_future_obs_max=flow_future_observations.max(),
                flow_future_obs_min=flow_future_observations.min(),
                current_flow_matching_loss=current_flow_matching_loss.mean(),
                future_flow_matching_loss=future_flow_matching_loss.mean(),
            )
        else:
            raise NotImplementedError

        # actor flow matching
        # if self.config['encoder'] is not None:
        #     observations = self.network.select('actor_critic_encoder')(
        #         batch['observations'])  # no gradients for the encoder

        rng, actor_noise_rng, actor_time_rng = jax.random.split(rng, 3)
        actor_noises = jax.random.normal(actor_noise_rng, shape=actions.shape, dtype=actions.dtype)
        actor_times = jax.random.uniform(actor_time_rng, shape=(batch_size,))
        actor_path_sample = self.cond_prob_path(x_0=actor_noises, x_1=actions, t=actor_times)
        actor_vf_pred = self.network.select('actor_vf')(
            actor_path_sample.x_t,
            actor_times,
            jax.lax.stop_gradient(observations),  # no gradients for the encoder
            params=grad_params,
        )
        actor_flow_matching_loss = jnp.square(actor_vf_pred - actor_path_sample.dx_t).mean()

        flow_matching_loss = critic_flow_matching_loss + actor_flow_matching_loss

        info.update(
            flow_matching_loss=flow_matching_loss,
            critic_flow_matching_loss=critic_flow_matching_loss,
            actor_flow_matching_loss=actor_flow_matching_loss,
        )

        return flow_matching_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""

        observations = batch['observations']
        actions = batch['actions']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=actions.shape, dtype=actions.dtype)
        if self.config['distill_type'] == 'fwd_sample':
            q_actions = self.network.select('actor')(noises, observations, params=grad_params)
        elif self.config['distill_type'] == 'fwd_int':
            q_action_vfs = self.network.select('actor')(noises, observations, params=grad_params)
            q_actions = noises + q_action_vfs
        q_actions = jnp.clip(q_actions, -1, 1)

        flow_actions = self.compute_fwd_flow_actions(noises, observations)
        flow_actions = jnp.clip(flow_actions, -1, 1)
        flow_actions = jax.lax.stop_gradient(flow_actions)  # stop gradients for the encoder

        # Q loss
        # if self.config['encoder'] is not None:
        #     observations = self.network.select('critic_vf_encoder')(observations)
        qs = self.network.select('critic')(observations, actions=q_actions)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # distill loss
        train_mask = jnp.float32((actions * 1e8 % 10)[:, 0] != 4)
        val_mask = 1.0 - train_mask

        distill_mse = (train_mask * jnp.square(q_actions - flow_actions).mean(axis=-1)).mean()
        distill_val_mse = (val_mask * jnp.square(q_actions - flow_actions).mean(axis=-1)).mean()
        distill_loss = self.config['alpha'] * distill_mse

        # Total loss
        actor_loss = q_loss + distill_loss

        # Additional metrics for logging.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=actions.shape, dtype=actions.dtype)
        if self.config['distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(noises, observations)
        elif self.config['distill_type'] == 'fwd_int':
            action_vfs = self.network.select('actor')(noises, observations)
            actions = noises + action_vfs
        actions = jnp.clip(actions, -1, 1)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'distill_mse': distill_mse,
            'distill_val_mse': distill_val_mse,
            'mse': mse,
        }

    def compute_fwd_flow_actions(self, noises, observations):
        noisy_actions = noises
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_actions,) = carry

            times = jnp.full(noisy_actions.shape[:-1], i * step_size)
            vf = self.network.select('actor_vf')(
                noisy_actions, times, observations,
            )
            new_noisy_actions = noisy_actions + vf * step_size

            return (new_noisy_actions,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_actions,), _ = jax.lax.scan(
            body_fn, (noisy_actions,), jnp.arange(num_flow_steps))

        return noisy_actions

    def compute_fwd_flow_goals(self, noises, observations, actions, use_target_network=False):
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

        noisy_goals = noises
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_goals,) = carry

            times = jnp.full(noisy_goals.shape[:-1], i * step_size)
            vf = self.network.select(module_name)(
                noisy_goals, times, observations, actions)
            new_noisy_goals = noisy_goals + vf * step_size

            return (new_noisy_goals,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals,), _ = jax.lax.scan(
            body_fn, (noisy_goals,), jnp.arange(num_flow_steps))

        return noisy_goals

    # def compute_fwd_flow_goals(self, noises, observations, use_target_network=False):
    #     if use_target_network:
    #         module_name = 'target_critic_vf'
    #     else:
    #         module_name = 'critic_vf'
    #
    #     def vector_field(time, noisy_goals, carry):
    #         (observations, ) = carry
    #         times = jnp.full(noisy_goals.shape[:-1], time)
    #
    #         vf = self.network.select(module_name)(
    #             noisy_goals, times, observations)
    #
    #         return vf
    #
    #     ode_term = ODETerm(vector_field)
    #     ode_sol = diffeqsolve(
    #         ode_term, self.ode_solver,
    #         t0=0.0, t1=1.0, dt0=1 / self.config['num_flow_steps'],
    #         y0=noises, args=(observations, ),
    #         throw=False,  # (chongyi): setting throw to false is important for speed
    #     )
    #     noisy_goals = ode_sol.ys[-1]
    #
    #     return noisy_goals

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, flow_matching_rng = jax.random.split(rng)

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        loss = flow_matching_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, flow_matching_rng, actor_rng = jax.random.split(rng, 4)

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        # value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        # for k, v in value_info.items():
        #     info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = reward_loss + critic_loss + flow_matching_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # self.target_update(new_network, 'reward')
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update=True, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # self.target_update(new_network, 'reward')
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
            self,
            observations,
            seed=None,
            temperature=1.0,
    ):
        """Sample actions from the actor."""
        if observations.shape == self.config['obs_dims']:
            observations = jnp.expand_dims(observations, axis=0)

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)

        seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            noise_seed,
            shape=(observations.shape[0], self.config['action_dim']),
            dtype=self.config['action_dtype']
        )
        if self.config['distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(noises, observations)
        elif self.config['distill_type'] == 'fwd_int':
            actions = noises + self.network.select('actor')(noises, observations)
        actions = jnp.clip(actions, -1, 1)
        actions = actions.squeeze()

        return actions

    @classmethod
    def create(
            cls,
            seed,
            ex_observations,
            ex_actions,
            config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        # obs_dim = ex_observations.shape[-1]
        # action_dim = ex_actions.shape[-1]
        # ex_orig_observations = ex_observations
        # ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],), dtype=ex_actions.dtype)
        if config['use_reward_func']:
            assert config['reward_env_info'] is not None
        ex_orig_observations = ex_observations

        ex_times = ex_actions[..., 0]
        obs_dims = ex_observations.shape[1:]
        obs_dim = obs_dims[-1]
        action_dim = ex_actions.shape[-1]
        action_dtype = ex_actions.dtype

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if 'mlp_hidden_dims' in encoder_module.keywords:
                obs_dim = encoder_module.keywords['mlp_hidden_dims'][-1]
            else:
                obs_dim = encoder_modules['impala'].mlp_hidden_dims[-1]
            rng, obs_rng = jax.random.split(rng, 2)
            ex_observations = jax.random.normal(
                obs_rng, shape=(ex_observations.shape[0], obs_dim), dtype=action_dtype)

            # encoders['critic'] = encoder_module()
            encoders['critic_vf'] = encoder_module()
            # encoders['actor'] = encoder_module()
            # encoders['actor_vf'] = encoder_module()

        # Define value and actor networks.
        critic_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            # encoder=encoders.get('critic'),
        )
        critic_vf_def = GCFMVectorField(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            vector_dim=obs_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            # state_encoder=encoders.get('critic_vf'),
        )

        actor_vf_def = GCFMVectorField(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            vector_dim=action_dim,
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            # state_encoder=encoders.get('actor_vf'),
        )
        actor_def = GCFMValue(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['actor_hidden_dims'],
            output_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            # state_encoder=encoders.get('actor'),
        )

        reward_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
            # encoder=encoders.get('actor_critic'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_observations, ex_times, ex_observations, ex_actions)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_observations, ex_times, ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations)),
            actor=(actor_def, (ex_actions, ex_observations)),
        )
        if config['reward_type'] == 'state':
            network_info.update(
                reward=(reward_def, (ex_observations,)),
                target_reward=(copy.deepcopy(reward_def), (ex_observations,)),
            )
        else:
            network_info.update(
                reward=(reward_def, (ex_observations, ex_actions)),
                target_reward=(copy.deepcopy(reward_def), (ex_observations, ex_actions)),
            )
        # if encoders.get('actor_bc_flow') is not None:
        #     # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
        #     network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
        if config['encoder'] is not None:
            network_info['critic_vf_encoder'] = (
                encoders.get('critic_vf'), (ex_orig_observations,))
            network_info['target_critic_vf_encoder'] = (
                copy.deepcopy(encoders.get('critic_vf')), (ex_orig_observations,))
        # if encoders.get('actor_critic') is not None:
        #     network_info['actor_critic_encoder'] = (encoders.get('actor_critic'), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        # params['modules_target_reward'] = params['modules_reward']
        if config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        if config['ode_solver_type'] == 'euler':
            ode_solver = Euler()
        elif config['ode_solver_type'] == 'dopri5':
            ode_solver = Dopri5()
        else:
            raise TypeError("Unknown ode_solver_type: {}".format(config['ode_solver_type']))

        config['obs_dims'] = obs_dims
        config['action_dim'] = action_dim
        config['action_dtype'] = action_dtype

        return cls(rng, network=network, cond_prob_path=cond_prob_path,
                   ode_solver=ode_solver, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='sarsa_ifac_q',  # Agent name.
            obs_dims=ml_collections.config_dict.placeholder(tuple),
            # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            action_dtype=ml_collections.config_dict.placeholder(np.dtype),
            # Action data type (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            network_type='mlp',  # Type of the network
            num_residual_blocks=1,  # Number of residual blocks for simba network.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            reward_layer_norm=True,  # Whether to use layer normalization for the reward.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            q_agg='mean',  # Aggregation method for target Q values.
            critic_noise_type='normal',  # Critic noise type. ('marginal_state', 'normal').
            critic_fm_loss_type='sarsa_squared',
            # Type of critic flow matching loss. ('naive_sarsa', 'coupled_sarsa', 'sarsa_squared')
            num_flow_goals=1,  # Number of future flow goals for the compute target q.
            clip_flow_goals=False,  # Whether to clip the flow goals.
            use_terminal_masks=False,  # Whether to use the terminal masks.
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            actor_freq=2,  # Actor update frequency.
            distill_type='fwd_sample',  # Distillation type. ('fwd_sample', 'fwd_int').
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            use_reward_func=False,  # Whether to use the ground truth reward function.
            use_target_reward=False,  # Whether to use the target reward network.
            reward_type='state',  # Reward type. ('state', 'state_action')
            encoder_actor_loss_grad=False,  # Whether to backpropagate gradients from the actor loss into the encoder.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            reward_env_info=ml_collections.config_dict.placeholder(dict),  # Environment information for computing the ground truth reward.
        )
    )
    return config
