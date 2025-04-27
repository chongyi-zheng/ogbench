import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

# from diffrax import (
#     diffeqsolve, ODETerm,
#     Euler, Dopri5,
# )

# from utils.env_utils import compute_reward
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCFMBilinearVectorField, GCActor, Value
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class SARSAIFQLGPIAgent(flax.struct.PyTreeNode):
    """SARSA Implicit Flow Q-Learning + Generalized Policy Improvement agent."""

    rng: Any
    network: Any
    cond_prob_path: Any
    # ode_solver: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        assert self.config['reward_type'] == 'state'
        rewards = batch['rewards']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)
        reward_preds = self.network.select('reward')(
            observations, params=grad_params,
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

        if self.config['vector_field_type'] == 'mlp':
            rng, noise_rng, latent_rng = jax.random.split(rng, 3)
            assert self.config['critic_noise_type'] == 'normal'
            noises = jax.random.normal(
                noise_rng,
                shape=observations.shape,
                dtype=observations.dtype
            )
            latents = jax.random.normal(
                latent_rng,
                shape=actions.shape,
                dtype=observations.dtype,
            )
            flow_goals = self.compute_fwd_flow_goals(
                noises,
                observations,
                actions,
                latents,
                observation_min=batch.get('observation_min', None),
                observation_max=batch.get('observation_max', None)
            )
        elif self.config['vector_field_type'] == 'bilinear':
            rng, noise_rng = jax.random.split(rng)
            if self.config['critic_noise_type'] == 'normal':
                noises = jax.random.normal(
                    noise_rng,
                    shape=(self.config['num_flow_latents'] * self.config['num_flow_goals'], *observations.shape[1:]),
                    dtype=observations.dtype
                )
            elif self.config['critic_noise_type'] == 'marginal_state':
                raise NotImplementedError

            rng, latent_rng = jax.random.split(rng)
            latents = jax.random.normal(
                latent_rng,
                shape=actions.shape,
                dtype=observations.dtype,
            )
            flow_goals = self.compute_fwd_flow_goals(
                noises, observations, actions, latents,
                observation_min=batch.get('observation_min', None),
                observation_max=batch.get('observation_max', None),
            )
            flow_goals = flow_goals.reshape([self.config['num_flow_latents'], self.config['num_flow_goals'],
                                             *observations.shape])

        assert self.config['reward_type'] == 'state'
        future_rewards = self.network.select('reward')(flow_goals)
        target_behavioral_q = 1.0 / (1 - self.config['discount']) * future_rewards

        behavioral_q = self.network.select('behavioral_critic')(
            jnp.concatenate([observations, latents], axis=-1), actions, params=grad_params)
        behavioral_q_loss = jnp.square(behavioral_q - target_behavioral_q).mean()

        # future_rewards = future_rewards.mean(axis=(0, 1))  # MC estimations over latent and future state dims.
        # target_q = 1.0 / (1 - self.config['discount']) * future_rewards.mean(axis=1).max(axis=0)
        latents = jax.random.normal(
            latent_rng,
            shape=actions.shape,
            dtype=actions.dtype,
        )
        target_q = self.network.select('behavioral_critic')(
            jnp.concatenate([observations, latents], axis=-1), actions)

        qs = self.network.select('critic')(batch['observations'], actions, params=grad_params)
        q_loss = self.expectile_loss(
            target_q - qs,
            target_q - qs,
            self.config['expectile']
        ).mean()

        critic_loss = behavioral_q_loss + q_loss

        # For logging
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        return critic_loss, {
            'critic_loss': critic_loss,
            'behavioral_q_loss': behavioral_q_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def flow_transition_loss(self, batch, grad_params, rng):
        """Compute the flow transition loss."""

        batch_size = batch['observations'].shape[0]
        # observations = batch['observations']
        # actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        if self.config['encoder'] is not None:
            next_observations = self.network.select('critic_vf_encoder')(
                batch['next_observations'], params=grad_params)

        # obs_actions = jnp.concatenate([observations, actions], axis=-1)
        # next_obs_actions = jnp.concatenate([next_observations, next_actions], axis=-1)

        # flow matching for the transition
        rng, time_rng, noise_rng = jax.random.split(rng, 3)
        times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=next_actions.dtype)
        if self.config['critic_noise_type'] == 'normal':
            noises = jax.random.normal(
                noise_rng, shape=next_actions.shape, dtype=next_actions.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            noises = jax.random.permutation(
                noise_rng, next_actions, axis=0)
        path_sample = self.cond_prob_path(
            x_0=noises, x_1=next_actions, t=times)
        vf_pred = self.network.select('transition_vf')(
            path_sample.x_t,
            times,
            next_observations,
            params=grad_params,
        )
        if self.config['vector_field_type'] == 'bilinear':
            vf_pred = jax.vmap(jnp.diag, -1, -1)(vf_pred)
        flow_matching_loss = jnp.square(
            jax.lax.stop_gradient(path_sample.dx_t) - vf_pred).mean()

        info = dict(
            flow_matching_loss=flow_matching_loss,
        )

        return flow_matching_loss, info

    def flow_occupancy_loss(self, batch, grad_params, rng):
        """Compute the flow occupancy loss."""

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

        # infer z using another batch of (s, a)
        # rng, perm_rng = jax.random.split(rng)
        # obs_actions = jnp.concatenate([observations, actions], axis=-1)
        # obs_actions = jax.random.permutation(
        #     perm_rng, obs_actions, axis=0)
        # next_obs_actions = jnp.concatenate([next_observations, next_actions], axis=-1)
        latents = self.compute_rev_flow_transitions(next_actions, next_observations)
        if self.config['vector_field_type'] == 'bilinear':
            latents = jax.vmap(jnp.diag, -1, -1)(latents)

        info = dict()
        # SARSA^2 flow matching for the occupancy
        rng, time_rng, current_noise_rng, future_noise_rng = jax.random.split(rng, 4)
        times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
        if self.config['critic_noise_type'] == 'normal':
            current_noises = jax.random.normal(
                current_noise_rng, shape=observations.shape, dtype=observations.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            current_noises = jax.random.permutation(
                current_noise_rng, observations, axis=0)
        current_path_sample = self.cond_prob_path(
            x_0=current_noises, x_1=observations, t=times)
        current_vf_pred = self.network.select('critic_vf')(
            current_path_sample.x_t,
            times,
            observations, actions, latents,
            params=grad_params,
        )
        if self.config['vector_field_type'] == 'bilinear':
            current_vf_pred = jax.vmap(jnp.diag, -1, -1)(current_vf_pred)
        # stop gradient for the image encoder
        current_flow_matching_loss = jnp.square(
            jax.lax.stop_gradient(current_path_sample.dx_t) - current_vf_pred).mean(axis=-1)

        future_noises = current_noises
        flow_future_observations = self.compute_fwd_flow_goals(
            future_noises, next_observations, next_actions, latents,
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
            use_target_network=True
        )
        if self.config['vector_field_type'] == 'bilinear':
            flow_future_observations = jax.vmap(jnp.diag, -1, -1)(flow_future_observations)
        future_path_sample = self.cond_prob_path(
            x_0=future_noises, x_1=flow_future_observations, t=times)
        future_vf_target = self.network.select('target_critic_vf')(
            future_path_sample.x_t,
            times,
            next_observations, next_actions, latents,
        )
        future_vf_pred = self.network.select('critic_vf')(
            future_path_sample.x_t,
            times,
            jax.lax.stop_gradient(observations), actions, latents,
            params=grad_params,
        )
        if self.config['vector_field_type'] == 'bilinear':
            future_vf_pred = jax.vmap(jnp.diag, -1, -1)(future_vf_pred)
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

        flow_matching_loss = critic_flow_matching_loss

        info.update(
            flow_matching_loss=flow_matching_loss,
        )

        return flow_matching_loss, info

    def behavioral_cloning_loss(self, batch, grad_params, rng):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observations']
        actions = batch['actions']

        if self.config['use_mixup']:
            rng, perm_rng, beta_rng = jax.random.split(rng, 3)
            batch_size = observations.shape[0]

            pdist = jnp.sum((observations[:, None] - observations[None]) ** 2, axis=-1)
            mixup_probs = jnp.exp(-pdist / (2 * self.config['mixup_bandwidth'] ** 2))
            mixup_probs /= jnp.sum(mixup_probs, axis=-1)
            c = jnp.cumsum(mixup_probs, axis=-1)
            u = jax.random.uniform(perm_rng, shape=(batch_size, 1))
            perms = (u < c).argmax(axis=-1)

            lam = jax.random.beta(beta_rng, self.config['mixup_alpha'], self.config['mixup_alpha'],
                                  shape=(batch_size, 1))
            mixup_observations = lam * observations + (1.0 - lam) * observations[perms]
            mixup_actions = lam * actions + (1.0 - lam) * actions[perms]

            dist = self.network.select('actor')(observations)
            log_prob = dist.log_prob(actions)
            mixup_dist = self.network.select('actor')(mixup_observations, params=grad_params)
            mixup_log_prob = mixup_dist.log_prob(mixup_actions)
            bc_loss = -mixup_log_prob.mean()

            return bc_loss, {
                'bc_loss': bc_loss,
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
                'mixup_bc_log_prob': mixup_log_prob.mean(),
                'mixup_mse': jnp.mean((mixup_dist.mode() - mixup_actions) ** 2),
                'mixup_std': jnp.mean(mixup_dist.scale_diag),
            }
        else:
            dist = self.network.select('actor')(observations, params=grad_params)
            log_prob = dist.log_prob(actions)
            bc_loss = -log_prob.mean()

            return bc_loss, {
                'bc_loss': bc_loss,
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the DDPG + BC actor loss."""

        observations = batch['observations']
        actions = batch['actions']

        # DDPG+BC loss.
        dist = self.network.select('actor')(observations, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        qs = self.network.select('critic')(observations, actions=q_actions)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        if self.config['use_mixup']:
            rng, perm_rng, beta_rng = jax.random.split(rng, 3)
            batch_size = observations.shape[0]

            pdist = jnp.sum((observations[:, None] - observations[None]) ** 2, axis=-1)
            mixup_probs = jnp.exp(-pdist / (2 * self.config['mixup_bandwidth'] ** 2))
            mixup_probs /= jnp.sum(mixup_probs, axis=-1)
            c = jnp.cumsum(mixup_probs, axis=-1)
            u = jax.random.uniform(perm_rng, shape=(batch_size, 1))
            perms = (u < c).argmax(axis=-1)

            lam = jax.random.beta(beta_rng, self.config['mixup_alpha'], self.config['mixup_alpha'],
                                  shape=(batch_size, 1))
            mixup_observations = lam * observations + (1.0 - lam) * observations[perms]
            mixup_actions = lam * actions + (1.0 - lam) * actions[perms]

            dist = self.network.select('actor')(observations)
            log_prob = dist.log_prob(actions)
            mixup_dist = self.network.select('actor')(mixup_observations, params=grad_params)
            mixup_log_prob = mixup_dist.log_prob(mixup_actions)
            bc_loss = -(self.config['alpha'] * mixup_log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
                'mixup_bc_log_prob': mixup_log_prob.mean(),
                'mixup_mse': jnp.mean((mixup_dist.mode() - mixup_actions) ** 2),
                'mixup_std': jnp.mean(mixup_dist.scale_diag),
            }
        else:
            dist = self.network.select('actor')(observations, params=grad_params)
            log_prob = dist.log_prob(actions)
            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

    def compute_rev_flow_transitions(self, next_observation_actions, observations):
        if self.config['vector_field_type'] == 'mlp':
            noisy_obs_actions = next_observation_actions
        elif self.config['vector_field_type'] == 'bilinear':
            noisy_obs_actions = jnp.repeat(next_observation_actions, self.config['latent_dim'], axis=-1)
        init_times = jnp.ones(noisy_obs_actions.shape[:-1], dtype=noisy_obs_actions.dtype)
        end_times = jnp.zeros(noisy_obs_actions.shape[:-1], dtype=noisy_obs_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_obs_actions, ) = carry

            times = i * step_size + init_times
            if self.config['vector_field_type'] == 'mlp':
                vf = self.network.select('transition_vf')(
                    noisy_obs_actions, times, observations)
            elif self.config['vector_field_type'] == 'bilinear':
                # _, _, vf = self.network.select('transition_vf')(
                #     noisy_obs_actions, times, observations, info=True)
                vf = self.network.select('transition_vf')(
                    noisy_obs_actions, times, observations, ngt_repr=True) / jnp.sqrt(self.config['latent_dim'])

            new_noisy_obs_actions = noisy_obs_actions + vf * jnp.expand_dims(step_size, axis=-1)
            if self.config['vector_field_type'] == 'mlp' and self.config['clip_flow_goals']:
                new_noisy_obs_actions = jnp.clip(new_noisy_obs_actions, -1, 1)

            return (new_noisy_obs_actions,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_obs_actions,), _ = jax.lax.scan(
            body_fn, (noisy_obs_actions,), jnp.arange(self.config['num_flow_steps']))
        if self.config['vector_field_type'] == 'bilinear':
            # _, phi, _ = self.network.select('transition_vf')(
            #     noisy_obs_actions, end_times, observations, info=True)
            phi = self.network.select('transition_vf')(
                noisy_obs_actions, end_times, observations, sa_repr=True)
            psi = noisy_obs_actions.reshape([
                -1, next_observation_actions.shape[-1], self.config['latent_dim']])
            noisy_obs_actions = jnp.einsum('ik,jlk->ijl', phi, psi) / jnp.sqrt(self.config['latent_dim'])

        if self.config['vector_field_type'] == 'bilinear' and self.config['clip_flow_goals']:
            noisy_obs_actions = jnp.clip(noisy_obs_actions, -1, 1)

        return noisy_obs_actions

    def compute_fwd_flow_goals(self, noises, observations, actions, latents,
                               observation_min=None, observation_max=None,
                               init_times=None, end_times=None,
                               use_target_network=False):
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

        if self.config['vector_field_type'] == 'mlp':
            noisy_goals = noises
        elif self.config['vector_field_type'] == 'bilinear':
            noisy_goals = jnp.repeat(noises, self.config['latent_dim'], axis=-1)
        if init_times is None:
            init_times = jnp.zeros(noisy_goals.shape[:-1], dtype=noisy_goals.dtype)
        if end_times is None:
            end_times = jnp.ones(noisy_goals.shape[:-1], dtype=noisy_goals.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_goals, ) = carry

            times = i * step_size + init_times
            if self.config['vector_field_type'] == 'mlp':
                vf = self.network.select(module_name)(
                    noisy_goals, times, observations, actions, latents)
            elif self.config['vector_field_type'] == 'bilinear':
                # _, _, vf = self.network.select(module_name)(
                #     noisy_goals, times, observations, actions, latents, info=True)
                vf = self.network.select(module_name)(
                    noisy_goals, times, observations, actions, latents, ngt_repr=True) / jnp.sqrt(self.config['latent_dim'])
            new_noisy_goals = noisy_goals + vf * jnp.expand_dims(step_size, axis=-1)
            if self.config['vector_field_type'] == 'mlp' and self.config['clip_flow_goals']:
                new_noisy_goals = jnp.clip(new_noisy_goals, observation_min + 1e-5, observation_max - 1e-5)

            return (new_noisy_goals,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals,), _ = jax.lax.scan(
            body_fn, (noisy_goals,), jnp.arange(self.config['num_flow_steps']))
        if self.config['vector_field_type'] == 'bilinear':
            # _, phi, _ = self.network.select(module_name)(
            #     noisy_goals, end_times, observations, actions, latents, info=True)
            phi = self.network.select(module_name)(
                noisy_goals, end_times, observations, actions, latents, sa_repr=True)
            psi = noisy_goals.reshape([-1, noises.shape[-1], self.config['latent_dim']])
            noisy_goals = jnp.einsum('ik,jlk->ijl', phi, psi) / jnp.sqrt(self.config['latent_dim'])

        if self.config['vector_field_type'] == 'bilinear' and self.config['clip_flow_goals']:
            noisy_goals = jnp.clip(noisy_goals, observation_min + 1e-5, observation_max - 1e-5)

        return noisy_goals

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, flow_transition_rng, flow_occupancy_rng, behavioral_cloning_rng = jax.random.split(
            rng, 4)

        flow_transition_loss, flow_transition_info = self.flow_transition_loss(
            batch, grad_params, flow_transition_rng)
        for k, v in flow_transition_info.items():
            info[f'flow_transition/{k}'] = v

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            batch, grad_params, flow_occupancy_rng)
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        bc_loss, bc_info = self.behavioral_cloning_loss(
            batch, grad_params, behavioral_cloning_rng)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        loss = flow_transition_loss + flow_occupancy_loss + bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, flow_transition_rng, flow_occupancy_rng, actor_rng = jax.random.split(
            rng, 5)

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_transition_loss, flow_transition_info = self.flow_transition_loss(
            batch, grad_params, flow_transition_rng)
        for k, v in flow_transition_info.items():
            info[f'flow_transition/{k}'] = v

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            batch, grad_params, flow_occupancy_rng)
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = reward_loss + critic_loss + flow_transition_loss + flow_occupancy_loss + actor_loss
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
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)

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

            encoders['critic'] = encoder_module()
            encoders['critic_vf'] = encoder_module()
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define value and actor networks.
        critic_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        behavioral_critic_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            encoder=encoders.get('critic'),
        )
        if config['vector_field_type'] == 'mlp':
            critic_vf_def = GCFMVectorField(
                network_type=config['network_type'],
                num_residual_blocks=config['num_residual_blocks'],
                vector_dim=obs_dim,
                time_sin_embedding=config['vector_field_time_sin_embedding'],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                # state_encoder=encoders.get('critic_vf'),
            )
            transition_vf_def = GCFMVectorField(
                network_type=config['network_type'],
                num_residual_blocks=config['num_residual_blocks'],
                vector_dim=action_dim,
                time_sin_embedding=config['vector_field_time_sin_embedding'],
                hidden_dims=config['transition_hidden_dims'],
                layer_norm=config['transition_hidden_dims'],
            )
        elif config['vector_field_type'] == 'bilinear':
            critic_vf_def = GCFMBilinearVectorField(
                network_type=config['network_type'],
                num_residual_blocks=config['num_residual_blocks'],
                vector_dim=obs_dim,
                latent_dim=config['latent_dim'],
                time_sin_embedding=config['vector_field_time_sin_embedding'],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
            )
            transition_vf_def = GCFMBilinearVectorField(
                network_type=config['network_type'],
                num_residual_blocks=config['num_residual_blocks'],
                vector_dim=action_dim,
                latent_dim=config['latent_dim'],
                time_sin_embedding=config['vector_field_time_sin_embedding'],
                hidden_dims=config['transition_hidden_dims'],
                layer_norm=config['transition_hidden_dims'],
            )
        else:
            raise NotImplementedError

        actor_def = GCActor(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )

        reward_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
            # encoder=encoders.get('actor_critic'),
        )

        assert config['reward_type'] == 'state'
        network_info = dict(
            critic=(critic_def, (ex_orig_observations, ex_actions)),
            behavioral_critic=(behavioral_critic_def, (
                jnp.concatenate([ex_orig_observations, ex_actions], axis=-1), ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_observations, ex_times,
                ex_observations, ex_actions,
                ex_actions)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_observations, ex_times,
                ex_observations, ex_actions,
                ex_actions)),
            transition_vf=(transition_vf_def, (
                # jnp.concatenate([ex_observations, ex_actions], axis=-1), ex_times,
                ex_actions, ex_times,
                ex_observations,
            )),
            actor=(actor_def, (ex_orig_observations, )),
            reward=(reward_def, (ex_observations,)),
        )
        if config['encoder'] is not None:
            network_info['critic_vf_encoder'] = (
                encoders.get('critic_vf'), (ex_orig_observations,))
            network_info['target_critic_vf_encoder'] = (
                copy.deepcopy(encoders.get('critic_vf')), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        if config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        config['obs_dims'] = obs_dims
        config['action_dim'] = action_dim
        config['action_dtype'] = action_dtype

        return cls(rng, network=network, cond_prob_path=cond_prob_path,
                   config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='sarsa_ifql_gpi',  # Agent name.
            obs_dims=ml_collections.config_dict.placeholder(tuple),
            # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            action_dtype=ml_collections.config_dict.placeholder(np.dtype),
            # Action data type (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            network_type='mlp',  # Type of the network.
            num_residual_blocks=1,  # Number of residual blocks for simba network.
            vector_field_type='mlp',  # Type of vector field network. ('mlp', 'bilinear')
            vector_field_time_sin_embedding=True,  # Whether to use time embedding in the vector field.
            transition_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            reward_layer_norm=True,  # Whether to use layer normalization for the reward.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            latent_dim=512,  # Latent dimension for phi and psi.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            q_agg='mean',  # Aggregation method for target Q values.
            critic_noise_type='normal',  # Critic noise type. ('marginal_state', 'normal').
            critic_fm_loss_type='sarsa_squared', # Type of critic flow matching loss. ('naive_sarsa', 'coupled_sarsa', 'sarsa_squared')
            num_flow_goals=4,  # Number of future flow goals for computing the target q.
            num_flow_latents=4,  # Number of flow latents for computing the target q.
            clip_flow_goals=False,  # Whether to clip the flow goals.
            use_terminal_masks=False,  # Whether to use the terminal masks.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            use_mixup=False,  # Whether to use mixup for the behavioral cloning loss. (prevent overfitting).
            mixup_alpha=2.0,  # mixup beta distribution parameter.
            mixup_bandwidth=2.75,  # mixup distance bandwith.
            actor_freq=2,  # Actor update frequency.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            reward_type='state',  # Reward type. ('state', 'state_action')
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            reward_env_info=ml_collections.config_dict.placeholder(dict),  # Environment information for computing the ground truth reward.
        )
    )
    return config
