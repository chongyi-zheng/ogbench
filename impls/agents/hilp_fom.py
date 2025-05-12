import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCActor, GCMetricValue, GCValue, FMValue, Value
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class HILPFOMAgent(flax.struct.PyTreeNode):
    """HILP + flow occupancy measure agent.

    Reference: https://github.com/enjeeneer/zero-shot-rl/blob/main/agents/fb/agent.py.

    """

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
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

        rng, noise_rng, latent_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(
            noise_rng,
            shape=(self.config['num_flow_goals'], *observations.shape),
            dtype=observations.dtype
        )
        latents = jax.random.normal(
            latent_rng,
            shape=(self.config['num_flow_goals'], *actions.shape[:-1], self.config['latent_dim']),
            dtype=observations.dtype,
        )
        latents = latents / jnp.linalg.norm(
            latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        flow_goals = self.compute_fwd_flow_goals(
            noises,
            jnp.broadcast_to(
                observations[None],
                (self.config['num_flow_goals'], *observations.shape)
            ),
            jnp.broadcast_to(
                actions[None],
                (self.config['num_flow_goals'], *actions.shape)
            ),
            latents,
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
        )

        future_rewards = self.network.select('reward')(flow_goals)

        # future_rewards = future_rewards.mean(axis=(0, 1))  # MC estimations over latent and future state dims.
        target_q = 1.0 / (1 - self.config['discount']) * future_rewards.mean(axis=0)
        qs = self.network.select('critic')(batch['observations'], actions, params=grad_params)
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

    def flow_occupancy_loss(self, batch, grad_params, rng):
        """Compute the flow occupancy loss."""

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(
                batch['observations'], params=grad_params)
            next_observations = self.network.select('target_critic_vf_encoder')(
                batch['next_observations'])

        # infer z
        latents = self.get_phis(next_observations) - self.get_phis(observations)
        latents = latents / jnp.linalg.norm(
            latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        # SARSA^2 flow matching for the occupancy
        rng, time_rng, current_noise_rng, future_noise_rng = jax.random.split(rng, 4)
        times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
        current_noises = jax.random.normal(
            current_noise_rng, shape=observations.shape, dtype=observations.dtype)
        current_path_sample = self.cond_prob_path(
            x_0=current_noises, x_1=observations, t=times)
        current_vf_pred = self.network.select('critic_vf')(
            current_path_sample.x_t,
            times,
            observations, actions, jax.lax.stop_gradient(latents),
            params=grad_params,
        )
        # stop gradient for the image encoder
        current_flow_matching_loss = jnp.square(
            jax.lax.stop_gradient(current_path_sample.dx_t) - current_vf_pred).mean(axis=-1)

        future_noises = jax.random.normal(
            current_noise_rng, shape=observations.shape, dtype=observations.dtype)
        flow_future_observations = self.compute_fwd_flow_goals(
            future_noises, next_observations, next_actions, jax.lax.stop_gradient(latents),
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
            use_target_network=True,
        )
        future_path_sample = self.cond_prob_path(
            x_0=future_noises, x_1=flow_future_observations, t=times)
        future_vf_target = self.network.select('target_critic_vf')(
            future_path_sample.x_t,
            times,
            next_observations, next_actions, jax.lax.stop_gradient(latents),
        )
        future_vf_pred = self.network.select('critic_vf')(
            future_path_sample.x_t,
            times,
            jax.lax.stop_gradient(observations), actions, jax.lax.stop_gradient(latents),
            params=grad_params,
        )
        future_flow_matching_loss = jnp.square(future_vf_target - future_vf_pred).mean(axis=-1)

        flow_matching_loss = ((1 - self.config['discount']) * current_flow_matching_loss
                              + self.config['discount'] * future_flow_matching_loss).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
            'flow_future_obs_max': flow_future_observations.max(),
            'flow_future_obs_min': flow_future_observations.min(),
            'current_flow_matching_loss': current_flow_matching_loss.mean(),
            'future_flow_matching_loss': future_flow_matching_loss.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the DDPG + BC actor loss."""

        observations = batch['observations']
        actions = batch['actions']

        # DDPG+BC loss.
        dummy_latents = jnp.zeros((self.config['batch_size'], self.config['latent_dim']),
                                  dtype=actions.dtype)
        dist = self.network.select('actor')(observations, dummy_latents, params=grad_params)
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

    def compute_fwd_flow_goals(self, noises, observations, actions, latents,
                               observation_min=None, observation_max=None,
                               init_times=None, end_times=None,
                               use_target_network=False):
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

        noisy_goals = noises
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
            vf = self.network.select(module_name)(
                noisy_goals, times, observations, actions, latents)
            new_noisy_goals = noisy_goals + vf * jnp.expand_dims(step_size, axis=-1)
            if self.config['clip_flow_goals']:
                new_noisy_goals = jnp.clip(new_noisy_goals, observation_min + 1e-5, observation_max - 1e-5)

            return (new_noisy_goals,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals,), _ = jax.lax.scan(
            body_fn, (noisy_goals,), jnp.arange(self.config['num_flow_steps']))

        return noisy_goals

    def hilp_repr_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v1_t
        q2 = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def hilp_value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_hilp_critic')(batch['observations'], batch['latents'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('hilp_value')(batch['observations'], batch['latents'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def hilp_critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('hilp_value')(batch['next_observations'], batch['latents'])

        q1, q2 = self.network.select('hilp_critic')(
            batch['observations'], batch['latents'], batch['actions'], params=grad_params
        )
        q = batch['latent_rewards'] + self.config['discount'] * batch['masks'] * next_v
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def hilp_actor_loss(self, batch, grad_params):
        """Compute the actor loss."""
        v = self.network.select('hilp_value')(batch['observations'], batch['latents'])
        q1, q2 = self.network.select('hilp_critic')(batch['observations'], batch['latents'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['latents'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, latent_rng, fb_repr_rng, actor_rng, flow_occupancy_rng = jax.random.split(rng, 5)

        # sample latents
        batch['latents'], batch['latent_rewards'] = self.sample_latents(batch, latent_rng)

        hilp_repr_loss, hilp_repr_info = self.hilp_repr_loss(batch, grad_params)
        for k, v in hilp_repr_info.items():
            info[f'hilp_repr/{k}'] = v

        hilp_value_loss, hilp_value_info = self.hilp_value_loss(batch, grad_params)
        for k, v in hilp_value_info.items():
            info[f'hilp_value/{k}'] = v

        hilp_critic_loss, hilp_critic_info = self.hilp_critic_loss(batch, grad_params)
        for k, v in hilp_critic_info.items():
            info[f'hilp_critic/{k}'] = v

        hilp_actor_loss, hilp_actor_info = self.hilp_actor_loss(batch, grad_params)
        for k, v in hilp_actor_info.items():
            info[f'hilp_actor/{k}'] = v

        loss = hilp_repr_loss + hilp_value_loss + hilp_critic_loss + hilp_actor_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, latent_rng, critic_rng, flow_occupancy_rng, actor_rng = jax.random.split(
            rng, 5)

        # sample latents and mix
        batch['latents'], batch['latent_rewards'] = self.sample_latents(batch, latent_rng)

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        hilp_repr_loss, hilp_repr_info = self.hilp_repr_loss(batch, grad_params)
        for k, v in hilp_repr_info.items():
            info[f'hilp_repr/{k}'] = v

        hilp_value_loss, hilp_value_info = self.hilp_value_loss(batch, grad_params)
        for k, v in hilp_value_info.items():
            info[f'hilp_value/{k}'] = v

        hilp_critic_loss, hilp_critic_info = self.hilp_critic_loss(batch, grad_params)
        for k, v in hilp_critic_info.items():
            info[f'hilp_critic/{k}'] = v

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            batch, grad_params, flow_occupancy_rng)
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        if full_update:
            # Update the actor.
            hilp_actor_loss, hilp_actor_info = self.hilp_actor_loss(batch, grad_params)
            for k, v in hilp_actor_info.items():
                info[f'hilp_actor/{k}'] = v

            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            hilp_actor_loss, actor_loss = 0.0, 0.0

        loss = reward_loss + critic_loss + hilp_repr_loss + hilp_value_loss + hilp_critic_loss + flow_occupancy_loss + hilp_actor_loss + actor_loss
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
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'hilp_critic')
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'hilp_critic')
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
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'hilp_critic')
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_latents(self, batch, rng):
        batch_size = batch['observations'].shape[0]
        latents = jax.random.normal(rng, shape=(batch_size, self.config['latent_dim']),
                                    dtype=batch['actions'].dtype)
        latents = latents / jnp.linalg.norm(latents, axis=1, keepdims=True) * jnp.sqrt(
            self.config['latent_dim'])

        phis = self.get_phis(batch['observations'])
        next_phis = self.get_phis(batch['next_observations'])
        rewards = ((next_phis - phis) * latents).sum(axis=1)

        return latents, rewards


    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dummy_latents = jnp.zeros((*observations.shape[:-1], self.config['latent_dim']),
                                  dtype=self.config['action_dtype'])
        dist = self.network.select('actor')(observations, dummy_latents, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def get_phis(
        self,
        observations,
    ):
        """Return phi(s)."""
        _, phis, _ = self.network.select('value')(observations, observations, info=True)
        return phis[0]

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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        action_dtype = ex_actions.dtype
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']),
                              dtype=ex_actions.dtype)
        obs_dims = ex_observations.shape[1:]
        obs_dim = obs_dims[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['hilp_value'] = encoder_module()
            encoders['hilp_critic'] = encoder_module()
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        critic_vf_def = GCFMVectorField(
            vector_dim=obs_dim,
            time_sin_embedding=config['vector_field_time_sin_embedding'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        value_def = GCMetricValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            latent_dim=config['latent_dim'],
            num_ensembles=2,
            encoder=encoders.get('value'),
        )
        hilp_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            gc_encoder=encoders.get('hilp_value'),
        )
        hilp_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('hilp_critic'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_observations, ex_times,
                ex_observations, ex_actions, ex_latents)),
            value=(value_def, (ex_observations, ex_observations)),
            hilp_value=(hilp_value_def, (ex_observations, ex_latents)),
            hilp_critic=(hilp_critic_def, (ex_observations, ex_latents, ex_actions)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_observations, ex_times,
                ex_observations, ex_actions, ex_latents)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
            target_hilp_critic=(copy.deepcopy(hilp_critic_def), (ex_observations, ex_latents, ex_actions)),
            actor=(actor_def, (ex_observations, ex_latents)),
            reward=(reward_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        if config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']
        params['modules_target_value'] = params['modules_value']
        params['modules_target_hilp_critic'] = params['modules_hilp_critic']

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
            agent_name='hilp_fom',  # Agent name.
            obs_dims=ml_collections.config_dict.placeholder(tuple), # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            action_dtype=ml_collections.config_dict.placeholder(np.dtype),  # Action dtype (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            vector_field_time_sin_embedding=False,  # Whether to use time embedding in the vector field.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            reward_layer_norm=True,  # Whether to use layer normalization for the reward.
            latent_dim=512,  # Latent dimension for transition latents.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            num_flow_steps=10,  # Number of flow steps.
            num_flow_goals=4,  # Number of future flow goals for computing the target q.
            clip_flow_goals=False,  # Whether to clip the flow goals.
            expectile=0.9,  # IQL style expectile.
            actor_freq=2,  # Actor update frequency.
            q_agg='min',  # Aggregation method for target q.
            alpha=10.0,  # Temperature in BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.625,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.375,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            value_geom_start=1,  # Whether the support of the geometric sampling is [0, inf) or [1, inf)
            num_value_goals=1,  # Number of value goals to sample
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            actor_geom_start=1,  # Whether the support the geometric sampling is [0, inf) or [1, inf)
            num_actor_goals=1,  # Number of actor goals to sample
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            relabel_reward=False,  # Whether to relabel the reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
