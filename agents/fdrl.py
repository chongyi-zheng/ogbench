import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, ValueVectorField, ActorVectorField


class FDRLAgent(flax.struct.PyTreeNode):
    """Flow distributional reinforcement learning (FDRL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, rng):
        """Compute the IQL value loss."""
        batch_size = batch['actions'].shape[0]
        rng, noise_rng = jax.random.split(rng)

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        # q1 = (noises1 + self.network.select('target_critic_flow1')(
        #     noises1, jnp.zeros_like(noises1), batch['observations'], batch['actions'])).squeeze(-1)
        # q2 = (noises2 + self.network.select('target_critic_flow2')(
        #     noises2, jnp.zeros_like(noises2), batch['observations'], batch['actions'])).squeeze(-1)
        q1 = self.compute_flow_returns(
            noises, batch['observations'], batch['actions'],
            flow_network_name='target_critic_flow1').squeeze(-1)
        q2 = self.compute_flow_returns(
            noises, batch['observations'], batch['actions'],
            flow_network_name='target_critic_flow2').squeeze(-1)
        q = jnp.minimum(q1, q2)

        v = self.network.select('value_onestep_flow')(batch['observations'], noises, params=grad_params)
        v = jnp.clip(
            v,
            self.config['reward_min'] / (1 - self.config['discount']),
            self.config['reward_max'] / (1 - self.config['discount']),
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the flow critic loss."""
        batch_size = batch['actions'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng = jax.random.split(rng, 5)

        # next_dist = self.network.select('target_actor')(batch['next_observations'])
        # next_actions = next_dist.mode()
        # noise = jnp.clip(
        #     (jax.random.normal(actor_rng, next_actions.shape) * self.config['actor_noise']),
        #     -self.config['actor_noise_clip'],
        #     self.config['actor_noise_clip'],
        # )
        # next_actions = jnp.clip(next_actions + noise, -1, 1)

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        # returns1 = self.compute_flow_returns(
        #     noises1, batch['next_observations'], batch['next_actions'],
        #     flow_network_name='target_critic_flow1')
        # returns2 = self.compute_flow_returns(
        #     noises2, batch['next_observations'], batch['next_actions'],
        #     flow_network_name='target_critic_flow2')
        # noisy_next_returns1 = times * returns1 + (1 - times) * noises1
        # noisy_next_returns2 = times * returns2 + (1 - times) * noises2
        # noisy_next_returns = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
        next_returns = jnp.expand_dims(
            self.network.select('value_onestep_flow')(batch['next_observations'], noises), axis=-1)
        next_returns = jnp.clip(
            next_returns,
            self.config['reward_min'] / (1 - self.config['discount']),
            self.config['reward_max'] / (1 - self.config['discount']),
        )
        # The following returns will be bounded automatically
        returns = (jnp.expand_dims(batch['rewards'], axis=-1) +
                   self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * next_returns)
        noisy_returns = times * returns + (1 - times) * noises

        # transformed_noisy_returns = (
        #     batch['rewards'][..., None] + self.config['discount'] * batch['masks'][..., None] * noisy_next_returns)
        # transformed_noisy_next_returns = (batch['masks'][..., None] * noisy_returns - batch['rewards'][..., None]) / self.config['discount']
        # target_vector_field1 = self.network.select('target_critic_flow1')(
        #     noisy_next_returns, times, batch['next_observations'], batch['next_actions'])
        # target_vector_field2 = self.network.select('target_critic_flow2')(
        #     noisy_next_returns, times, batch['next_observations'], batch['next_actions'])
        # target_vector_field1 = returns1 - noises1
        # target_vector_field2 = returns2 - noises2
        target_vector_field = returns - noises

        vector_field1 = self.network.select('critic_flow1')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        vector_field2 = self.network.select('critic_flow2')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        vector_field_loss = ((vector_field1 - target_vector_field) ** 2 +
                             (vector_field2 - target_vector_field) ** 2).mean()

        if grad_params is not None:
            params1 = grad_params['modules_critic_flow1']
            params2 = grad_params['modules_critic_flow2']
        else:
            params1 = self.network.params['modules_critic_flow1']
            params2 = self.network.params['modules_critic_flow2']

        weight_l2_loss = sum(
            (w ** 2).sum()
            for w in jax.tree.leaves(params1)
        ) + sum(
            (w ** 2).sum()
            for w in jax.tree.leaves(params2)
        )

        # Additional metrics for logging.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q1 = self.compute_flow_returns(
            q_noises, batch['observations'], batch['actions'],
            flow_network_name='critic_flow1',
        ).squeeze(-1)
        q2 = self.compute_flow_returns(
            q_noises, batch['observations'], batch['actions'],
            flow_network_name='critic_flow2',
        ).squeeze(-1)
        q = jnp.minimum(q1, q2)
        # target_q = q_noises + self.network.select('critic_flow')(
        #     q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])
        # next_q = self.network.select('target_critic')(batch['next_observations'], next_actions)
        # mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        # next_q = next_q - self.config['alpha_critic'] * mse
        # target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        # q_loss = jnp.square(q - target_q).mean()

        critic_loss = vector_field_loss + self.config['alpha'] * weight_l2_loss

        return critic_loss, {
            'vector_field_loss': vector_field_loss,
            'weight_l2_loss': weight_l2_loss,
            # 'q_loss': q_loss,
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    # def actor_loss(self, batch, grad_params, rng):
    #     """Compute the actor loss (DDPG+BC)."""
    #     # batch_size = batch['actions'].shape[0]
    #     # rng, sample_rng, noise_rng = jax.random.split(rng, 3)
    #
    #     dist = self.network.select('actor')(batch['observations'], params=grad_params)
    #     q_actions = jnp.clip(dist.mode(), -1, 1)
    #
    #     # rng, noise_rng = jax.random.split(rng)
    #     # noises = jax.random.normal(noise_rng, (batch_size, 1))
    #     # q = noises + self.network.select('critic_flow')(
    #     #     noises, jnp.zeros_like(noises), batch['observations'], q_actions)
    #     q = self.network.select('critic')(batch['observations'], q_actions)
    #
    #     q_loss = -q.mean()
    #     if self.config['normalize_q_loss']:
    #         # Normalize Q values by the absolute mean to make the loss scale invariant.
    #         lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
    #         q_loss = lam * q_loss
    #
    #     # BC loss.
    #     mse = jnp.square(q_actions - batch['actions']).sum(axis=-1)
    #     bc_loss = (self.config['alpha_actor'] * mse).mean()
    #
    #     actor_loss = q_loss + bc_loss
    #
    #     if self.config['tanh_squash']:
    #         action_std = dist._distribution.stddev()
    #     else:
    #         action_std = dist.stddev().mean()
    #
    #     return actor_loss, {
    #         'actor_loss': actor_loss,
    #         'q_loss': q_loss,
    #         'bc_loss': bc_loss,
    #         'std': action_std.mean(),
    #         'mse': mse.mean(),
    #     }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * x_1 + (1 - t) * x_0
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, value_rng, critic_rng, actor_rng = jax.random.split(rng, 4)

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
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
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # self.target_update(new_network, 'critic')
        # self.target_update(new_network, 'actor')
        self.target_update(new_network, 'critic_flow1')
        self.target_update(new_network, 'critic_flow2')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flow_network_name',))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name='critic_flow',
    ):
        """Compute returns from the return flow model using the Euler method."""
        noisy_returns = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_returns, ) = carry

            times = i * step_size + init_times
            vector_field = self.network.select(flow_network_name)(
                noisy_returns, times, observations, actions)
            new_noisy_returns = noisy_returns + vector_field * step_size
            new_noisy_returns = jnp.clip(
                new_noisy_returns,
                self.config['reward_min'] / (1 - self.config['discount']),
                self.config['reward_max'] / (1 - self.config['discount']),
            )

            return (new_noisy_returns, ), None

        # Use lax.scan to do the iteration
        (noisy_returns, ), _ = jax.lax.scan(
            func, (noisy_returns,), jnp.arange(self.config['num_flow_steps']))
        noisy_returns = jnp.clip(
            noisy_returns,
            self.config['reward_min'] / (1 - self.config['discount']),
            self.config['reward_max'] / (1 - self.config['discount']),
        )

        return noisy_returns

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
        init_times=None,
        end_times=None,
    ):
        noisy_actions = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_actions, ) = carry

            times = i * step_size + init_times
            vector_field = self.network.select('actor_flow')(
                observations, noisy_actions, times)
            new_noisy_actions = noisy_actions + vector_field * step_size
            new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)

            return (new_noisy_actions, ), None

        # Use lax.scan to do the iteration
        (noisy_actions, ), _ = jax.lax.scan(
            func, (noisy_actions,), jnp.arange(self.config['num_flow_steps']))

        noisy_actions = jnp.clip(noisy_actions, -1, 1)

        return noisy_actions


    # @jax.jit
    # def sample_actions(
    #     self,
    #     observations,
    #     seed=None,
    #     temperature=1.0,
    # ):
    #     """Sample actions from the actor."""
    #     dist = self.network.select('actor')(observations, temperature=temperature)
    #     actions = dist.mode()
    #     noise = jnp.clip(
    #         (jax.random.normal(seed, actions.shape) * self.config['actor_noise'] * temperature),
    #         -self.config['actor_noise_clip'],
    #         self.config['actor_noise_clip'],
    #     )
    #     actions = jnp.clip(actions + noise, -1, 1)
    #     return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        seed, action_seed, q_seed = jax.random.split(seed, 3)

        # Sample `num_samples` noises and propagate them through the flow.
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        actions = self.compute_flow_actions(noises, n_observations)

        # Pick the action with the highest Q-value.
        # q = self.network.select('critic')(n_observations, actions=actions).min(axis=0)
        q_noises = jax.random.normal(q_seed, (self.config['num_samples'], 1))
        q1 = self.compute_flow_returns(
            q_noises, n_observations, actions,
            flow_network_name='critic_flow1').squeeze(-1)
        q2 = self.compute_flow_returns(
            q_noises, n_observations, actions,
            flow_network_name='critic_flow2').squeeze(-1)
        q = jnp.minimum(q1, q2)
        actions = actions[jnp.argmax(q)]
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_returns = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        min_reward = example_batch['min_reward']
        max_reward = example_batch['max_reward']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value_onestep_flow'] = encoder_module()
            encoders['critic_flow'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        value_onestep_flow_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value_onestep_flow'),
        )
        critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        # actor_def = Actor(
        #     hidden_dims=config['actor_hidden_dims'],
        #     action_dim=action_dim,
        #     layer_norm=config['actor_layer_norm'],
        #     tanh_squash=config['tanh_squash'],
        #     state_dependent_std=False,
        #     const_std=True,
        #     final_fc_init_scale=config['actor_fc_scale'],
        #     encoder=encoders.get('actor'),
        # )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            # critic=(critic_def, (ex_observations, ex_actions)),
            # target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            value_onestep_flow=(value_onestep_flow_def, (ex_observations, ex_returns)),
            critic_flow1=(critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            critic_flow2=(critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow1=(copy.deepcopy(critic_flow1_def), (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow2=(copy.deepcopy(critic_flow2_def), (ex_returns, ex_times, ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
            # actor=(actor_def, (ex_observations,)),
            # target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow1'] = params['modules_critic_flow1']
        params['modules_target_critic_flow2'] = params['modules_critic_flow2']
        # params['modules_target_critic'] = params['modules_critic']
        # params['modules_target_actor'] = params['modules_actor']

        config['action_dim'] = action_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fdrl',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            min_reward=ml_collections.config_dict.placeholder(float),  # Minimum reward (will be set automatically).
            max_reward=ml_collections.config_dict.placeholder(float),  # Maximum reward (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=1.0,  # Weight L2 norm regularization coefficient.
            num_samples=32,  # Number of action samples for rejection sampling.
            num_flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
