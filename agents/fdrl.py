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
        returns = self.compute_flow_returns(
            noises, batch['next_observations'], batch['next_actions'])
        noisy_next_returns = times * returns + (1 - times) * noises

        transformed_noisy_returns = (
            batch['rewards'][..., None] + self.config['discount'] * batch['masks'][..., None] * noisy_next_returns)
        # transformed_noisy_next_returns = (batch['masks'][..., None] * noisy_returns - batch['rewards'][..., None]) / self.config['discount']
        target_vector_field = self.network.select('target_critic_flow')(
            noisy_next_returns, times, batch['next_observations'], batch['next_actions'])

        vector_field = self.network.select('critic_flow')(
            transformed_noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        vector_field_loss = jnp.square(vector_field - target_vector_field).mean()

        # Additional metrics for logging.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q = self.compute_flow_returns(
            q_noises, batch['observations'], batch['actions'],
            use_target_flow=False,
        ).squeeze(-1)
        # target_q = q_noises + self.network.select('critic_flow')(
        #     q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])
        # next_q = self.network.select('target_critic')(batch['next_observations'], next_actions)
        # mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        # next_q = next_q - self.config['alpha_critic'] * mse
        # target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        # q_loss = jnp.square(q - target_q).mean()

        critic_loss = vector_field_loss

        return critic_loss, {
            'vector_field_loss': vector_field_loss,
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
        x_t = (1 - t) * x_0 + t * x_1
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
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        self.target_update(new_network, 'critic_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('use_target_flow',))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        use_target_flow=True,
    ):
        """Compute returns from the return flow model using the Euler method."""
        if use_target_flow:
            flow_network_name = 'target_critic_flow'
        else:
            flow_network_name = 'critic_flow'

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

            return (new_noisy_returns, ), None

        # Use lax.scan to do the iteration
        (noisy_returns, ), _ = jax.lax.scan(
            func, (noisy_returns,), jnp.arange(self.config['num_flow_steps']))

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
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        # q = self.network.select('critic')(n_observations, actions=actions).min(axis=0)
        q_noises = jax.random.normal(q_seed, (self.config['num_samples'], 1))
        q = self.compute_flow_returns(q_noises, n_observations, actions).squeeze(-1)
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

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['critic_flow'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        # critic_def = Value(
        #     hidden_dims=config['value_hidden_dims'],
        #     layer_norm=config['value_layer_norm'],
        #     num_ensembles=1,
        #     encoder=encoders.get('critic'),
        # )
        critic_flow_def = ValueVectorField(
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
            critic_flow=(critic_flow_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow=(copy.deepcopy(critic_flow_def), (ex_returns, ex_times, ex_observations, ex_actions)),
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
        params['modules_target_critic_flow'] = params['modules_critic_flow']
        # params['modules_target_critic'] = params['modules_critic']
        # params['modules_target_actor'] = params['modules_actor']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fdrl',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha_actor=1.0,  # Actor BC coefficient.
            alpha_critic=1.0,  # Critic BC coefficient.
            num_samples=32,  # Number of action samples for rejection sampling.
            num_flow_steps=10,  # Number of flow steps.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
