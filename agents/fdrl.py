import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, ValueVectorField


class FDRLAgent(flax.struct.PyTreeNode):
    """Flow distributional reinforcement learning (FDRL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the flow critic loss."""
        batch_size = batch['actions'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng = jax.random.split(rng, 5)

        next_dist = self.network.select('actor')(batch['next_observations'])
        if self.config['const_std']:
            next_actions = jnp.clip(next_dist.mode(), -1, 1)
        else:
            next_actions = jnp.clip(next_dist.sample(seed=rng), -1, 1)

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        next_returns = self.compute_flow_returns(
            noises, batch['next_observations'], next_actions)
        noisy_next_returns = times * next_returns + (1 - times) * noises

        transformed_noisy_returns = (
            batch['rewards'][..., None] + self.config['discount'] * batch['masks'][..., None] * noisy_next_returns)
        target_vector_field = self.network.select('target_critic_flow')(
            noisy_next_returns, times, batch['next_observations'], next_actions)

        vector_field = self.network.select('critic_flow')(
            transformed_noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(vector_field - target_vector_field).mean()

        # Additional metrics for logging.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q = q_noises + self.network.select('critic_flow')(
            q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss (DDPG+BC)."""
        batch_size = batch['actions'].shape[0]
        rng, sample_rng, noise_rng = jax.random.split(rng, 3)

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=sample_rng), -1, 1)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, 1))
        q = noises + self.network.select('critic_flow')(
            noises, jnp.zeros_like(noises), batch['observations'], q_actions)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            # Normalize Q values by the absolute mean to make the loss scale invariant.
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        log_prob = dist.log_prob(batch['actions'])
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
        self.target_update(new_network, 'critic_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
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
            encoders['critic_flow'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic_flow=(critic_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow=(copy.deepcopy(critic_def), (ex_returns, ex_times, ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow'] = params['modules_critic_flow']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fdrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=10.0,  # BC coefficient in DDPG+BC.
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
