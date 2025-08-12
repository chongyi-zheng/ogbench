import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class DualLSIFAgent(flax.struct.PyTreeNode):
    """Dual least square importance filtering agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def behavioral_cloning_loss(self, batch, grad_params):
        """Compute the behavioral cloning loss."""
        dist = self.network.select('actor_bc')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -log_prob.mean()

        return bc_loss, {
            'bc_loss': bc_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the critic loss."""
        initial_ratios1, initial_ratios2 = self.network.select('critic')(
            batch['initial_observations'], params=grad_params)
        initial_ratios1, initial_ratios2 = jax.nn.relu(initial_ratios2), jax.nn.relu(initial_ratios2)
        next_ratios1, next_ratios2 = self.network.select('critic')(
            batch['next_observations'], params=grad_params)
        next_ratios1, next_ratios2 = jax.nn.relu(next_ratios1), jax.nn.relu(next_ratios2)

        target_ratios1, target_ratios2 = self.network.select('target_critic')(batch['observations'])
        target_ratios = jnp.minimum(target_ratios1, target_ratios2)
        target_ratios = jax.nn.relu(target_ratios)
        dist = self.network.select('actor')(batch['observations'])
        behavioral_dist = self.network.select('actor_bc')(batch['observations'])
        log_weights = dist.log_prob(batch['actions']) - behavioral_dist.log_prob(batch['actions'])
        weights = jnp.exp(log_weights)

        critic_loss1 = (
            0.5 * (next_ratios1 - self.config['discount'] * weights * target_ratios) ** 2
            - (1 - self.config['discount']) * initial_ratios1
        )
        critic_loss2 = (
            0.5 * (next_ratios2 - self.config['discount'] * weights * target_ratios) ** 2
            - (1 - self.config['discount']) * initial_ratios2
        )
        critic_loss = jnp.mean(critic_loss1) + jnp.mean(critic_loss2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'next_ratios_mean': jnp.stack([next_ratios1, next_ratios2]).mean(),
            'next_ratios_max': jnp.stack([next_ratios1, next_ratios2]).max(),
            'next_ratios_min': jnp.stack([next_ratios1, next_ratios2]).min(),
            'initial_ratios_mean': jnp.stack([initial_ratios1, initial_ratios2]).mean(),
            'target_ratios_mean': jnp.stack([target_ratios1, target_ratios2]).mean(),
        }

    def actor_loss(self, batch, grad_params):
        """Compute the actor loss."""
        ratios1, ratios2 = self.network.select('critic')(batch['observations'])
        ratios = jnp.minimum(ratios1, ratios2)
        ratios = jax.nn.relu(ratios)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        behavioral_dist = self.network.select('actor_bc')(batch['observations'])
        log_weights = dist.log_prob(batch['actions']) - behavioral_dist.log_prob(batch['actions'])
        weights = jnp.exp(log_weights)

        q = weights * ratios * batch['rewards']

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        bc_loss, bc_info = self.behavioral_cloning_loss(batch, grad_params)
        for k, v in bc_info.items():
            info[f'behavioral_cloning/{k}'] = v

        loss = critic_loss + actor_loss + bc_loss
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
        self.target_update(new_network, 'critic')

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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )
        actor_bc_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations,)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations,)),
            actor=(actor_def, (ex_observations,)),
            actor_bc=(actor_bc_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dual_lsif',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization for the critic.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            const_std=False,  # Whether to use constant standard deviation for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
