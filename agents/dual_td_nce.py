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


class DualTDNCEAgent(flax.struct.PyTreeNode):
    """Dual TD NCE agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # def value_loss(self, batch, grad_params):
    #     """Compute the IQL value loss."""
    #     q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
    #     q = jnp.minimum(q1, q2)
    #     v = self.network.select('value')(batch['observations'], params=grad_params)
    #     value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
    #
    #     return value_loss, {
    #         'value_loss': value_loss,
    #         'v_mean': v.mean(),
    #         'v_max': v.max(),
    #         'v_min': v.min(),
    #     }

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
        logits1, logits2 = self.network.select('critic')(batch['observations'], params=grad_params)
        next_logits1, next_logits2 = self.network.select('critic')(batch['next_observations'], params=grad_params)

        target_logits1, target_logits2 = self.network.select('target_critic')(batch['observations'])
        target_logits = jnp.minimum(target_logits1, target_logits2)
        dist = self.network.select('actor')(batch['observations'])
        behavioral_dist = self.network.select('actor_bc')(batch['observations'])
        log_ratios = target_logits + dist.log_prob(batch['actions']) - behavioral_dist.log_prob(batch['actions'])
        ratios = jnp.exp(log_ratios)

        critic_loss1 = (
            (1 - self.config['discount']) * optax.losses.sigmoid_binary_cross_entropy(logits1, jnp.ones_like(logits1))
            + self.config['discount'] * ratios * optax.losses.sigmoid_binary_cross_entropy(next_logits1, jnp.ones_like(logits1))
            + optax.losses.sigmoid_binary_cross_entropy(logits1, jnp.zeros_like(logits1))
        )
        critic_loss2 = (
            (1 - self.config['discount']) * optax.losses.sigmoid_binary_cross_entropy(logits2, jnp.ones_like(logits2))
            + self.config['discount'] * ratios * optax.losses.sigmoid_binary_cross_entropy(next_logits2, jnp.ones_like(logits2))
            + optax.losses.sigmoid_binary_cross_entropy(logits2, jnp.zeros_like(logits2))
        )
        critic_loss = jnp.mean(critic_loss1) + jnp.mean(critic_loss2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'logits_mean': jnp.stack([logits1, logits2]).mean(),
            'logits_max': jnp.stack([logits1, logits2]).max(),
            'logits_min': jnp.stack([logits1, logits2]).min(),
            'ratios': ratios.mean(),
        }

    def actor_loss(self, batch, grad_params):
        """Compute the actor loss."""
        # DDPG+BC loss.
        # dist = self.network.select('actor')(batch['observations'], params=grad_params)
        # behavioral_dist = self.network.select('actor_bc')(batch['observations'])
        # if self.config['const_std']:
        #     q_actions = jnp.clip(dist.mode(), -1, 1)
        # else:
        #     q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        logits, logits2 = self.network.select('critic')(batch['observations'])
        logits = jnp.minimum(logits, logits2)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        behavioral_dist = self.network.select('actor_bc')(batch['observations'])
        log_ratios = logits + dist.log_prob(batch['actions']) - behavioral_dist.log_prob(batch['actions'])
        ratios = jnp.exp(log_ratios)

        # q1, q2 = self.network.select('critic')(batch['observations'], actions=q_actions)
        # q = jnp.minimum(q1, q2)
        q = ratios * batch['rewards']

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
        # log_prob = dist.log_prob(batch['actions'])
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
            agent_name='dual_td_nce',  # Agent name.
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
