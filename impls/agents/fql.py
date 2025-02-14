import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDiscreteActor, GCFMVectorField, GCFMValue, GCValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""

        rng, next_noise_rng = jax.random.split(rng)
        next_noises = jax.random.normal(next_noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        if self.config['distill_type'] == 'fwd_sample':
            next_actions = self.network.select('actor')(next_noises, batch['next_observations'])
        elif self.config['distill_type'] == 'fwd_int':
            next_action_vfs = self.network.select('actor')(next_noises, batch['next_observations'])
            next_actions = next_noises + next_action_vfs
        next_actions = jnp.clip(next_actions, -1, 1)

        if self.config['vf_q_loss']:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], actions=next_action_vfs)
        else:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'mean':
            next_q = next_qs.mean(axis=0)
        else:
            next_q = next_qs.min(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        if self.config['vf_q_loss']:
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
            action_vfs = self.network.select('actor')(noises, batch['observations'])
            q = self.network.select('critic')(
                batch['observations'], action_vfs, params=grad_params)
        else:
            q = self.network.select('critic')(
                batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

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

        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        path_sample = self.cond_prob_path(x_0=noises, x_1=actions, t=times)
        vf_pred = self.network.select('actor_vf')(
            path_sample.x_t,
            times,
            observations,
            params=grad_params,
        )
        flow_matching_loss = jnp.square(vf_pred - path_sample.dx_t).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        if self.config['distill_type'] == 'fwd_sample':
            q_actions = self.network.select('actor')(noises, batch['observations'], params=grad_params)
        elif self.config['distill_type'] == 'fwd_int':
            q_action_vfs = self.network.select('actor')(noises, batch['observations'], params=grad_params)
            q_actions = noises + q_action_vfs
        # TODO (chongyiz): we need to clip q_actions here?
        q_actions = jnp.clip(q_actions, -1, 1)

        flow_actions = self.compute_fwd_flow_samples(noises, batch['observations'])
        flow_actions = jnp.clip(flow_actions, -1, 1)

        # distill loss
        distill_loss = self.config['alpha'] * jnp.pow(q_actions - flow_actions, 2).mean()

        # Q loss
        if self.config['vf_q_loss']:
            qs = self.network.select('critic')(batch['observations'], actions=q_action_vfs)
        else:
            qs = self.network.select('critic')(batch['observations'], actions=q_actions)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss
        actor_loss = q_loss + distill_loss

        # Additional metrics for logging.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        if self.config['distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(noises, batch['observations'])
        elif self.config['distill_type'] == 'fwd_int':
            action_vfs = self.network.select('actor')(noises, batch['observations'])
            actions = noises + action_vfs
        actions = jnp.clip(actions, -1, 1)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'mse': mse,
        }

    def compute_fwd_flow_samples(self, noises, observations):
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
            vf = self.network.select('actor_vf')(noisy_actions, times, observations)
            new_noisy_actions = noisy_actions + vf * step_size

            return (new_noisy_actions, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_actions,), _ = jax.lax.scan(
            body_fn, (noisy_actions,), jnp.arange(num_flow_steps))

        return noisy_actions

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, flow_matching_rng, actor_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + flow_matching_loss + actor_loss
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
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        action_dim = self.network.model_def.modules['actor'].output_dim

        seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            noise_seed, shape=(observations.shape[0], action_dim), dtype=observations.dtype)
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
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        rng, time_rng = jax.random.split(rng)
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],))
        # ex_noises = jax.random.normal(noise_rng, shape=ex_actions.shape, dtype=ex_actions.dtype)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_vf'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            # critic_def = GCDiscreteBilinearCritic(
            #     hidden_dims=config['value_hidden_dims'],
            #     latent_dim=config['latent_dim'],
            #     layer_norm=config['layer_norm'],
            #     ensemble=True,
            #     value_exp=True,
            #     state_encoder=encoders.get('critic_state'),
            #     goal_encoder=encoders.get('critic_goal'),
            #     action_dim=action_dim,
            # )
            raise NotImplementedError
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=2,
                gc_encoder=encoders.get('critic'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_vf_def = GCFMVectorField(
                vector_dim=action_dim,
                hidden_dims=config['actor_hidden_dims'],
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor_vf'),
            )
            actor_def = GCFMValue(
                hidden_dims=config['actor_hidden_dims'],
                output_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor'),
            )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations)),
            actor=(actor_def, (ex_actions, ex_observations)),
        )
        # if encoders.get('actor_bc_flow') is not None:
        #     # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
        #     network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='fql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            distill_type='fwd_sample',  # Distillation type. ('fwd_sample', 'fwd_int').
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            num_flow_steps=10,  # Number of flow steps.
            discrete=False,  # Whether the action space is discrete.
            vf_q_loss=False,  # Whether to use vector fields to compute Q in the Q loss.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
