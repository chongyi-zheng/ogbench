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


class MCFACAgent(flax.struct.PyTreeNode):
    """Monte Carlo Flow Actor-Critic (FAC) agent."""

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        rewards = batch['rewards']

        reward_preds = self.network.select('reward')(
            observations,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""

        # rng, next_noise_rng = jax.random.split(rng)
        # next_noises = jax.random.normal(next_noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        # next_action_vfs = self.network.select('actor')(next_noises, batch['next_observations'])
        # next_actions = next_noises + next_action_vfs
        # next_actions = jnp.clip(next_actions, -1, 1)
        #
        # next_qs = self.network.select('target_critic')(
        #     batch['next_observations'], actions=next_actions)
        # if self.config['q_agg'] == 'mean':
        #     next_q = next_qs.mean(axis=0)
        # else:
        #     next_q = next_qs.min(axis=0)
        #
        # target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        #
        # q = self.network.select('critic')(
        #     batch['observations'], actions=batch['actions'], params=grad_params)
        # critic_loss = jnp.square(q - target_q).mean()

        # return critic_loss, {
        #     'critic_loss': critic_loss,
        #     'q_mean': q.mean(),
        #     'q_max': q.max(),
        #     'q_min': q.min(),
        # }

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

        if self.config['distill_type'] == 'fwd_int':
            rng, g_rng = jax.random.split(rng)
            shortcut_noises = jax.random.normal(g_rng, shape=goals.shape, dtype=goals.dtype)
            sampled_goals = self.compute_fwd_flow_samples(
                shortcut_noises, observations, actions=actions)

            shortcut_goal_preds = shortcut_noises + self.network.select('critic')(
                shortcut_noises, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(shortcut_goal_preds - sampled_goals[None]).mean()
        else:
            distill_loss = 0.0
        critic_loss = distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'distillation_loss': distill_loss,
        }

    def flow_matching_loss(self, batch, grad_params, rng):
        """Compute the flow matching loss."""

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

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
        actor_flow_matching_loss = jnp.square(vf_pred - path_sample.dx_t).mean()

        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        times = jax.random.uniform(time_rng, shape=(batch_size,))
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select('critic_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        critic_flow_matching_loss = jnp.square(vf_pred - path_sample.dx_t).mean()

        flow_matching_loss = actor_flow_matching_loss + critic_flow_matching_loss

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
            'actor_flow_matching_loss': actor_flow_matching_loss,
            'critic_flow_matching_loss': critic_flow_matching_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        q_action_vfs = self.network.select('actor')(noises, batch['observations'], params=grad_params)
        q_actions = noises + q_action_vfs
        # TODO (chongyiz): we need to clip q_actions here?
        q_actions = jnp.clip(q_actions, -1, 1)

        flow_actions = self.compute_fwd_flow_samples(noises, batch['observations'])
        flow_actions = jnp.clip(flow_actions, -1, 1)

        # distill loss
        distill_loss = self.config['alpha'] * jnp.pow(q_actions - flow_actions, 2).mean()

        # Q loss
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
        actions = noises + self.network.select('actor')(noises, batch['observations'])
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

    def compute_fwd_flow_samples(self, noises, observations, actions=None, module_name='actor_vf'):
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
            vf = self.network.select(module_name)(noisy_actions, times, observations, actions=actions)
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

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

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

        ex_goals = ex_observations
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
            encoders['critic_vf'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_vf'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['reward'] = encoder_module()

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
            critic_vf_def = GCFMVectorField(
                vector_dim=ex_observations.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                state_encoder=encoders.get('critic_vf'),
            )
            critic_def = GCFMValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                output_dim=action_dim,
                state_encoder=encoders.get('critic'),
            )
            # critic_def = GCValue(
            #     hidden_dims=config['value_hidden_dims'],
            #     layer_norm=config['layer_norma'],
            #     num_ensembels=2,
            #     gc_encoder=encoders.get('critic'),
            # )

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

        reward_def = GCValue(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('reward'),
        )

        network_info = dict(
            critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations)),
            actor=(actor_def, (ex_actions, ex_observations)),
            reward=(reward_def, (ex_observations,)),
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

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='fac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            num_flow_steps=10,  # Number of flow steps.
            discrete=False,  # Whether the action space is discrete.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            relabel_reward=False,  # Whether to relabel the reward.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric samling for future value goals.
            num_value_goals=1,  # Number of value goals to sample
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            num_actor_goals=1,  # Number of actor goals to sample
        )
    )
    return config
