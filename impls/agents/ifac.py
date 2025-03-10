import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCFMValue, Value
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class IFACAgent(flax.struct.PyTreeNode):
    """Implicit Flow Actor Critic agent."""

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
        if self.config['reward_type'] == 'state_action':
            actions = batch['actions']
        else:
            actions = None
        rewards = batch['rewards']
        
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_critic_encoder')(observations)
        reward_preds = self.network.select('reward')(
            observations, actions=actions,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        goals = batch['value_goals']

        if self.config['encoder'] is not None:
            observations = self.network.select('value_vf_encoder')(batch['observations'])
            goals = self.network.select('value_vf_encoder')(batch['value_goals'])

        rng, noise_rng = jax.random.split(rng)
        if self.config['value_noise_type'] == 'normal':
            noises = jax.random.normal(noise_rng, shape=observations.shape, dtype=observations.dtype)
        elif self.config['value_noise_type'] == 'marginal_state':
            noises = jax.random.permutation(noise_rng, observations, axis=0)
        elif self.config['value_noise_type'] == 'marginal_goal':
            noises = jax.random.permutation(noise_rng, goals, axis=0)
        flow_goals = self.compute_fwd_flow_goals(noises, observations)
        flow_goals = jnp.clip(flow_goals, self.config['dataset_obs_min'], self.config['dataset_obs_max'])

        if self.config['reward_type'] == 'state_action':
            rng, noise_rng = jax.random.split(rng)
            a_noises = jax.random.normal(
                noise_rng, shape=actions.shape, dtype=actions.dtype)
            if self.config['distill_type'] == 'fwd_sample':
                goal_actions = self.network.select('actor')(a_noises, flow_goals)
            elif self.config['distill_type'] == 'fwd_int':
                goal_action_vfs = self.network.select('actor')(a_noises, flow_goals)
                goal_actions = a_noises + goal_action_vfs
            goal_actions = jnp.clip(goal_actions, -1, 1)
        else:
            assert self.config['reward_type'] == 'state'
            goal_actions = None

        if self.config['use_target_reward']:
            future_rewards = self.network.select('target_reward')(flow_goals, actions=goal_actions)
        else:
            future_rewards = self.network.select('reward')(flow_goals, actions=goal_actions)

        target_v = rewards + self.config['discount'] / (1 - self.config['discount']) * future_rewards
        v = self.network.select('value')(observations, params=grad_params)
        value_loss = self.expectile_loss(target_v - v, target_v - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the critic loss."""

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        masks = batch['masks']
        next_observations = batch['next_observations']

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_critic_encoder')(
                batch['observations'], params=grad_params)

        qs = self.network.select('critic')(
            observations, actions,
            params=grad_params,
        )

        next_v = self.network.select('value')(next_observations)
        target_q = rewards + self.config['discount'] * masks * next_v
        critic_loss = jnp.square(target_q - qs).mean()

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
        goals = batch['value_goals']
        
        if self.config['encoder'] is not None:
            observations = self.network.select('value_vf_encoder')(
                batch['observations'], params=grad_params)
            goals = self.network.select('value_vf_encoder')(
                batch['value_goals'], params=grad_params)

        # value flow matching
        rng, value_noise_rng, value_time_rng = jax.random.split(rng, 3)
        if self.config['value_noise_type'] == 'normal':
            value_noises = jax.random.normal(value_noise_rng, shape=goals.shape, dtype=actions.dtype)
        elif self.config['value_noise_type'] == 'marginal_state':
            value_noises = jax.random.permutation(value_noise_rng, observations, axis=0)
        elif self.config['value_noise_type'] == 'marginal_goal':
            value_noises = jax.random.permutation(value_noise_rng, goals, axis=0)
        value_times = jax.random.uniform(value_time_rng, shape=(batch_size, ))
        value_path_sample = self.cond_prob_path(x_0=value_noises, x_1=goals, t=value_times)
        value_vf_pred = self.network.select('value_vf')(
            value_path_sample.x_t,
            value_times,
            observations,
            params=grad_params,
        )
        value_flow_matching_loss = jnp.square(value_vf_pred - value_path_sample.dx_t).mean()

        # actor flow matching
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_critic_encoder')(
                batch['observations'])  # no gradients for the encoder
        
        rng, actor_noise_rng, actor_time_rng = jax.random.split(rng, 3)
        actor_noises = jax.random.normal(actor_noise_rng, shape=actions.shape, dtype=actions.dtype)
        actor_times = jax.random.uniform(actor_time_rng, shape=(batch_size, ))
        actor_path_sample = self.cond_prob_path(x_0=actor_noises, x_1=actions, t=actor_times)
        actor_vf_pred = self.network.select('actor_vf')(
            actor_path_sample.x_t,
            actor_times,
            observations,
            params=grad_params,
        )
        actor_flow_matching_loss = jnp.square(actor_vf_pred - actor_path_sample.dx_t).mean()

        flow_matching_loss = value_flow_matching_loss + actor_flow_matching_loss

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
            'value_flow_matching_loss': value_flow_matching_loss,
            'actor_flow_matching_loss': actor_flow_matching_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""

        observations = batch['observations']
        actions = batch['actions']

        if self.config['encoder'] is not None:
            if self.config['encoder_actor_loss_grad']:
                observations = self.network.select('actor_critic_encoder')(observations, params=grad_params)
            else:
                observations = self.network.select('actor_critic_encoder')(observations)

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
        qs = self.network.select('critic')(observations, actions=q_actions)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
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

            return (new_noisy_actions, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_actions,), _ = jax.lax.scan(
            body_fn, (noisy_actions,), jnp.arange(num_flow_steps))

        return noisy_actions

    def compute_fwd_flow_goals(self, noises, observations):
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
            vf = self.network.select('value_vf')(
                noisy_goals, times, observations)
            new_noisy_goals = noisy_goals + vf * step_size

            return (new_noisy_goals, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, ), _ = jax.lax.scan(
            body_fn, (noisy_goals, ), jnp.arange(num_flow_steps))

        return noisy_goals

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, value_rng, flow_matching_rng, actor_rng = jax.random.split(rng, 4)

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = reward_loss + value_loss + critic_loss + flow_matching_loss + actor_loss
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
        self.target_update(new_network, 'reward')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        if len(observations.shape) == 1 or len(observations.shape) == 3:
            batch_size = 1
        else:
            batch_size = observations.shape[0]
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_critic_encoder')(observations)
        action_dim = self.network.model_def.modules['actor'].output_dim

        seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(noise_seed, shape=(batch_size, action_dim)).squeeze()
        if self.config['distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(noises, observations)
        elif self.config['distill_type'] == 'fwd_int':
            actions = noises + self.network.select('actor')(noises, observations)
        actions = jnp.clip(actions, -1, 1)
        # actions = actions.squeeze()

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

        assert config['dataset_obs_min'] is not None
        assert config['dataset_obs_max'] is not None

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        ex_orig_observations = ex_observations
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],), dtype=ex_actions.dtype)
        # ex_noises = jax.random.normal(noise_rng, shape=ex_actions.shape, dtype=ex_actions.dtype)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if 'mlp_hidden_dims' in encoder_module.keywords:
                observation_dim = encoder_module.keywords['mlp_hidden_dims'][-1]
            else:
                observation_dim = encoder_modules['impala'].mlp_hidden_dims[-1]
            rng, obs_rng = jax.random.split(rng, 2)
            ex_observations = jax.random.normal(
                obs_rng, shape=(ex_observations.shape[0], observation_dim), dtype=ex_actions.dtype)
            
            encoders['value_vf'] = encoder_module()
            # encoders['critic'] = encoder_module()
            # encoders['actor_vf'] = encoder_module()
            # encoders['actor'] = encoder_module()
            # encoders['reward'] = encoder_module()
            encoders['actor_critic'] = encoder_module()

        # Define value and actor networks.
        critic_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            # encoder=encoders.get('actor_critic'),
        )
        value_vf_def = GCFMVectorField(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            vector_dim=observation_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        value_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )

        actor_vf_def = GCFMVectorField(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            vector_dim=action_dim,
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            # state_encoder=encoders.get('actor_critic'),
        )
        actor_def = GCFMValue(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['actor_hidden_dims'],
            output_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            # state_encoder=encoders.get('actor_critic'),
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
            value_vf=(value_vf_def, (
                ex_observations, ex_times, ex_observations)),
            value=(value_def, (ex_observations,)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations)),
            actor=(actor_def, (ex_actions, ex_observations)),
        )
        if config['reward_type'] == 'state':
            network_info.update(
                reward=(reward_def, (ex_observations, )),
                target_reward=(copy.deepcopy(reward_def), (ex_observations, )),
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
        if encoders.get('value_vf') is not None:
            network_info['value_vf_encoder'] = (encoders.get('value_vf'), (ex_orig_observations,))
        if encoders.get('actor_critic') is not None:
            network_info['actor_critic_encoder'] = (encoders.get('actor_critic'), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_reward'] = params['modules_reward']

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='ifac',  # Agent name.
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
            critic_loss_type='mse',  # Critic loss type. ('mse', 'expectile').
            value_noise_type='normal',  # Critic noise type. ('marginal_state', 'marginal_goal', 'normal').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            distill_type='fwd_sample',  # Distillation type. ('fwd_sample', 'fwd_int').
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            use_target_reward=True,  # Whether to use the target reward network.
            reward_type='state',  # Reward type. ('state', 'state_action')
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            encoder_actor_loss_grad=False,  # Whether to backpropagate gradients from the actor loss into the encoder.
            # Dataset hyperparameters.
            relabel_reward=False,  # Whether to relabel the reward.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            num_value_goals=1,  # Number of value goals to sample
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            num_actor_goals=1,  # Number of actor goals to sample
            dataset_obs_min=ml_collections.config_dict.placeholder(jnp.ndarray),
            dataset_obs_max=ml_collections.config_dict.placeholder(jnp.ndarray),
        )
    )
    return config
