import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDiscreteActor, GCActor, GCFMVectorField, GCFMValue, GCValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class TDFMRLAgent(flax.struct.PyTreeNode):
    """Temporal Difference Flow Matching RL (FMRL) agent.
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
        if self.config['reward_type'] == 'state_action':
            actions = batch['actions']
        else:
            actions = None
        rewards = batch['rewards']

        reward_preds = self.network.select('reward')(
            observations, actions=actions,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        qs = self.network.select('critic')(observations, actions, params=grad_params)

        rng, g_noise_rng, a_noise_rng = jax.random.split(rng, 3)
        if self.config['critic_noise_type'] == 'normal':
            g_noises = jax.random.normal(g_noise_rng, shape=observations.shape, dtype=observations.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            g_noises = jax.random.permutation(g_noise_rng, observations, axis=0)
        else:
            raise NotImplementedError
        flow_goals = self.compute_fwd_flow_goals(g_noises, observations, actions)

        if self.config['reward_type'] == 'state_action':
            dist = self.network.select('actor')(flow_goals)
            if self.config['const_std']:
                goal_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                rng, goal_action_rng = jax.random.split(rng)
                goal_actions = jnp.clip(dist.sample(seed=goal_action_rng), -1, 1)
        else:
            assert self.config['reward_type'] == 'state'
            goal_actions = None

        target_q = ((1 - self.config['discount']) * rewards
                    + self.config['discount'] * self.network.select('reward')(flow_goals, actions=goal_actions))

        if self.config['critic_loss_type'] == 'mse':
            critic_loss = jnp.square(target_q - qs).mean()
        elif self.config['critic_loss_type'] == 'expectile':
            critic_loss = self.expectile_loss(
                target_q - qs, target_q - qs, self.config['expectile']).mean()

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
        next_observations = batch['next_observations']
        actions = batch['actions']

        # critic flow matching
        rng, next_time_rng, next_noise_rng = jax.random.split(rng, 3)
        next_times = jax.random.uniform(next_time_rng, shape=(batch_size, ), dtype=next_observations.dtype)
        if self.config['critic_noise_type'] == 'normal':
            next_noises = jax.random.normal(next_noise_rng, shape=next_observations.shape, dtype=next_observations.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            next_noises = jax.random.permutation(next_noise_rng, observations, axis=0)
        next_path_sample = self.cond_prob_path(x_0=next_noises, x_1=next_observations, t=next_times)
        next_vf_pred = self.network.select('critic_vf')(
            next_path_sample.x_t,
            next_times,
            observations,
            actions=actions,
            params=grad_params,
        )
        next_loss = jnp.square(next_vf_pred - next_path_sample.dx_t).mean()

        # sample next actions
        dist = self.network.select('actor')(next_observations)
        if self.config['const_std']:
            next_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            rng, next_action_rng = jax.random.split(rng)
            next_actions = jnp.clip(dist.sample(seed=next_action_rng), -1, 1)

        rng, future_goal_rng = jax.random.split(rng)
        if self.config['critic_noise_type'] == 'normal':
            future_goal_noises = jax.random.normal(
                future_goal_rng, shape=observations.shape, dtype=observations.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            future_goal_noises = jax.random.permutation(future_goal_rng, observations, axis=0)
        # if self.config['distill_type'] == 'fwd_int':
        #     future_goals = future_goal_noises[None, None, :] + self.network.select('target_critic')(
        #         future_goal_noises, next_observations, actions=next_actions)
        #     if self.config['q_agg'] == 'min':
        #         future_goals = future_goals.min(axis=0)
        #     else:
        #         future_goals = future_goals.mean(axis=0)
        # else:
        #     future_goals = self.compute_fwd_flow_samples(
        #         future_goal_noises, next_observations,
        #         actions=next_actions,
        #         use_target_network=self.config['use_target_critic']
        #     )
        future_flow_goals = self.compute_fwd_flow_goals(
            future_goal_noises, next_observations, next_actions,
            use_target_network=self.config['use_target_critic_vf']
        )
        future_flow_goals = jax.lax.stop_gradient(future_flow_goals)

        rng, future_time_rng, future_noise_rng = jax.random.split(rng, 3)
        future_times = jax.random.uniform(
            future_time_rng, shape=(batch_size,), dtype=future_flow_goals.dtype)
        if self.config['critic_noise_type'] == 'normal':
            future_noises = jax.random.normal(
                future_noise_rng, shape=future_flow_goals.shape, dtype=future_flow_goals.dtype)
        elif self.config['critic_noise_type'] == 'marginal_state':
            future_noises = jax.random.permutation(future_noise_rng, observations, axis=0)
        future_path_sample = self.cond_prob_path(
            x_0=future_noises, x_1=future_flow_goals, t=future_times)
        future_vf_pred = self.network.select('critic_vf')(
            future_path_sample.x_t,
            future_times,
            observations,
            actions=actions,
            params=grad_params,
        )
        future_loss = jnp.square(future_vf_pred - future_path_sample.dx_t).mean()

        flow_matching_loss = ((1 - self.config['discount']) * next_loss +
                              self.config['discount'] * future_loss)

        # distillation loss
        # if self.config['distill_type'] == 'fwd_int':
        #     rng, g_rng = jax.random.split(rng)
        #     shortcut_noises = jax.random.normal(
        #         g_rng, shape=observations.shape, dtype=observations.dtype)
        #     # TODO (chongyi): use bilinear vector field here.
        #     sampled_goals = self.compute_fwd_flow_samples(
        #         shortcut_noises, observations, actions=actions)
        #
        #     shortcut_goal_preds = shortcut_noises[None, None, :] + self.network.select('critic')(
        #         shortcut_noises, observations, actions=actions, params=grad_params)
        #     # shortcut_goal_preds = jax.vmap(jax.vmap(jnp.diag, -1, -1), 0, 0)(shortcut_goal_preds)
        #     distill_loss = jnp.square(shortcut_goal_preds - sampled_goals[None]).mean()
        # else:
        #     distill_loss = 0.0
        # critic_loss = cfm_loss + distill_loss

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
            'critic_next_flow_matching_loss': next_loss,
            'critic_future_flow_matching_loss': future_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        observations = batch['observations']
        actions = batch['actions']

        dist = self.network.select('actor')(observations, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            rng, q_action_rng = jax.random.split(rng)
            q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

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

    def compute_fwd_flow_goals(self, noises, observations, actions,
                               use_target_network=False):
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

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
            vf = self.network.select(module_name)(
                noisy_goals, times, observations, actions)
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
        seed, actor_seed = jax.random.split(seed)
        actions = dist.sample(seed=actor_seed)
        if not self.config['discrete']:
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
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0], ))

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_vf'] = encoder_module()
            encoders['critic'] = encoder_module()
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
            # critic_def = GCValue(
            #     hidden_dims=config['value_hidden_dims'],
            #     layer_norm=config['layer_norm'],
            #     num_ensembles=2,
            #     gc_encoder=encoders.get('critic'),
            # )
            critic_vf_def = GCFMVectorField(
                vector_dim=ex_observations.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                state_encoder=encoders.get('critic_vf'),
            )
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
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=not config['const_std'],
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        reward_def = GCValue(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('reward'),
        )

        network_info = dict(
            critic_vf=(critic_vf_def, (
                ex_observations, ex_times, ex_observations, ex_actions)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_observations, ex_times, ex_observations, ex_actions)),
            critic=(critic_def, (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, )),
        )
        # if config['distill_type'] == 'fwd_int':
        #     target_critic = (
        #         copy.deepcopy(critic_def),
        #         (ex_goals, ex_observations, ex_actions)
        #     )
        #     network_info['target_critic'] = target_critic
        # else:
        #     target_critic_vf = (
        #         copy.deepcopy(critic_vf_def),
        #         (ex_goals, ex_times, ex_observations, ex_actions)
        #     )
        #     network_info['target_critic_vf'] = target_critic_vf
        if config['reward_type'] == 'state':
            network_info.update(
                reward=(reward_def, (ex_observations, )),
            )
        else:
            network_info.update(
                reward=(reward_def, (ex_observations, None, ex_actions)),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='td_fmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            q_agg='mean',  # Aggregation method for target Q values.
            critic_loss_type='mse',  # Critic loss type. ('mse', 'expectile').
            critic_noise_type='normal',  # Critic noise type. ('marginal_state', 'normal').
            use_target_critic_vf=True,  # Whether to use the target critic velocity field to do bootstrap
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            reward_type='state',  # Reward type. ('state', 'state_action')
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
