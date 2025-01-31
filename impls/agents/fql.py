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
    """Flow Q-Learning (FQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    FQL with DDPG+BC only fits a Q function, while FQL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    # def critic_loss(self, batch, grad_params, rng, module_name='critic'):
    #     """Compute the contrastive value loss for the Q or V function."""
    #     batch_size = batch['observations'].shape[0]
    #
    #     rng, guidance_rng, time_rng, path_rng, likelihood_rng = jax.random.split(rng, 5)
    #     use_guidance = (jax.random.uniform(guidance_rng) >= self.config['uncond_prob'])
    #
    #     observations = jax.lax.select(
    #         use_guidance,
    #         batch['observations'],
    #         jnp.zeros_like(batch['observations'])
    #     )
    #     if module_name == 'critic':
    #         actions = jax.lax.select(
    #             use_guidance,
    #             batch['actions'],
    #             jnp.zeros_like(batch['actions'])
    #         )
    #     else:
    #         actions = None
    #     goals = batch['value_goals']
    #
    #     times = jax.random.uniform(time_rng, shape=(batch_size,))
    #     noises = jax.random.normal(path_rng, shape=goals.shape)
    #     path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
    #     vf_pred = self.network.select(module_name + '_vf')(
    #         path_sample.x_t,
    #         times,
    #         observations,
    #         actions=actions,
    #         params=grad_params,
    #     )
    #     cfm_loss = jnp.pow(vf_pred - path_sample.dx_t[None], 2).mean()
    #
    #     # q = self.compute_log_likelihood(
    #     #     goals, observations, likelihood_rng, actions=actions)
    #
    #     if self.config['distill_type'] == 'fwd_int':
    #         rng, g_rng = jax.random.split(rng)
    #         shortcut_noises = jax.random.normal(g_rng, shape=goals.shape, dtype=goals.dtype)
    #         sampled_goals = self.compute_fwd_flow_samples(
    #             shortcut_noises, observations, actions=actions)
    #         shortcut_preds = self.network.select(module_name)(
    #             shortcut_noises, observations, actions=actions, params=grad_params)
    #         distill_loss = jnp.pow(
    #             shortcut_preds - (sampled_goals - shortcut_noises)[None], 2).mean()
    #     else:
    #         distill_loss = 0.0
    #     critic_loss = cfm_loss + distill_loss
    #
    #     return critic_loss, {
    #         'critic_loss': critic_loss,
    #         'cond_flow_matching_loss': cfm_loss,
    #         'distillation_loss': distill_loss,
    #         # 'v_mean': q.mean(),
    #         # 'v_max': q.max(),
    #         # 'v_min': q.min(),
    #     }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        next_actions = self.network.select('actor')(noises, batch['next_observations'])
        next_actions = next_actions.squeeze(0)
        next_actions = jnp.clip(next_actions, -1, 1)

        # next_dist = self.network.select('actor')(batch['next_observations'])
        # next_actions = next_dist.sample(seed=rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        if self.config['min_q']:
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
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
        noises = jax.random.normal(noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        path_sample = self.cond_prob_path(x_0=noises, x_1=actions, t=times)
        vf_pred = self.network.select('actor_vf')(
            path_sample.x_t,
            times,
            observations,
            params=grad_params,
        )
        flow_matching_loss = jnp.pow(vf_pred - path_sample.dx_t[None], 2).mean()

        return flow_matching_loss, {
            'flow_matching_loss': flow_matching_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss (DDPG+BC, PG+BC, AWR, or SfBC)."""

        if self.config['actor_loss'] == 'pgbc':
            raise NotImplementedError
            # PG+BC loss.
            assert not self.config['discrete']

            batch_size = batch['observations'].shape[0]
            observations = batch['observations']
            rewards = batch['rewards']

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                rng, q_action_rng = jax.random.split(rng)
                q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)
            q_action_log_prob = dist.log_prob(q_actions)

            num_candidates = self.config['num_candidates']
            observations = jnp.repeat(
                jnp.expand_dims(observations, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *observations.shape[1:]])
            q_actions = jnp.repeat(
                jnp.expand_dims(q_actions, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *q_actions.shape[1:]])

            if self.config['distill_type'] == 'fwd_int':
                rng, g_rng = jax.random.split(rng)
                noises = jax.random.normal(g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = noises + self.network.select('critic')(
                    noises, observations, actions=q_actions)
                sampled_goals = sampled_goals.min(axis=0)
            else:
                rng, g_rng = jax.random.split(rng)
                noises = jax.random.normal(g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = self.compute_fwd_flow_samples(
                    noises, observations, actions=q_actions)

            goal_rewards = self.network.select('reward')(sampled_goals)
            goal_rewards = goal_rewards.reshape([batch_size, num_candidates])
            q = (1 - self.config['discount']) * rewards + self.config['discount'] * goal_rewards.mean(axis=-1)

            exp_q = jnp.exp(q)
            exp_q = jnp.minimum(exp_q, 100.0)
            q_loss = -(jax.lax.stop_gradient(exp_q) * q_action_log_prob).mean()

            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'q_action_log_prob': q_action_log_prob.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'awr':
            raise NotImplementedError

            # AWR loss.
            batch_size = batch['observations'].shape[0]
            observations = batch['observations']
            actions = batch['actions']

            num_candidates = self.config['num_candidates']
            observations = jnp.repeat(
                jnp.expand_dims(observations, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *observations.shape[1:]])
            actions = jnp.repeat(
                jnp.expand_dims(actions, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *actions.shape[1:]])
            if self.config['distill_type'] == 'fwd_int':
                rng, v_g_rng, q_g_rng = jax.random.split(rng, 3)
                v_noises = jax.random.normal(v_g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_v_goals = v_noises + self.network.select('value')(
                    v_noises, observations)
                q_noises = jax.random.normal(q_g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_q_goals = q_noises + self.network.select('critic')(
                    q_noises, observations, actions=actions)
                sampled_v_goals = sampled_v_goals.min(axis=0)
                sampled_q_goals = sampled_q_goals.min(axis=0)
            else:
                rng, v_g_rng, q_g_rng = jax.random.split(rng, 3)
                v_noises = jax.random.normal(v_g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_v_goals = self.compute_fwd_flow_samples(
                    v_noises, observations)
                q_noises = jax.random.normal(q_g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_q_goals = self.compute_fwd_flow_samples(
                    q_noises, observations, actions=actions)

            v_goal_rewards = self.network.select('reward')(sampled_v_goals)
            v_goal_rewards = v_goal_rewards.reshape([batch_size, num_candidates])
            q_goal_rewards = self.network.select('reward')(sampled_q_goals)
            q_goal_rewards = q_goal_rewards.reshape([batch_size, num_candidates])
            v = v_goal_rewards.mean(axis=-1)
            q = q_goal_rewards.mean(axis=-1)

            adv = q - v  # log p(g | s, a) - log p(g | s)

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            # batch_size = batch['observations'].shape[0]
            # observations = batch['observations']
            # actions = batch['actions']

            # dist = self.network.select('actor')(batch['observations'], params=grad_params)
            # if self.config['const_std']:
            #     q_actions = jnp.clip(dist.mode(), -1, 1)
            # else:
            #     rng, q_action_rng = jax.random.split(rng)
            #     q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(
                noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
            q_actions = self.network.select('actor')(noises, batch['observations'], params=grad_params)
            q_actions = q_actions.squeeze(0)
            q_actions = jnp.clip(q_actions, -1, 1)

            qs = self.network.select('critic')(batch['observations'], q_actions)
            if self.config['min_q']:
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()

            flow_actions = self.compute_fwd_flow_samples(
                noises, batch['observations'])
            bc_loss = self.config['alpha'] * jnp.pow(q_actions - flow_actions, 2).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
            }
        elif self.config['actor_loss'] == 'sfbc':
            raise NotImplementedError

            # BC loss.
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -log_prob.mean()

            actor_loss = bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_loss': bc_loss,
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

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

            # Time for this iteration
            times = jnp.full(noisy_actions.shape[:-1], i * step_size)

            vf = self.network.select('actor_vf')(noisy_actions, times, observations)

            # Update goals and divergence integral. We need to consider Q ensemble here.
            if self.config['min_q']:
                new_noisy_actions = jnp.min(noisy_actions[None] + vf * step_size, axis=0)
            else:
                new_noisy_actions = jnp.mean(noisy_actions[None] + vf * step_size, axis=0)

            # Return updated carry and scan output
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

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        if self.config['actor_loss'] == 'awr':
            rng, value_rng = jax.random.split(rng)
            value_loss, value_info = self.critic_loss(batch, grad_params, value_rng)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + flow_matching_loss + actor_loss
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
        if self.config['actor_loss'] == 'sfbc':
            num_candidates = self.config['num_candidates']
            observations = jnp.repeat(
                jnp.expand_dims(observations, axis=0),
                num_candidates, axis=0
            )
            rewards = self.network.select('reward')(observations)

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        action_dim = self.network.model_def.modules['actor'].output_dim

        rng, noise_rng = jax.random.split(seed)
        noises = jax.random.normal(
            noise_rng, shape=(observations.shape[0], action_dim), dtype=observations.dtype)
        actions = self.network.select('actor')(noises, observations)
        actions = actions.squeeze()
        actions = jnp.clip(actions, -1, 1)

        if self.config['actor_loss'] == 'sfbc':
            # SfBC: selecting from behavioral candidates
            observations = jnp.repeat(
                jnp.expand_dims(observations, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *observations.shape[1:]])
            actions = jnp.repeat(
                jnp.expand_dims(actions, axis=1),
                num_candidates, axis=1
            ).reshape([-1, *actions.shape[1:]])

            if self.config['distill_type'] == 'fwd_int':
                seed, g_seed = jax.random.split(seed)
                noises = jax.random.normal(g_seed, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = noises + self.network.select('critic')(
                    noises, observations, actions=actions)
                sampled_goals = sampled_goals.min(axis=0)
            else:
                seed, g_seed = jax.random.split(seed)
                noises = jax.random.normal(g_seed, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = self.compute_fwd_flow_samples(
                    noises, observations, actions=actions)
            goal_rewards = self.network.select('reward')(sampled_goals)
            goal_rewards = goal_rewards.reshape([num_candidates, num_candidates])
            q = (1 - self.config['discount']) * rewards + self.config['discount'] * goal_rewards.mean(axis=-1)
            argmax_idxs = jnp.argmax(q, axis=-1)
            actions = actions[argmax_idxs]

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

        rng, time_rng, noise_rng = jax.random.split(rng, 3)
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],))
        ex_noises = jax.random.normal(noise_rng, shape=ex_actions.shape, dtype=ex_actions.dtype)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            if config['actor_loss'] == 'awr':
                encoders['value'] = encoder_module()

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
            # critic_vf_def = GCFMVectorField(
            #     vector_dim=ex_goals.shape[-1],
            #     hidden_dims=config['value_hidden_dims'],
            #     layer_norm=config['layer_norm'],
            #     num_ensembles=config['num_ensembles_q'],
            #     state_encoder=encoders.get('value'),
            # )
            # if config['distill_type'] == 'fwd_int':
            #     output_dim = ex_goals.shape[-1]
            #     activate_final = False
            # else:
            #     output_dim = 1
            #     activate_final = True
            # critic_def = GCFMValue(
            #     hidden_dims=config['value_hidden_dims'],
            #     output_dim=output_dim,
            #     layer_norm=config['layer_norm'],
            #     activate_final=activate_final,
            #     num_ensembles=2,
            #     state_encoder=encoders.get('value'),
            # )
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

        if config['actor_loss'] == 'awr':
            raise NotImplementedError
            value_vf_def = GCFMVectorField(
                vector_dim=ex_goals.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
                state_encoder=encoders.get('value'),
            )
            if config['distill_type'] == 'fwd_int':
                output_dim = ex_goals.shape[-1]
                activate_final = False
            else:
                output_dim = 1
                activate_final = True
            value_def = GCFMValue(
                hidden_dims=config['value_hidden_dims'],
                output_dim=output_dim,
                layer_norm=config['layer_norm'],
                activate_final=activate_final,
                num_ensembles=1,
                state_encoder=encoders.get('value'),
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
                layer_norm=config['layer_norm'],
                state_encoder=encoders.get('actor'),
            )
            actor_def = GCFMValue(
                hidden_dims=config['actor_hidden_dims'],
                output_dim=action_dim,
                layer_norm=config['layer_norm'],
                state_encoder=encoders.get('actor'),
            )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations)),
            actor=(actor_def, (ex_noises, ex_observations)),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, )),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
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
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            min_q=False,  # Whether to take the minimum over Q ensembles
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            uncond_prob=0.0,  # Probability of training the marginal velocity field vs the guided velocity field.
            num_flow_steps=20,  # Number of steps for solving ODEs using the Euler method.
            actor_loss='ddpgbc',  # Actor loss type ('ddpgbc', 'pgbc', 'awr' or 'sfbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            num_candidates=32,
            # Number of behavioral candidates for SfBC or number of future goal candidates for DDPGBC.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric samling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
