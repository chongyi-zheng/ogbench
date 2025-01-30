from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCFMVectorField, GCFMValue, GCValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class FMRLAgent(flax.struct.PyTreeNode):
    """Flow Matching RL (FMRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    FMRL with DDPG+BC only fits a Q function, while FMRL with AWR fits both Q and V functions to compute advantages.
    """

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

        reward_loss = jnp.pow(reward_preds - rewards, 2).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        rng, guidance_rng, time_rng, path_rng, likelihood_rng = jax.random.split(rng, 5)
        use_guidance = (jax.random.uniform(guidance_rng) >= self.config['uncond_prob'])
        
        observations = jax.lax.select(
            use_guidance,
            batch['observations'], 
            jnp.zeros_like(batch['observations'])
        )
        if module_name == 'critic':
            actions = jax.lax.select(
                use_guidance, 
                batch['actions'], 
                jnp.zeros_like(batch['actions'])
            )
        else:
            actions = None
        goals = batch['value_goals']
        
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        noises = jax.random.normal(path_rng, shape=goals.shape)
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select(module_name + '_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        cfm_loss = jnp.pow(vf_pred - path_sample.dx_t[None], 2).mean()

        # q = self.compute_log_likelihood(
        #     goals, observations, likelihood_rng, actions=actions)

        if self.config['distill_type'] == 'fwd_int':
            rng, g_rng = jax.random.split(rng)
            shortcut_noises = jax.random.normal(g_rng, shape=goals.shape, dtype=goals.dtype)
            sampled_goals = self.compute_fwd_flow_samples(
                shortcut_noises, observations, actions=actions)
            shortcut_preds = self.network.select(module_name)(
                shortcut_noises, observations, actions=actions, params=grad_params)
            distill_loss = jnp.pow(
                shortcut_preds - (sampled_goals - shortcut_noises)[None], 2).mean()
        else:
            distill_loss = 0.0
        critic_loss = cfm_loss + distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'cond_flow_matching_loss': cfm_loss,
            'distillation_loss': distill_loss,
            # 'v_mean': q.mean(),
            # 'v_max': q.max(),
            # 'v_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss (DDPG+BC, PG+BC, AWR, or SfBC)."""

        if self.config['actor_loss'] == 'pgbc':
            # PG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                rng, q_action_rng = jax.random.split(rng)
                q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)
            q_action_log_prob = dist.log_prob(q_actions)

            if self.config['distill_type'] == 'fwd_int':
                rng, q_rng = jax.random.split(rng)
                q = self.compute_shortcut_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)
                q = q.min(axis=0)
            else:
                rng, q_rng = jax.random.split(rng)
                q = self.compute_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)

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
            # AWR loss.
            batch_size = batch['observations'].shape[0]
            observations = batch['observations']
            actions = batch['actions']
            # goals = batch['actor_goals']

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

                # rng, v_rng, q_rng = jax.random.split(rng, 3)
                # v = self.compute_log_likelihood(
                #     batch['actor_goals'], batch['observations'], v_rng)
                # q = self.compute_log_likelihood(
                #     batch['actor_goals'], batch['observations'], q_rng, actions=batch['actions'])

            v_goal_rewards = self.network.select('reward')(sampled_v_goals)
            v_goal_rewards = v_goal_rewards.reshape([batch_size, num_candidates])
            q_goal_rewards = self.network.select('reward')(sampled_q_goals)
            q_goal_rewards = q_goal_rewards.reshape([batch_size, num_candidates])
            q = q_goal_rewards.mean(axis=-1)
            v = v_goal_rewards.mean(axis=-1)

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

            batch_size = batch['observations'].shape[0]
            observations = batch['observations']
            rewards = batch['rewards']

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                rng, q_action_rng = jax.random.split(rng)
                q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

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
                # rng, q_rng = jax.random.split(rng)
                # q = self.compute_shortcut_log_likelihood(
                #     batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)
                # q = q.min(axis=0)
                rng, g_rng = jax.random.split(rng)
                noises = jax.random.normal(g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = noises + self.network.select('critic')(
                    noises, observations, actions=q_actions)
                sampled_goals = sampled_goals.min(axis=0)
            else:
                # rng, q_rng = jax.random.split(rng)
                # q = self.compute_rev_flow_samples(
                #     batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)
                rng, g_rng = jax.random.split(rng)
                noises = jax.random.normal(g_rng, shape=observations.shape, dtype=observations.dtype)
                sampled_goals = self.compute_fwd_flow_samples(noises, observations, actions=q_actions)

            goal_rewards = self.network.select('reward')(sampled_goals)
            goal_rewards = goal_rewards.reshape([batch_size, num_candidates])
            q = (1 - self.config['discount']) * rewards + self.config['discount'] * goal_rewards.mean(axis=-1)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            # q_loss = -q.mean()
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
        elif self.config['actor_loss'] == 'sfbc':
            # BC loss.
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
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

    def compute_fwd_flow_samples(self, noises, observations, actions=None):
        if actions is not None:
            module_name = 'critic_vf'
        else:
            module_name = 'value_vf'

        noisy_goals = noises
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_goals, ) = carry

            # Time for this iteration
            times = jnp.full(noisy_goals.shape[:-1], i * step_size)

            vf = self.network.select(module_name)(
                noisy_goals, times, observations, actions)

            # Update goals and divergence integral. We need to consider Q ensemble here.
            new_noisy_goals = jnp.min(noisy_goals[None] + vf * step_size, axis=0)

            # Return updated carry and scan output
            return (new_noisy_goals, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, ), _ = jax.lax.scan(
            body_fn, (noisy_goals, ), jnp.arange(num_flow_steps))

        return noisy_goals

    def compute_rev_flow_samples(self, goals, observations, actions=None):
        if actions is not None:
            module_name = 'critic_vf'
        else:
            module_name = 'value_vf'

        noises = goals
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noises, ) = carry

            # Time for this iteration
            times = 1.0 - jnp.full(goals.shape[:-1], i * step_size)

            vf = self.network.select(module_name)(
                noises, times, observations, actions)

            # Update goals and divergence integral. We need to consider Q ensemble here.
            new_noises = jnp.min(noises[None] - vf * step_size, axis=0)

            # Return updated carry and scan output
            return (new_noises, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noises, ), _ = jax.lax.scan(
            body_fn, (noises, ), jnp.arange(num_flow_steps))

        return noises

    # def compute_log_likelihood(self, goals, observations, rng, actions=None):
    #     if actions is not None:
    #         module_name = 'critic_vf'
    #         num_ensembles = self.config['num_ensembles_q']
    #     else:
    #         module_name = 'value_vf'
    #         num_ensembles = 1
    #
    #     noisy_goals = goals
    #     div_int = jnp.zeros(goals.shape[:-1])
    #     num_flow_steps = self.config['num_flow_steps']
    #     step_size = 1.0 / num_flow_steps
    #
    #     # Define the body function to be scanned
    #     def body_fn(carry, i):
    #         """
    #         carry: (noisy_goals, div_int, rng)
    #         i: current step index
    #         """
    #         noisy_goals, div_int, rng = carry
    #
    #         # Time for this iteration
    #         times = 1.0 - jnp.full(noisy_goals.shape[:-1], i * step_size)
    #
    #         if self.config['exact_divergence']:
    #             def compute_exact_div(noisy_goals, times, observations, actions):
    #                 def vf_func(noisy_goal, time, observation, action):
    #                     noisy_goal = jnp.expand_dims(noisy_goal, 0)
    #                     time = jnp.expand_dims(time, 0)
    #                     observation = jnp.expand_dims(observation, 0)
    #                     if action is not None:
    #                         action = jnp.expand_dims(action, 0)
    #                     vf = self.network.select(module_name)(
    #                         noisy_goal, time, observation, action).squeeze(1)
    #
    #                     return vf.reshape(-1)
    #
    #                 def div_func(noisy_goal, time, observation, action):
    #                     jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
    #                     jac = jac.reshape([num_ensembles, noisy_goal.shape[-1], noisy_goal.shape[-1]])
    #
    #                     return jnp.trace(jac, axis1=-2, axis2=-1)
    #
    #                 vf = self.network.select(module_name)(
    #                     noisy_goals, times, observations, actions)
    #
    #                 if actions is not None:
    #                     div = jax.vmap(div_func, in_axes=(0, 0, 0, 0), out_axes=1)(
    #                         noisy_goals, times, observations, actions)
    #                 else:
    #                     div = jax.vmap(div_func, in_axes=(0, 0, 0, None), out_axes=1)(
    #                         noisy_goals, times, observations, actions)
    #
    #                 return vf, div
    #
    #             vf, div = compute_exact_div(noisy_goals, times, observations, actions)
    #         else:
    #             def compute_hutchinson_div(noisy_goals, times, observations, actions, rng):
    #                 # Define vf_func for jvp
    #                 def vf_func(goals):
    #                     vf = self.network.select(module_name)(
    #                         goals,
    #                         times,
    #                         observations,
    #                         actions=actions,
    #                     )
    #
    #                     return vf.reshape([-1, *vf.shape[2:]])
    #
    #                 # Split RNG and sample noise
    #                 z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #
    #                 # Forward (vf) and linearization (jac_vf_dot_z)
    #                 vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z, ))
    #                 vf = vf.reshape([num_ensembles, -1, *vf.shape[1:]])
    #                 jac_vf_dot_z = jac_vf_dot_z.reshape([num_ensembles, -1, *jac_vf_dot_z.shape[1:]])
    #
    #                 # Hutchinson's trace estimator
    #                 # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
    #                 div = jnp.einsum("eij,ij->ei", jac_vf_dot_z, z)
    #
    #                 return vf, div
    #
    #             rng, div_rng = jax.random.split(rng)
    #             vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, div_rng)
    #
    #         # Update goals and divergence integral. We need to consider Q ensemble here.
    #         new_noisy_goals = jnp.min(noisy_goals[None] - vf * step_size, axis=0)
    #         new_div_int = jnp.min(div_int[None] - div * step_size, axis=0)
    #
    #         # Return updated carry and scan output
    #         return (new_noisy_goals, new_div_int, rng), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     (noisy_goals, div_int, rng), _ = jax.lax.scan(
    #         body_fn, (noisy_goals, div_int, rng), jnp.arange(num_flow_steps))
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     return log_prob

    # def compute_shortcut_log_likelihood(self, goals, observations, key, actions=None):
    #     if actions is not None:
    #         module_name = 'critic'
    #     else:
    #         module_name = 'value'
    #
    #     if self.config['exact_divergence']:
    #         noise_preds = goals[None] + self.network.select(module_name)(
    #             goals, observations, actions)
    #
    #         def shortcut_func(g, s, a):
    #             shortcut_pred = self.network.select(module_name)(
    #                 g, s, a)
    #
    #             return shortcut_pred
    #
    #         jac = jax.vmap(jax.jacrev(shortcut_func), in_axes=(0, 0, 0), out_axes=1)(goals, observations, actions)
    #         div_int = jnp.trace(jac, axis1=-2, axis2=-1)
    #     else:
    #         def shortcut_func(g):
    #             shortcut_preds = self.network.select(module_name)(
    #                 g, observations, actions)
    #
    #             return shortcut_preds
    #
    #         key, z_key = jax.random.split(key)
    #         z = jax.random.normal(z_key, shape=goals.shape, dtype=goals.dtype)
    #         # z = jax.random.rademacher(z_key, shape=goals.shape, dtype=goals.dtype)
    #         shortcut_preds, jac_sc_dot_z = jax.jvp(shortcut_func, (goals,), (z,))
    #         noise_preds = goals[None] + shortcut_preds
    #         div_int = jnp.einsum("eij,ij->ei", jac_sc_dot_z, z)
    #
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noise_preds ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     return log_prob

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['actor_loss'] == 'awr':
            rng, value_rng = jax.random.split(rng)
            value_loss, value_info = self.critic_loss(batch, grad_params, value_rng, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = reward_loss + critic_loss + value_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

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
            observations = jnp.repeat(observations[None], num_candidates, axis=0)

        dist = self.network.select('actor')(observations, temperature=temperature)
        seed, actor_seed = jax.random.split(seed)
        actions = dist.sample(seed=actor_seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        if self.config['actor_loss'] == 'sfbc':
            # TODO (chongyiz): fixme
            raise NotImplementedError

            # SfBC: selecting from behavioral candidates
            if self.config['distill_type'] == 'log_prob':
                q = self.network.select('critic')(goals, observations, actions=actions)
                q = q.min(axis=0)
            elif self.config['distill_type'] == 'fwd_int':
                seed, q_seed = jax.random.split(seed)
                q = self.compute_shortcut_log_likelihood(
                    goals, observations, q_seed, actions=actions)
                q = q.min(axis=0)
            else:
                seed, q_seed = jax.random.split(seed)
                q = self.compute_log_likelihood(goals, observations, q_seed, actions=actions)
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
            encoders['critic'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value'] = encoder_module()
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
                vector_dim=ex_goals.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_ensembles_q'],
                state_encoder=encoders.get('value'),
            )
            if config['distill_type'] == 'fwd_int':
                output_dim = ex_goals.shape[-1]
                activate_final = False
            else:
                output_dim = 1
                activate_final = True
            critic_def = GCFMValue(
                hidden_dims=config['value_hidden_dims'],
                output_dim=output_dim,
                layer_norm=config['layer_norm'],
                activate_final=activate_final,
                num_ensembles=2,
                state_encoder=encoders.get('value'),
            )

        if config['actor_loss'] == 'awr':
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
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        reward_def = GCValue(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        network_info = dict(
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, )),
            reward=(reward_def, (ex_observations, )),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value_vf=(value_vf_def, (ex_goals, ex_times, ex_observations)),
                value=(value_def, (ex_goals, ex_observations)),
            )
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
            agent_name='fmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512),  # Reward network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            num_ensembles_q=2,  # Number of ensemble for the critic.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            uncond_prob=0.0,  # Probability of training the marginal velocity field vs the guided velocity field.
            num_flow_steps=20,  # Number of steps for solving ODEs using the Euler method.
            distill_type='none',  # Distillation type ('none', 'log_prob', 'fwd_int').
            actor_loss='sfbc',  # Actor loss type ('ddpgbc', 'pgbc', 'awr' or 'sfbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            num_candidates=32,  # Number of behavioral candidates for SfBC or number of future goal candidates for DDPGBC.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
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
