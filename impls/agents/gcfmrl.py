from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from diffrax import (
    diffeqsolve, ODETerm,
    Euler, Dopri5, Tsit5,
    RecursiveCheckpointAdjoint, DirectAdjoint, ImplicitAdjoint
)

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCFMVectorField, GCFMValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class GCFMRLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Flow Matching RL (GCFMRL) agent.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    ode_solver: Any
    ode_adjoint: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the contrastive value loss for the Q or V function."""
        rng, time_rng, noise_rng, q_rng = jax.random.split(rng, 4)

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        if self.config['noise_type'] == 'normal':
            noises = jax.random.normal(noise_rng, shape=goals.shape, dtype=goals.dtype)
        else:
            noises = jnp.roll(batch['value_goals'], shift=1, axis=0)
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select('critic_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        fm_loss = jnp.square(vf_pred - path_sample.dx_t).mean()

        if self.config['noise_type'] == 'normal':
            flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
                goals, observations, q_rng, actions=actions, info=True)
        else:
            raise NotImplementedError
            q = self.compute_log_likelihood(
                goals, observations, q_rng, actions=actions, return_ratio=True)

        if self.config['distill_type'] == 'log_prob':
            log_prob_pred = self.network.select('critic')(
                goals, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(flow_log_prob - log_prob_pred).mean()
        elif self.config['distill_type'] == 'noise_div_int':
            shortcut_noise_pred = self.network.select('critic_noise')(
                goals, observations, actions=actions, params=grad_params)
            noise_distill_loss = jnp.square(shortcut_noise_pred - flow_noise).mean()
            shortcut_div_int_pred = self.network.select('critic_div')(
                goals, observations, actions=actions, params=grad_params)
            div_int_distill_loss = jnp.square(shortcut_div_int_pred - flow_div_int).mean()
            distill_loss = noise_distill_loss + div_int_distill_loss
        elif self.config['distill_type'] == 'div_int':
            shortcut_div_int_pred = self.network.select('critic')(
                goals, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(shortcut_div_int_pred - flow_div_int).mean()
        elif self.config['distill_type'] == 'rev_int':
            raise NotImplementedError
            flow_noises = self.compute_rev_flow_samples(
                goals, observations, actions=actions)
            shortcut_noise_preds = goals + self.network.select('critic')(
                goals, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(shortcut_noise_preds - flow_noises).mean()
        else:
            distill_loss = 0.0
        critic_loss = fm_loss + distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'flow_matching_loss': fm_loss,
            'distillation_loss': distill_loss,
            'flow_log_prob_mean': flow_log_prob.mean(),
            'flow_log_prob_max': flow_log_prob.max(),
            'flow_log_prob_min': flow_log_prob.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss."""

        # DDPG+BC loss.
        assert not self.config['discrete']

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            rng, q_action_rng = jax.random.split(rng)
            q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

        if self.config['distill_type'] == 'log_prob':
            q = self.network.select('critic')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
        elif self.config['distill_type'] == 'noise_div_int':
            shortcut_noise_pred = self.network.select('critic_noise')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
            shortcut_div_int_pred = self.network.select('critic_div')(
                batch['actor_goals'], batch['observations'], actions=q_actions)

            gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + shortcut_noise_pred ** 2, axis=-1)
            q = gaussian_log_prob + shortcut_div_int_pred  # log p_1(g | s, a)
        elif self.config['distill_type'] == 'div_int':
            q = self.network.select('critic')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
        elif self.config['distill_type'] == 'rev_int':
            raise NotImplementedError
            rng, q_rng = jax.random.split(rng)
            if self.config['noise_type'] == 'normal':
                q = self.compute_shortcut_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions
                )
            else:
                q = self.compute_shortcut_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions,
                    return_ratio=True
                )
        else:
            rng, q_rng = jax.random.split(rng)
            if self.config['noise_type'] == 'normal':
                q = self.compute_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions
                )
            else:
                raise NotImplementedError
                q = self.compute_log_likelihood(
                    batch['actor_goals'], batch['observations'], q_rng, actions=q_actions,
                    return_ratio=True
                )

            # q = self.compute_log_likelihood(
            #     batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
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

    # def compute_rev_flow_samples(self, goals, observations, actions=None):
    #     noises = goals
    #     num_flow_steps = self.config['num_flow_steps']
    #     step_size = 1.0 / num_flow_steps
    #
    #     def body_fn(carry, i):
    #         """
    #         carry: (noises, )
    #         i: current step index
    #         """
    #         (noises, ) = carry
    #
    #         # Time for this iteration
    #         times = 1.0 - jnp.full(goals.shape[:-1], i * step_size)
    #
    #         vf = self.network.select('critic_vf')(
    #             noises, times, observations, actions)
    #
    #         # Update goals and divergence integral. We need to consider Q ensemble here.
    #         new_noises = noises - vf * step_size
    #
    #         # Return updated carry and scan output
    #         return (new_noises, ), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     (noises, ), _ = jax.lax.scan(
    #         body_fn, (noises, ), jnp.arange(num_flow_steps))
    #
    #     return noises

    def compute_rev_flow_samples(self, goals, observations, actions=None):
        def vector_field(time, noises, carry):
            (observations, actions) = carry
            times = jnp.full(noises.shape[:-1], time)

            vf = self.network.select('critic_vf')(
                noises, times, observations, actions=actions)

            return vf

        ode_term = ODETerm(vector_field)
        ode_sol = diffeqsolve(
            ode_term, self.ode_solver,
            t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
            y0=goals, args=(observations, actions),
            adjoint=self.ode_adjoint,
        )
        noises = ode_sol.ys[-1]

        return noises

    # def compute_log_likelihood(self, goals, observations, rng, actions=None,
    #                            return_ratio=False):
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
    #         if self.config['div_type'] == 'exact':
    #             def compute_exact_div(noisy_goals, times, observations, actions):
    #                 def vf_func(noisy_goal, time, observation, action):
    #                     noisy_goal = jnp.expand_dims(noisy_goal, 0)
    #                     time = jnp.expand_dims(time, 0)
    #                     observation = jnp.expand_dims(observation, 0)
    #                     if action is not None:
    #                         action = jnp.expand_dims(action, 0)
    #                     vf = self.network.select('critic_vf')(
    #                         noisy_goal, time, observation, action).squeeze(0)
    #
    #                     return vf
    #
    #                 def div_func(noisy_goal, time, observation, action):
    #                     jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
    #                     # jac = jac.reshape([noisy_goal.shape[-1], noisy_goal.shape[-1]])
    #
    #                     return jnp.trace(jac, axis1=-2, axis2=-1)
    #
    #                 vf = self.network.select('critic_vf')(
    #                     noisy_goals, times, observations, actions)
    #
    #                 if actions is not None:
    #                     div = jax.vmap(div_func, in_axes=(0, 0, 0, 0), out_axes=0)(
    #                         noisy_goals, times, observations, actions)
    #                 else:
    #                     div = jax.vmap(div_func, in_axes=(0, 0, 0, None), out_axes=0)(
    #                         noisy_goals, times, observations, actions)
    #
    #                 return vf, div
    #
    #             vf, div = compute_exact_div(noisy_goals, times, observations, actions)
    #         else:
    #             def compute_hutchinson_div(noisy_goals, times, observations, actions, rng):
    #                 # Define vf_func for jvp
    #                 def vf_func(goals):
    #                     vf = self.network.select('critic_vf')(
    #                         goals,
    #                         times,
    #                         observations,
    #                         actions=actions,
    #                     )
    #
    #                     # return vf.reshape([-1, *vf.shape[2:]])
    #                     return vf
    #
    #                 # Split RNG and sample noise
    #                 if self.config['div_type'] == 'hutchinson_normal':
    #                     z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #                 elif self.config['div_type'] == 'hutchinson_rademacher':
    #                     z = jax.random.rademacher(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #
    #                 # Forward (vf) and linearization (jac_vf_dot_z)
    #                 vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z, ))
    #                 # vf = vf.reshape([-1, *vf.shape[1:]])
    #                 # jac_vf_dot_z = jac_vf_dot_z.reshape([-1, *jac_vf_dot_z.shape[1:]])
    #
    #                 # Hutchinson's trace estimator
    #                 # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
    #                 div = jnp.einsum("ij,ij->i", jac_vf_dot_z, z)
    #
    #                 return vf, div
    #
    #             rng, div_rng = jax.random.split(rng)
    #             vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, div_rng)
    #
    #         # Update goals and divergence integral. We need to consider Q ensemble here.
    #         # new_noisy_goals = jnp.min(noisy_goals[None] - vf * step_size, axis=0)
    #         # new_div_int = jnp.min(div_int[None] - div * step_size, axis=0)
    #         new_noisy_goals = noisy_goals - vf * step_size
    #         new_div_int = div_int - div * step_size
    #
    #         # Return updated carry and scan output
    #         return (new_noisy_goals, new_div_int, rng), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     (noisy_goals, div_int, rng), _ = jax.lax.scan(
    #         body_fn, (noisy_goals, div_int, rng), jnp.arange(num_flow_steps))
    #
    #     if return_ratio:
    #         log_ratio = div_int
    #         return log_ratio
    #     else:
    #         # Finally, compute log_prob using the final noisy_goals and div_int
    #         gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
    #         log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #         return log_prob

    def compute_log_likelihood(self, goals, observations, key, actions=None,
                               info=False):
        if self.config['div_type'] == 'exact':
            def vector_field(time, noise_div_int, carry):
                noises, _ = noise_div_int
                observations, actions, _ = carry

                # Forward (vf) and linearization (jac_vf_dot_z)
                times = jnp.full(noises.shape[:-1], time)

                def single_vf(t, n, obs, a):
                    noise = jnp.expand_dims(n, 0)
                    time = jnp.expand_dims(t, 0)
                    observation = jnp.expand_dims(obs, 0)
                    if a is not None:
                        action = jnp.expand_dims(a, 0)
                    else:
                        action = a

                    vf = self.network.select('critic_vf')(
                        noise, time, observation, action).squeeze(0)

                    return vf

                vf = self.network.select('critic_vf')(
                    noises, times, observations, actions)

                if actions is not None:
                    jac = jax.vmap(
                        jax.jacrev(single_vf, argnums=1),
                        in_axes=(0, 0, 0, 0), out_axes=0
                    )(times, noises, observations, actions)
                else:
                    jac = jax.vmap(
                        jax.jacrev(single_vf, argnums=1),
                        in_axes=(0, 0, 0, None), out_axes=0
                    )(times, noises, observations, actions)

                div = jnp.trace(jac, axis1=-2, axis2=-1)

                return vf, div
        else:
            def vector_field(time, noise_div_int, carry):
                noises, _ = noise_div_int
                observations, actions, z = carry

                # Split RNG and sample noise
                # key, z_key = jax.random.split(key)
                # if self.config['div_type'] == 'hutchinson_normal':
                #     z = jax.random.normal(z_key, shape=goals.shape, dtype=goals.dtype)
                # elif self.config['div_type'] == 'hutchinson_rademacher':
                #     z = jax.random.rademacher(z_key, shape=goals.shape, dtype=goals.dtype)

                # Forward (vf) and linearization (jac_vf_dot_z)
                times = jnp.full(noises.shape[:-1], time)
                # vf, jac_vf_dot_z = jax.jvp(
                #     lambda n: self.network.select('critic_vf')(n, times, observations, actions),
                #     (noises,), (z,)
                # )

                def single_jvp(z):
                    vf, jac_vf_dot_z = jax.jvp(
                        lambda n: self.network.select('critic_vf')(n, times, observations, actions),
                        (noises,), (z,)
                    )

                    return vf, jac_vf_dot_z

                vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
                div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)

                vf = vf[..., 0]  # vf are the same along the final dimension
                div = div.mean(axis=-1)

                return vf, div

        key, z_key = jax.random.split(key)
        if self.config['div_type'] == 'hutchinson_normal':
            z = jax.random.normal(
                z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        elif self.config['div_type'] == 'hutchinson_rademacher':
            z = jax.random.rademacher(
                z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        else:
            z = None

        ode_term = ODETerm(vector_field)
        ode_sol = diffeqsolve(
            ode_term, self.ode_solver,
            t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
            y0=(goals, jnp.zeros(goals.shape[:-1])),
            args=(observations, actions, z),
            adjoint=self.ode_adjoint,
        )
        noises, div_int = jax.tree.map(
            lambda x: x[-1], ode_sol.ys)

        # Finally, compute log_prob using the final noisy_goals and div_int
        gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
        log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

        if info:
            return log_prob, noises, div_int
        else:
            return log_prob

    # def compute_shortcut_log_likelihood(self, goals, observations, key, actions=None,
    #                                     return_ratio=False):
    #     if self.config['div_type'] == 'exact':
    #         noise_preds = goals + self.network.select('critic')(
    #             goals, observations, actions)
    #
    #         def shortcut_func(g, s, a):
    #             shortcut_pred = self.network.select('critic')(
    #                 g, s, a)
    #
    #             return shortcut_pred
    #
    #         jac = jax.vmap(jax.jacrev(shortcut_func), in_axes=(0, 0, 0), out_axes=0)(goals, observations, actions)
    #         div_int = jnp.trace(jac, axis1=-2, axis2=-1)
    #     else:
    #         def shortcut_func(g):
    #             shortcut_preds = self.network.select('critic')(
    #                 g, observations, actions)
    #
    #             return shortcut_preds
    #
    #         key, z_key = jax.random.split(key)
    #         # z = jax.random.normal(z_key, shape=goals.shape, dtype=goals.dtype)
    #         # # z = jax.random.rademacher(z_key, shape=goals.shape, dtype=goals.dtype)
    #         if self.config['div_type'] == 'hutchinson_normal':
    #             z = jax.random.normal(z_key, shape=goals.shape, dtype=goals.dtype)
    #         elif self.config['div_type'] == 'hutchinson_rademacher':
    #             z = jax.random.rademacher(z_key, shape=goals.shape, dtype=goals.dtype)
    #
    #         shortcut_preds, jac_sc_dot_z = jax.jvp(shortcut_func, (goals,), (z,))
    #         noise_preds = goals + shortcut_preds
    #         div_int = jnp.einsum("ij,ij->i", jac_sc_dot_z, z)
    #
    #     if return_ratio:
    #         log_ratio = div_int
    #         return log_ratio
    #     else:
    #         gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noise_preds ** 2, axis=-1)
    #         log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #         return log_prob

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""

        dist = self.network.select('actor')(observations, goals, temperature=temperature)
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
            encoders['critic_vf_state'] = encoder_module()
            encoders['critic_vf_goal'] = encoder_module()
            if config['distill_type'] == 'noise_div_int':
                encoders['critic_noise_state'] = encoder_module()
                encoders['critic_noise_goal'] = encoder_module()
                encoders['critic_div_state'] = encoder_module()
                encoders['critic_div_goal'] = encoder_module()
            else:
                encoders['critic_state'] = encoder_module()
                encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

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
                layer_norm=config['value_layer_norm'],
                state_encoder=encoders.get('critic_vf_state'),
                goal_encoder=encoders.get('critic_vf_goal'),
            )
            if config['distill_type'] == 'noise_div_int':
                critic_noise_def = GCFMValue(
                    hidden_dims=config['value_hidden_dims'],
                    output_dim=ex_goals.shape[-1],
                    layer_norm=config['value_layer_norm'],
                    state_encoder=encoders.get('critic_noise_state'),
                    goal_encoder=encoders.get('critic_noise_goal'),
                )
                critic_div_def = GCFMValue(
                    hidden_dims=config['value_hidden_dims'],
                    output_dim=1,
                    layer_norm=config['value_layer_norm'],
                    state_encoder=encoders.get('critic_div_state'),
                    goal_encoder=encoders.get('critic_div_goal'),
                )
            else:
                critic_def = GCFMValue(
                    hidden_dims=config['value_hidden_dims'],
                    output_dim=1,
                    layer_norm=config['value_layer_norm'],
                    state_encoder=encoders.get('critic_state'),
                    goal_encoder=encoders.get('critic_goal'),
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
                layer_norm=config['actor_layer_norm'],
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        if config['distill_type'] == 'noise_div_int':
            network_info['critic_noise'] = (critic_noise_def, (ex_goals, ex_observations, ex_actions))
            network_info['critic_div'] = (critic_div_def, (ex_goals, ex_observations, ex_actions))
        else:
            network_info['critic'] = (critic_def, (ex_goals, ex_observations, ex_actions))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        if config['ode_solver_type'] == 'euler':
            ode_solver = Euler()
        elif config['ode_solver_type'] == 'dopri5':
            ode_solver = Dopri5()
        elif config['ode_solver_type'] == 'tsit5':
            ode_solver = Tsit5()
        else:
            raise TypeError("Unknown ode_solver_type: {}".format(config['ode_solver_type']))

        if config['ode_adjoint_type'] == 'recursive_checkpoint':
            ode_adjoint = RecursiveCheckpointAdjoint()
        elif config['ode_adjoint_type'] == 'direct':
            ode_adjoint = DirectAdjoint()
        elif config['ode_adjoint_type'] == 'implicit':
            ode_adjoint = ImplicitAdjoint()
        else:
            raise TypeError("Unknown ode_adjoint_type: {}".format(config['ode_adjoint_type']))

        return cls(rng, network=network, cond_prob_path=cond_prob_path,
                   ode_solver=ode_solver, ode_adjoint=ode_adjoint,
                   config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcfmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5').
            ode_adjoint_type='recursive_checkpoint',  # Type of ODE adjoint ('recursive_checkpoint', 'direct', 'implicit').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            noise_type='normal',  # Noise distribution type ('normal', 'marginal')
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            div_type='exact',  # Divergence estimator type ('exact', 'hutchinson_normal', 'hutchinson_rademacher').
            distill_type='none',  # Distillation type ('none', 'log_prob', 'rev_int').
            num_hutchinson_ests=32,  # Number of random vectors for hutchinson divergence estimation.
            alpha=0.1,  # BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
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
        )
    )
    return config
