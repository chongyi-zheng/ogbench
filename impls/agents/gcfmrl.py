from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from diffrax import (
    diffeqsolve, ODETerm,
    Euler, Dopri5, Tsit5,
    RecursiveCheckpointAdjoint, DirectAdjoint, BacksolveAdjoint
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

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the contrastive value loss for the Q or V function."""
        rng, time_rng, noise_rng, q_rng = jax.random.split(rng, 4)

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        dataset_obs_mean = batch['dataset_obs_mean']
        dataset_obs_var = batch['dataset_obs_var']
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

        # if self.config['div_type'] == 'hutchinson_normal':
        #     zs = jax.random.normal(
        #         z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        # elif self.config['div_type'] == 'hutchinson_rademacher':
        #     zs = jax.random.rademacher(
        #         z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        # else:
        #     zs = None

        # if self.config['noise_type'] == 'normal':
        #     flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
        #         goals, observations,
        #         dataset_obs_mean, dataset_obs_var,
        #         q_rng, actions=actions, info=True)
        # else:
        #     flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
        #         goals, observations,
        #         dataset_obs_mean, dataset_obs_var,
        #         q_rng, actions=actions, info=True)
        flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
            goals, observations,
            dataset_obs_mean, dataset_obs_var,
            q_rng, actions=actions, info=True)

        if self.config['distill_type'] == 'log_prob':
            # assert self.config['noise_type'] != 'marginal'
            # if zs is not None:
            #     log_prob_pred = jax.vmap(lambda z: self.network.select('critic')(
            #         goals, observations, actions, z, params=grad_params), in_axes=-1, out_axes=-1)(zs)
            #     log_prob_pred = log_prob_pred.mean(axis=-1)
            # else:
            #     log_prob_pred = self.network.select('critic')(
            #         goals, observations, actions, params=grad_params)
            log_prob_pred = self.network.select('critic')(
                goals, observations, actions, params=grad_params)

            if self.config['distill_loss_type'] == 'mse':
                log_prob_distill_loss = jnp.square(flow_log_prob - log_prob_pred).mean()
            elif self.config['distill_loss_type'] == 'expectile':
                log_prob_distill_loss = self.expectile_loss(
                    flow_log_prob - log_prob_pred,
                    flow_log_prob - log_prob_pred,
                    self.config['expectile']
                ).mean()

            noise_distill_loss, div_int_distill_loss = 0.0, 0.0
        elif self.config['distill_type'] == 'noise_div_int':
            # assert self.config['noise_type'] != 'marginal'
            shortcut_noise_pred = self.network.select('critic_noise')(
                goals, observations, actions, params=grad_params)
            noise_distill_loss = jnp.square(flow_noise - shortcut_noise_pred).mean()
            # if zs is not None:
            #     shortcut_div_int_pred = jax.vmap(lambda z: self.network.select('critic_div')(
            #         goals, observations, actions, z, params=grad_params), in_axes=-1, out_axes=-1)(zs)
            #     shortcut_div_int_pred = shortcut_div_int_pred.mean(axis=-1)
            # else:
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         goals, observations, actions, params=grad_params)
            shortcut_div_int_pred = self.network.select('critic_div')(
                goals, observations, actions, params=grad_params)

            if self.config['distill_loss_type'] == 'mse':
                div_int_distill_loss = jnp.square(flow_div_int - shortcut_div_int_pred).mean()
            elif self.config['distill_loss_type'] == 'expectile':
                div_int_distill_loss = self.expectile_loss(
                    flow_div_int - shortcut_div_int_pred,
                    flow_div_int - shortcut_div_int_pred,
                    self.config['expectile']
                ).mean()

            # gaussian_log_prob = -0.5 * jnp.sum(
            #     jnp.log(2 * jnp.pi) + shortcut_noise_pred ** 2, axis=-1)
            # log_prob_pred = gaussian_log_prob + shortcut_div_int_pred  # log p_1(g | s, a)
            # log_prob_distill_loss = jnp.square(log_prob_pred - flow_log_prob).mean()
            log_prob_distill_loss = 0.0
        # elif self.config['distill_type'] == 'div_int':
        #     shortcut_div_int_pred = self.network.select('critic')(
        #         goals, observations, actions=actions, params=grad_params)
        #     distill_loss = jnp.square(shortcut_div_int_pred - flow_div_int).mean()
        else:
            # distill_loss = 0.0
            log_prob_distill_loss, noise_distill_loss, div_int_distill_loss = 0.0, 0.0, 0.0
        distill_loss = log_prob_distill_loss + noise_distill_loss + div_int_distill_loss
        critic_loss = fm_loss + distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'flow_matching_loss': fm_loss,
            'distill_loss': distill_loss,
            'log_prob_distill_loss': log_prob_distill_loss,
            'noise_distill_loss': noise_distill_loss,
            'div_int_distill_loss': div_int_distill_loss,
            'flow_log_prob_mean': flow_log_prob.mean(),
            'flow_log_prob_max': flow_log_prob.max(),
            'flow_log_prob_min': flow_log_prob.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss."""

        observations = batch['observations']
        goals = batch['actor_goals']

        # DDPG+BC loss.
        assert not self.config['discrete']

        dist = self.network.select('actor')(observations, goals, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            rng, q_action_rng = jax.random.split(rng)
            q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

        # rng, z_rng = jax.random.split(rng)
        # if self.config['div_type'] == 'hutchinson_normal':
        #     zs = jax.random.normal(
        #         z_rng,
        #         shape=(*batch['actor_goals'].shape, self.config['num_hutchinson_ests']),
        #         dtype=batch['actor_goals'].dtype
        #     )
        # elif self.config['div_type'] == 'hutchinson_rademacher':
        #     zs = jax.random.rademacher(
        #         z_rng,
        #         shape=(*batch['actor_goals'].shape, self.config['num_hutchinson_ests']),
        #         dtype=batch['actor_goals'].dtype
        #     )
        # else:
        #     zs = None

        if self.config['distill_type'] == 'log_prob':
            # assert self.config['noise_type'] != 'marginal'
            # if zs is not None:
            #     q = jax.vmap(lambda z: self.network.select('critic')(
            #         batch['actor_goals'], batch['observations'], q_actions, z), in_axes=-1, out_axes=-1)(zs)
            #     q = q.mean(axis=-1)
            # else:
            #     q = self.network.select('critic')(
            #         batch['actor_goals'], batch['observations'], q_actions)
            q = self.network.select('critic')(
                goals, observations, q_actions)
        elif self.config['distill_type'] == 'noise_div_int':
            # assert self.config['noise_type'] != 'marginal'
            shortcut_noise_pred = self.network.select('critic_noise')(
                goals, observations, q_actions)
            # if zs is not None:
            #     shortcut_div_int_pred = jax.vmap(lambda z: self.network.select('critic_div')(
            #         batch['actor_goals'], batch['observations'], q_actions, z), in_axes=-1, out_axes=-1)(zs)
            #     shortcut_div_int_pred = shortcut_div_int_pred.mean(axis=-1)
            # else:
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         batch['actor_goals'], batch['observations'], q_actions)
            shortcut_div_int_pred = self.network.select('critic_div')(
                goals, observations, q_actions)

            if self.config['noise_type'] == 'normal':
                gaussian_log_prob = -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi) + shortcut_noise_pred ** 2, axis=-1)
            else:
                # gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
                gaussian_log_prob = -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi) + jnp.log(batch['dataset_obs_var'][None])
                    + (shortcut_noise_pred - batch['dataset_obs_mean'][None]) ** 2 / batch['dataset_obs_var'][None],
                    axis=-1
                )
            q = gaussian_log_prob + shortcut_div_int_pred  # log p_1(g | s, a)
        # elif self.config['distill_type'] == 'div_int':
        #     q = self.network.select('critic')(
        #         batch['actor_goals'], batch['observations'], actions=q_actions)
        else:
            rng, q_rng = jax.random.split(rng)
            q = self.compute_log_likelihood(
                batch['actor_goals'], batch['observations'],
                batch['dataset_obs_mean'], batch['dataset_obs_var'],
                q_rng, actions=q_actions,
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

    # def compute_rev_flow_samples(self, goals, observations, actions=None):
    #     def vector_field(time, noises, carry):
    #         (observations, actions) = carry
    #         times = jnp.full(noises.shape[:-1], time)
    #
    #         vf = self.network.select('critic_vf')(
    #             noises, times, observations, actions=actions)
    #
    #         return vf
    #
    #     ode_term = ODETerm(vector_field)
    #     ode_sol = diffeqsolve(
    #         ode_term, self.ode_solver,
    #         t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
    #         y0=goals, args=(observations, actions),
    #         adjoint=self.ode_adjoint,
    #     )
    #     noises = ode_sol.ys[-1]
    #
    #     return noises

    # def compute_log_likelihood(self, goals, observations, rng, actions=None,
    #                            info=False):
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
    #         noisy_goals, div_int, z = carry
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
    #                 # def div_func(noisy_goal, time, observation, action):
    #                 #     jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
    #                 #     # jac = jac.reshape([noisy_goal.shape[-1], noisy_goal.shape[-1]])
    #                 #
    #                 #     return jnp.trace(jac, axis1=-2, axis2=-1)
    #
    #                 vf = self.network.select('critic_vf')(
    #                     noisy_goals, times, observations, actions)
    #
    #                 if actions is not None:
    #                     jac = jax.vmap(
    #                         jax.jacrev(vf_func),
    #                         in_axes=(0, 0, 0, 0), out_axes=0
    #                     )(noisy_goals, times, observations, actions)
    #                 else:
    #                     jac = jax.vmap(
    #                         jax.jacrev(vf_func),
    #                         in_axes=(0, 0, 0, None), out_axes=0
    #                     )(noisy_goals, times, observations, actions)
    #
    #                 div = jnp.trace(jac, axis1=-2, axis2=-1)
    #
    #                 return vf, div
    #
    #             vf, div = compute_exact_div(noisy_goals, times, observations, actions)
    #         else:
    #             def compute_hutchinson_div(noisy_goals, times, observations, actions, z):
    #                 # def vf_func(goals):
    #                 #     vf = self.network.select('critic_vf')(
    #                 #         goals,
    #                 #         times,
    #                 #         observations,
    #                 #         actions=actions,
    #                 #     )
    #                 #
    #                 #     return vf
    #
    #                 def single_jvp(z):
    #                     vf, jac_vf_dot_z = jax.jvp(
    #                         lambda n: self.network.select('critic_vf')(n, times, observations, actions),
    #                         (noisy_goals,), (z,)
    #                     )
    #
    #                     return vf, jac_vf_dot_z
    #
    #                 # Split RNG and sample noise
    #                 # if self.config['div_type'] == 'hutchinson_normal':
    #                 #     z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #                 # elif self.config['div_type'] == 'hutchinson_rademacher':
    #                 #     z = jax.random.rademacher(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #
    #                 # Forward (vf) and linearization (jac_vf_dot_z)
    #                 # vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z, ))
    #                 vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
    #
    #                 # Hutchinson's trace estimator
    #                 div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)
    #
    #                 vf = vf[..., 0]
    #                 div = div.mean(axis=-1)
    #
    #                 return vf, div
    #
    #             vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, z)
    #
    #         # Update goals and divergence integral. We need to consider Q ensemble here.
    #         new_noisy_goals = noisy_goals - vf * step_size
    #         new_div_int = div_int - div * step_size
    #
    #         # Return updated carry and scan output
    #         return (new_noisy_goals, new_div_int, z), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     rng, z_rng = jax.random.split(rng)
    #     if self.config['div_type'] == 'hutchinson_normal':
    #         z = jax.random.normal(
    #             z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     elif self.config['div_type'] == 'hutchinson_rademacher':
    #         z = jax.random.rademacher(
    #             z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     else:
    #         z = None
    #     (noisy_goals, div_int, _), _ = jax.lax.scan(
    #         body_fn, (noisy_goals, div_int, z), jnp.arange(num_flow_steps))
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noisy_goals, div_int
    #     else:
    #         return log_prob

    def compute_log_likelihood(self, goals, observations,
                               dataset_obs_mean, dataset_obs_var,
                               rng, actions=None,
                               info=False):
        if self.config['div_type'] == 'exact':
            @jax.jit
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

            @jax.jit
            def vector_field(time, noise_div_int, carry):
                noises, _ = noise_div_int
                observations, actions, zs = carry

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

                if self.config['hutchinson_prod_type'] == 'jvp':
                    def single_jvp(z):
                        vf, jac_vf_z_prod = jax.jvp(
                            lambda n: self.network.select('critic_vf')(n, times, observations, actions),
                            (noises,), (z,)
                        )

                        return vf, jac_vf_z_prod

                    vf, jac_vf_z_prod = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(zs)
                    div = jnp.einsum("ijl,ijl->il", zs, jac_vf_z_prod).mean(axis=-1)
                    vf = vf[..., 0]
                elif self.config['hutchinson_prod_type'] == 'vjp':
                    def single_vjp(z):
                        vf, vjp_func = jax.vjp(
                            lambda n: self.network.select('critic_vf')(n, times, observations, actions),
                            noises
                        )

                        z_trans_jac_vf_prod = vjp_func(z)[0]

                        return vf, z_trans_jac_vf_prod

                    vf, z_trans_jac_vf_prod = jax.vmap(single_vjp, in_axes=-1, out_axes=-1)(zs)
                    div = jnp.einsum("ijl,ijl->il", z_trans_jac_vf_prod, zs).mean(axis=-1)
                    vf = vf[..., 0]
                elif self.config['hutchinson_prod_type'] == 'grad':
                    def single_grad(z):
                        _, vjp_func = jax.vjp(
                            lambda n: jnp.einsum(
                                "ij,ij->i",
                                self.network.select('critic_vf')(n, times, observations, actions),
                                z
                            ),
                            noises
                        )

                        grad = vjp_func(jnp.ones((z.shape[0], ), dtype=z.dtype))[0]
                        return grad

                    vf = self.network.select('critic_vf')(noises, times, observations, actions)
                    grad_vf_dot_z = jax.vmap(single_grad, in_axes=-1, out_axes=-1)(zs)
                    div = jnp.einsum("ijl,ijl->il", grad_vf_dot_z, zs).mean(axis=-1)

                # vf = vf[..., 0]  # vf are the same along the final dimension
                # div = div.mean(axis=-1)

                return vf, div

        rng, z_rng = jax.random.split(rng)
        if self.config['div_type'] == 'hutchinson_normal':
            zs = jax.random.normal(
                z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        elif self.config['div_type'] == 'hutchinson_rademacher':
            zs = jax.random.rademacher(
                z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        else:
            zs = None

        ode_term = ODETerm(vector_field)
        ode_sol = diffeqsolve(
            ode_term, self.ode_solver,
            t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
            y0=(goals, jnp.zeros(goals.shape[:-1])),
            args=(observations, actions, zs),
            adjoint=self.ode_adjoint,
            throw=False,  # (chongyi): setting throw to false is important for speed
        )
        noises, div_int = jax.tree.map(
            lambda x: x[-1], ode_sol.ys)

        # Finally, compute log_prob using the final noisy_goals and div_int
        if self.config['noise_type'] == 'normal':
            gaussian_log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
        else:
            # gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
            gaussian_log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + jnp.log(dataset_obs_var[None])
                + (noises - dataset_obs_mean[None]) ** 2 / dataset_obs_var[None],
                axis=-1
            )
        log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

        if info:
            return log_prob, noises, div_int
        else:
            return log_prob

    # def compute_log_likelihood(self, goals, observations, key, actions=None,
    #                            info=False):
    #     if self.config['div_type'] == 'exact':
    #         @jax.jit
    #         def vector_field(time, noise_div_int, carry):
    #             noises, _ = noise_div_int
    #             observations, actions, _ = carry
    #
    #             # Forward (vf) and linearization (jac_vf_dot_z)
    #             times = jnp.full(noises.shape[:-1], time)
    #
    #             def single_vf(t, n, obs, a):
    #                 noise = jnp.expand_dims(n, 0)
    #                 time = jnp.expand_dims(t, 0)
    #                 observation = jnp.expand_dims(obs, 0)
    #                 if a is not None:
    #                     action = jnp.expand_dims(a, 0)
    #                 else:
    #                     action = a
    #
    #                 vf = self.network.select('critic_vf')(
    #                     noise, time, observation, action).squeeze(0)
    #
    #                 return vf
    #
    #             vf = self.network.select('critic_vf')(
    #                 noises, times, observations, actions)
    #
    #             if actions is not None:
    #                 jac = jax.vmap(
    #                     jax.jacrev(single_vf, argnums=1),
    #                     in_axes=(0, 0, 0, 0), out_axes=0
    #                 )(times, noises, observations, actions)
    #             else:
    #                 jac = jax.vmap(
    #                     jax.jacrev(single_vf, argnums=1),
    #                     in_axes=(0, 0, 0, None), out_axes=0
    #                 )(times, noises, observations, actions)
    #
    #             div = jnp.trace(jac, axis1=-2, axis2=-1)
    #
    #             return vf, div
    #     else:
    #
    #         @jax.jit
    #         def vector_field(time, noise_div_int, carry):
    #             noises, _ = noise_div_int
    #             observations, actions, z = carry
    #
    #             # Forward (vf) and linearization (jac_vf_dot_z)
    #             times = jnp.full(noises.shape[:-1], time)
    #
    #             def single_jvp(z):
    #                 vf, jac_vf_dot_z = jax.jvp(
    #                     lambda n: self.network.select('critic_vf')(n, times, observations, actions),
    #                     (noises,), (z,)
    #                 )
    #
    #                 return vf, jac_vf_dot_z
    #
    #             vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
    #             div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)
    #
    #             vf = vf[..., 0]  # vf are the same along the final dimension
    #             div = div.mean(axis=-1)
    #
    #             return vf, div
    #
    #     key, z_key = jax.random.split(key)
    #     if self.config['div_type'] == 'hutchinson_normal':
    #         z = jax.random.normal(
    #             z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     elif self.config['div_type'] == 'hutchinson_rademacher':
    #         z = jax.random.rademacher(
    #             z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     else:
    #         z = None
    #
    #     ode_term = ODETerm(vector_field)
    #     ode_sol = diffeqsolve(
    #         ode_term, self.ode_solver,
    #         t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
    #         y0=(goals, jnp.zeros(goals.shape[:-1])),
    #         args=(observations, actions, z),
    #         adjoint=self.ode_adjoint,
    #     )
    #     noises, div_int = jax.tree.map(
    #         lambda x: x[-1], ode_sol.ys)
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noises, div_int
    #     else:
    #         return log_prob

    # def compute_log_likelihood(self, goals, observations, key, actions=None,
    #                            info=False):
    #     key, z_key = jax.random.split(key)
    #     if self.config['div_type'] == 'hutchinson_normal':
    #         z = jax.random.normal(
    #             z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     elif self.config['div_type'] == 'hutchinson_rademacher':
    #         z = jax.random.rademacher(
    #             z_key, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     else:
    #         z = None
    #
    #     # reference: https://github.com/jax-ml/jax/issues/7269#issuecomment-879190286
    #     sol = odeint(
    #         lambda y, t, *args: jax.tree.map(lambda x: -x, self.vector_field_func(y, -t, *args)),
    #         (goals, jnp.zeros(goals.shape[:-1], dtype=goals.dtype)),
    #         -jnp.array([1.0, 0.0], dtype=goals.dtype),
    #         observations, actions, z,
    #     )
    #
    #     noises, div_int = jax.tree.map(
    #         lambda x: x[-1], sol)
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noises, div_int
    #     else:
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
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],))

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
        # if config['distill_type'] == 'noise_div_int':
        #     network_info['critic_noise'] = (critic_noise_def, (ex_goals, ex_observations, ex_actions))
        #     network_info['critic_div'] = (critic_div_def, (ex_goals, ex_observations, ex_actions))
        # else:
        #     network_info['critic'] = (critic_def, (ex_goals, ex_observations, ex_actions))

        if config['distill_type'] == 'noise_div_int':
            network_info.update(
                critic_noise=(critic_noise_def, (ex_goals, ex_observations, ex_actions)),
                critic_div=(critic_div_def, (ex_goals, ex_observations, ex_actions)),
            )
            # if config['div_type'] == 'exact':
            #     network_info.update(
            #         critic_div=(critic_div_def, (ex_goals, ex_observations, ex_actions)),
            #     )
            # else:
            #     network_info.update(
            #         critic_div=(critic_div_def, (ex_goals, ex_observations, ex_actions, ex_zs)),
            #     )
        else:
            # if config['div_type'] == 'exact':
            #     network_info.update(
            #         critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
            #     )
            # else:
            #     network_info.update(
            #         critic=(critic_def, (ex_goals, ex_observations, ex_actions, ex_zs)),
            #     )
            network_info.update(
                critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
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
        elif config['ode_adjoint_type'] == 'back_solve':
            ode_adjoint = BacksolveAdjoint()
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
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5', 'tsit5').
            ode_adjoint_type='recursive_checkpoint',  # Type of ODE adjoint ('recursive_checkpoint', 'direct', 'back_solve').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            noise_type='normal',  # Noise distribution type ('normal', 'marginal')
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            div_type='exact',  # Divergence estimator type ('exact', 'hutchinson_normal', 'hutchinson_rademacher').
            hutchinson_prod_type='vjp',  # Hutchinson estimator product type. ('vjp', 'jvp', 'grad')
            distill_type='none',  # Distillation type ('none', 'log_prob', 'rev_int').
            distill_loss_type='mse',  # Distillation loss type. ('mse', 'expectile').
            expectile=0.9,  # IQL style expectile.
            num_hutchinson_ests=4,  # Number of random vectors for hutchinson divergence estimation.
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
