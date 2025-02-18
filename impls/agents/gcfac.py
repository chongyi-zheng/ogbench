import copy
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
from utils.networks import GCFMVectorField, GCFMValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class GCFlowActorCriticAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Flow Actor Critic (GCFAC) agent.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    ode_solver: Any
    ode_adjoint: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the contrastive value loss for the Q or V function."""
        rng, time_rng, noise_rng = jax.random.split(rng, 3)

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        noises = jax.random.normal(noise_rng, shape=goals.shape, dtype=goals.dtype)
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select('critic_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        fm_loss = jnp.square(vf_pred - path_sample.dx_t).mean()
        critic_loss = fm_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'flow_matching_loss': fm_loss,
        }

    def distillation_loss(self, batch, grad_params, rng):
        rng, q_rng = jax.random.split(rng)

        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

        # rng, z_rng = jax.random.split(rng)
        # if self.config['div_type'] == 'hutchinson_normal':
        #     zs = jax.random.normal(rng, shape=goals.shape, dtype=goals.dtype)
        # elif self.config['div_type'] == 'hutchinson_rademacher':
        #     zs = jax.random.rademacher(rng, shape=goals.shape, dtype=goals.dtype)
        # else:
        #     zs = None
        flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
            goals, observations,
            q_rng, actions=actions, info=True,
            use_target_network=self.config['use_target_network']
        )
        # flow_log_prob = jnp.clip(flow_log_prob, -self.config['log_prob_clip'], self.config['log_prob_clip'])

        # if zs is not None:
        #     zs = zs.reshape([goals.shape[0], -1])

        if self.config['distill_type'] == 'log_prob':
            # if zs is not None:
            #     # log_prob_pred = jax.vmap(lambda z: self.network.select('critic')(
            #     #     goals, observations, actions, z, params=grad_params), in_axes=-1, out_axes=-1)(zs)
            #     # log_prob_pred = jnp.clip(log_prob_pred, -self.config['log_prob_clip'], self.config['log_prob_clip'])
            #     # log_prob_pred = log_prob_pred.mean(axis=-1)
            #     log_prob_pred = self.network.select('critic')(
            #         goals, observations, actions, zs, params=grad_params)
            #     log_prob_pred = jnp.clip(log_prob_pred, -self.config['log_prob_clip'], self.config['log_prob_clip'])
            # else:
            #     log_prob_pred = self.network.select('critic')(
            #         goals, observations, actions, params=grad_params)
            log_prob_pred = self.network.select('critic')(
                goals, observations, actions, params=grad_params)
            log_prob_distill_loss = jnp.square(log_prob_pred - flow_log_prob).mean()

            noise_distill_loss, div_int_distill_loss = 0.0, 0.0
        elif self.config['distill_type'] == 'noise_div_int':
            shortcut_noise_pred = self.network.select('critic_noise')(
                goals, observations, actions, params=grad_params)
            noise_distill_loss = jnp.square(shortcut_noise_pred - flow_noise).mean()
            # if zs is not None:
            #     # shortcut_div_int_pred = jax.vmap(lambda z: self.network.select('critic_div')(
            #     #     goals, observations, actions, z, params=grad_params), in_axes=-1, out_axes=-1)(zs)
            #     # shortcut_div_int_pred = shortcut_div_int_pred.mean(axis=-1)
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         goals, observations, actions, zs, params=grad_params)
            # else:
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         goals, observations, actions, params=grad_params)
            shortcut_div_int_pred = self.network.select('critic_div')(
                goals, observations, actions, params=grad_params)
            div_int_distill_loss = jnp.square(shortcut_div_int_pred - flow_div_int).mean()

            # gaussian_log_prob = -0.5 * jnp.sum(
            #     jnp.log(2 * jnp.pi) + shortcut_noise_pred ** 2, axis=-1)
            # log_prob_pred = gaussian_log_prob + shortcut_div_int_pred  # log p_1(g | s, a)
            # log_prob_pred = jnp.clip(log_prob_pred, -self.config['log_prob_clip'], self.config['log_prob_clip'])
            # log_prob_distill_loss = jnp.square(log_prob_pred - flow_log_prob).mean()
            log_prob_distill_loss = 0.0
        # elif self.config['distill_type'] == 'div_int':
        #     shortcut_div_int_pred = self.network.select('critic')(
        #         goals, observations, actions=actions, params=grad_params)
        #     div_int_distill_loss = jnp.square(shortcut_div_int_pred - flow_div_int).mean()
        #
        #     log_prob_distill_loss, noise_distill_loss = 0.0, 0.0
        else:
            log_prob_distill_loss, noise_distill_loss, div_int_distill_loss = 0.0, 0.0, 0.0
        distill_loss = log_prob_distill_loss + noise_distill_loss + div_int_distill_loss

        return distill_loss, {
            'distill_loss': distill_loss,
            'log_prob_distill_loss': log_prob_distill_loss,
            'noise_distill_loss': noise_distill_loss,
            'div_int_distill_loss': div_int_distill_loss,
            'flow_log_prob_mean': flow_log_prob.mean(),
            'flow_log_prob_max': flow_log_prob.max(),
            'flow_log_prob_min': flow_log_prob.min(),
        }

    def flow_matching_loss(self, batch, grad_params, rng):
        """Compute the flow matching loss."""

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['actor_goals']

        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        times = jax.random.uniform(time_rng, shape=(batch_size,))
        path_sample = self.cond_prob_path(x_0=noises, x_1=actions, t=times)
        vf_pred = self.network.select('actor_vf')(
            path_sample.x_t,
            times,
            observations,
            goals,
            params=grad_params,
        )
        actor_flow_matching_loss = jnp.square(vf_pred - path_sample.dx_t).mean()

        return actor_flow_matching_loss, {
            'actor_flow_matching_loss': actor_flow_matching_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss."""
        # batch_size, action_dim = batch['actions'].shape

        # DDPG+BC loss.
        # assert not self.config['discrete']
        #
        # dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        # if self.config['const_std']:
        #     q_actions = jnp.clip(dist.mode(), -1, 1)
        # else:
        #     rng, q_action_rng = jax.random.split(rng)
        #     q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        if self.config['actor_distill_type'] == 'fwd_sample':
            q_actions = self.network.select('actor')(
                noises, batch['observations'], batch['actor_goals'], params=grad_params)
        elif self.config['actor_distill_type'] == 'fwd_int':
            q_actions = noises + self.network.select('actor')(
                noises, batch['observations'], batch['actor_goals'], params=grad_params)
        q_actions = jnp.clip(q_actions, -1, 1)

        flow_q_actions = self.compute_fwd_flow_samples(
            noises, batch['observations'], batch['actor_goals'])
        flow_q_actions = jnp.clip(flow_q_actions, -1, 1)

        # distill_loss = self.config['alpha'] * jnp.pow(q_actions - flow_q_actions, 2).mean()

        # rng, z_rng = jax.random.split(rng)
        # if self.config['div_type'] == 'hutchinson_normal':
        #     zs = jax.random.normal(
        #         z_rng, shape=batch['actor_goals'].shape, dtype=batch['actor_goals'].dtype)
        # elif self.config['div_type'] == 'hutchinson_rademacher':
        #     zs = jax.random.rademacher(
        #         z_rng, shape=batch['actor_goals'].shape, dtype=batch['actor_goals'].dtype)
        # else:
        #     zs = None

        # if zs is not None:
        #     zs = zs.reshape([batch['actor_goals'].shape[0], -1])

        if self.config['distill_type'] == 'log_prob':
            # if zs is not None:
            #     # q = jax.vmap(lambda z: self.network.select('critic')(
            #     #     batch['actor_goals'], batch['observations'], q_actions, z), in_axes=-1, out_axes=-1)(zs)
            #     # q = q.mean(axis=-1)
            #     q = self.network.select('critic')(
            #         batch['actor_goals'], batch['observations'], q_actions, zs)
            # else:
            #     q = self.network.select('critic')(
            #         batch['actor_goals'], batch['observations'], q_actions)
            q = self.network.select('critic')(
                batch['actor_goals'], batch['observations'], q_actions)
        elif self.config['distill_type'] == 'noise_div_int':
            shortcut_noise_pred = self.network.select('critic_noise')(
                batch['actor_goals'], batch['observations'], q_actions)
            # if zs is not None:
            #     # shortcut_div_int_pred = jax.vmap(lambda z: self.network.select('critic_div')(
            #     #     batch['actor_goals'], batch['observations'], q_actions, z), in_axes=-1, out_axes=-1)(zs)
            #     # shortcut_div_int_pred = shortcut_div_int_pred.mean(axis=-1)
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         batch['actor_goals'], batch['observations'], q_actions, zs)
            # else:
            #     shortcut_div_int_pred = self.network.select('critic_div')(
            #         batch['actor_goals'], batch['observations'], q_actions)
            shortcut_div_int_pred = self.network.select('critic_div')(
                batch['actor_goals'], batch['observations'], q_actions)

            gaussian_log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + shortcut_noise_pred ** 2, axis=-1)
            q = gaussian_log_prob + shortcut_div_int_pred  # log p_1(g | s, a)
        # elif self.config['distill_type'] == 'div_int':
        #     q = self.network.select('critic')(
        #         batch['actor_goals'], batch['observations'], actions=q_actions)
        else:
            # q = self.compute_log_likelihood(
            #     batch['actor_goals'], batch['observations'],
            #     zs, actions=q_actions,
            # )
            rng, q_rng = jax.random.split(rng)
            q = self.compute_log_likelihood(
                batch['actor_goals'], batch['observations'],
                q_rng, actions=q_actions,
            )

            # q = self.compute_log_likelihood(
            #     batch['actor_goals'], batch['observations'], q_rng, actions=q_actions)
        # q = jnp.clip(q, -self.config['log_prob_clip'], self.config['log_prob_clip'])

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
        # log_prob = dist.log_prob(batch['actions'])

        # bc_loss = -(self.config['alpha'] * log_prob).mean()
        distill_loss = self.config['alpha'] * jnp.square(q_actions - flow_q_actions).mean()
        actor_loss = q_loss + distill_loss

        # Additional metrics for logging.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(
            noise_rng, shape=batch['actions'].shape, dtype=batch['actions'].dtype)
        if self.config['actor_distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(
                noises, batch['observations'], batch['actor_goals'])
        elif self.config['actor_distill_type'] == 'fwd_int':
            actions = noises + self.network.select('actor')(
                noises, batch['observations'], batch['actor_goals'])
        # actions = noises + self.network.select('actor')(
        #     noises, batch['observations'], batch['actor_goals'])
        actions = jnp.clip(actions, -1, 1)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            # 'bc_loss': bc_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            # 'bc_log_prob': log_prob.mean(),
            # 'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            # 'std': jnp.mean(dist.scale_diag),
            'mse': mse,
        }

    # def compute_fwd_flow_samples(self, noises, observations, goals):
    #     def vector_field(time, noises, carry):
    #         observations, goals = carry
    #         times = jnp.full(noises.shape[:-1], time)
    #
    #         vf = self.network.select('actor_vf')(
    #             noises, times, observations, goals)
    #
    #         return vf
    #
    #     ode_term = ODETerm(vector_field)
    #     ode_sol = diffeqsolve(
    #         ode_term, self.ode_solver,
    #         t0=0.0, t1=1.0, dt0=1 / self.config['num_flow_steps'],
    #         y0=noises, args=(observations, goals),
    #         adjoint=self.ode_adjoint,
    #         throw=False,  # (chongyi): setting throw to false is important for speed
    #     )
    #     noises = ode_sol.ys[-1]
    #
    #     return noises

    def compute_fwd_flow_samples(self, noises, observations, goals):
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
                noisy_actions, times, observations, goals)
            new_noisy_actions = noisy_actions + vf * step_size

            return (new_noisy_actions,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_actions,), _ = jax.lax.scan(
            body_fn, (noisy_actions,), jnp.arange(num_flow_steps))

        return noisy_actions

    # def compute_log_likelihood(self, goals, observations,
    #                            key, actions=None,
    #                            info=False):
    #     if self.config['div_type'] == 'exact':
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
    #         def vector_field(time, noise_div_int, carry):
    #             noises, _ = noise_div_int
    #             observations, actions, z = carry
    #
    #             # Split RNG and sample noise
    #             # key, z_key = jax.random.split(key)
    #             # if self.config['div_type'] == 'hutchinson_normal':
    #             #     z = jax.random.normal(z_key, shape=goals.shape, dtype=goals.dtype)
    #             # elif self.config['div_type'] == 'hutchinson_rademacher':
    #             #     z = jax.random.rademacher(z_key, shape=goals.shape, dtype=goals.dtype)
    #
    #             # Forward (vf) and linearization (jac_vf_dot_z)
    #             times = jnp.full(noises.shape[:-1], time)
    #             # vf, jac_vf_dot_z = jax.jvp(
    #             #     lambda n: self.network.select('critic_vf')(n, times, observations, actions),
    #             #     (noises,), (z,)
    #             # )
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
    #         throw=False,  # (chongyi): setting throw to false is important for speed
    #     )
    #     noises, div_int = jax.tree.map(
    #         lambda x: x[-1], ode_sol.ys)
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(
    #         jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noises, div_int
    #     else:
    #         return log_prob

    # def compute_log_likelihood(self, goals, observations, zs, actions=None,
    #                            info=False, use_target_network=False):
    #     if use_target_network:
    #         module_name = 'target_critic_vf'
    #     else:
    #         module_name = 'critic_vf'
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
    #         noisy_goals, div_int, zs = carry
    #
    #         # Time for this iteration
    #         times = jnp.full(noisy_goals.shape[:-1], 1.0 - i * step_size)
    #
    #         if self.config['div_type'] == 'exact':
    #             def compute_exact_div(noisy_goals, times, observations, actions):
    #                 def vf_func(noisy_goal, time, observation, action):
    #                     noisy_goal = jnp.expand_dims(noisy_goal, 0)
    #                     time = jnp.expand_dims(time, 0)
    #                     observation = jnp.expand_dims(observation, 0)
    #                     if action is not None:
    #                         action = jnp.expand_dims(action, 0)
    #                     vf = self.network.select(module_name)(
    #                         noisy_goal, time, observation, action).squeeze(0)
    #
    #                     return vf
    #
    #                 vf = self.network.select(module_name)(
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
    #             def compute_hutchinson_div(noisy_goals, times, observations, actions, zs):
    #                 # Split RNG and sample noise
    #                 # if self.config['div_type'] == 'hutchinson_normal':
    #                 #     z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #                 # elif self.config['div_type'] == 'hutchinson_rademacher':
    #                 #     z = jax.random.rademacher(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)
    #
    #                 # Forward (vf) and linearization (jac_vf_dot_z)
    #                 vf, jac_vf_dot_z = jax.jvp(
    #                     lambda gs: self.network.select(module_name)(
    #                         gs, times, observations, actions),
    #                     (noisy_goals,), (zs,)
    #                 )
    #                 # vf = vf.reshape([-1, *vf.shape[1:]])
    #                 # jac_vf_dot_z = jac_vf_dot_z.reshape([-1, *jac_vf_dot_z.shape[1:]])
    #
    #                 # Hutchinson's trace estimator
    #                 # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
    #                 div = jnp.einsum("ij,ij->i", jac_vf_dot_z, zs)
    #
    #                 return vf, div
    #
    #             # rng, div_rng = jax.random.split(rng)
    #             vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, zs)
    #
    #         new_noisy_goals = noisy_goals - vf * step_size
    #         new_div_int = div_int - div * step_size
    #
    #         # Return updated carry and scan output
    #         return (new_noisy_goals, new_div_int, zs), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     (noisy_goals, div_int, _), _ = jax.lax.scan(
    #         body_fn, (noisy_goals, div_int, zs), jnp.arange(num_flow_steps))
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noisy_goals, div_int
    #     else:
    #         return log_prob

    # def compute_log_likelihood(self, goals, observations, zs, actions=None,
    #                            info=False, use_target_network=False):
    #     if use_target_network:
    #         module_name = 'target_critic_vf'
    #     else:
    #         module_name = 'critic_vf'
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
    #         noisy_goals, div_int, zs = carry
    #         # rng, div_rng = jax.random.split(rng)
    #
    #         # Time for this iteration
    #         times = jnp.full(noisy_goals.shape[:-1], 1.0 - i * step_size)
    #
    #         if self.config['div_type'] == 'exact':
    #             def compute_exact_div(noisy_goals, times, observations, actions):
    #                 def vf_func(noisy_goal, time, observation, action):
    #                     noisy_goal = jnp.expand_dims(noisy_goal, 0)
    #                     time = jnp.expand_dims(time, 0)
    #                     observation = jnp.expand_dims(observation, 0)
    #                     if action is not None:
    #                         action = jnp.expand_dims(action, 0)
    #                     vf = self.network.select(module_name)(
    #                         noisy_goal, time, observation, action).squeeze(0)
    #
    #                     return vf
    #
    #                 vf = self.network.select(module_name)(
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
    #             def compute_hutchinson_div(noisy_goals, times, observations, actions, zs):
    #                 def single_jvp(z):
    #                     vf, jac_vf_dot_z = jax.jvp(
    #                         lambda g: self.network.select(module_name)(g, times, observations, actions),
    #                         (noisy_goals,), (z,)
    #                     )
    #
    #                     return vf, jac_vf_dot_z
    #
    #                 # Split RNG and sample noise
    #                 # if self.config['div_type'] == 'hutchinson_normal':
    #                 #     z = jax.random.normal(
    #                 #         rng,
    #                 #         shape=(*noisy_goals.shape, self.config['num_hutchinson_ests']),
    #                 #         dtype=noisy_goals.dtype
    #                 #     )
    #                 # elif self.config['div_type'] == 'hutchinson_rademacher':
    #                 #     z = jax.random.rademacher(
    #                 #         rng,
    #                 #         shape=(*noisy_goals.shape, self.config['num_hutchinson_ests']),
    #                 #         dtype=noisy_goals.dtype
    #                 #     )
    #                 # else:
    #                 #     raise NotImplementedError
    #
    #                 # Forward (vf) and linearization (jac_vf_dot_z)
    #                 # vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z, ))
    #                 vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(zs)
    #
    #                 # Hutchinson's trace estimator
    #                 div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, zs)
    #
    #                 vf = vf[..., 0]
    #                 div = div.mean(axis=-1)
    #
    #                 return vf, div
    #
    #             vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, zs)
    #
    #         new_noisy_goals = noisy_goals - vf * step_size
    #         new_div_int = div_int - div * step_size
    #
    #         # Return updated carry and scan output
    #         return (new_noisy_goals, new_div_int, zs), None
    #
    #     # Use lax.scan to iterate over num_flow_steps
    #     # rng, z_rng = jax.random.split(rng)
    #     # if self.config['div_type'] == 'hutchinson_normal':
    #     #     zs = jax.random.normal(
    #     #         z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     # elif self.config['div_type'] == 'hutchinson_rademacher':
    #     #     zs = jax.random.rademacher(
    #     #         z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
    #     # else:
    #     #     zs = None
    #     (noisy_goals, div_int, _), _ = jax.lax.scan(
    #         body_fn, (noisy_goals, div_int, zs), jnp.arange(num_flow_steps))
    #
    #     # Finally, compute log_prob using the final noisy_goals and div_int
    #     gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
    #     log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)
    #
    #     if info:
    #         return log_prob, noisy_goals, div_int
    #     else:
    #         return log_prob

    def compute_log_likelihood(self, goals, observations, rng, actions=None,
                               info=False, use_target_network=False):
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

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

                    vf = self.network.select(module_name)(
                        noise, time, observation, action).squeeze(0)

                    return vf

                vf = self.network.select(module_name)(
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

                # Forward (vf) and linearization (jac_vf_dot_z)
                times = jnp.full(noises.shape[:-1], time)

                def single_jvp(z):
                    vf, jac_vf_dot_z = jax.jvp(
                        lambda n: self.network.select(module_name)(n, times, observations, actions),
                        (noises,), (z,)
                    )

                    return vf, jac_vf_dot_z

                vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
                div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)

                vf = vf[..., 0]  # vf are the same along the final dimension
                div = div.mean(axis=-1)

                return vf, div

        rng, z_rng = jax.random.split(rng)
        if self.config['div_type'] == 'hutchinson_normal':
            z = jax.random.normal(
                z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        elif self.config['div_type'] == 'hutchinson_rademacher':
            z = jax.random.rademacher(
                z_rng, shape=(*goals.shape, self.config['num_hutchinson_ests']), dtype=goals.dtype)
        else:
            z = None

        ode_term = ODETerm(vector_field)
        ode_sol = diffeqsolve(
            ode_term, self.ode_solver,
            t0=1.0, t1=0.0, dt0=-1 / self.config['num_flow_steps'],
            y0=(goals, jnp.zeros(goals.shape[:-1])),
            args=(observations, actions, z),
            adjoint=self.ode_adjoint,
            throw=False,  # (chongyi): setting throw to false is important for speed
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

        rng, critic_rng, distill_rng, flow_matching_rng, actor_rng = jax.random.split(rng, 5)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        distill_loss, distill_info = self.distillation_loss(batch, grad_params, distill_rng)
        for k, v in distill_info.items():
            info[f'distillation/{k}'] = v

        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + distill_loss + flow_matching_loss + actor_loss
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
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""

        # dist = self.network.select('actor')(observations, goals, temperature=temperature)
        # seed, actor_seed = jax.random.split(seed)
        # actions = dist.sample(seed=actor_seed)
        # if not self.config['discrete']:
        #     actions = jnp.clip(actions, -1, 1)

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
            if goals is not None:
                goals = jnp.expand_dims(goals, axis=0)
        action_dim = self.network.model_def.modules['actor'].output_dim

        seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            noise_seed, shape=(observations.shape[0], action_dim), dtype=observations.dtype)
        if self.config['actor_distill_type'] == 'fwd_sample':
            actions = self.network.select('actor')(
                noises, observations, goals)
        elif self.config['actor_distill_type'] == 'fwd_int':
            actions = noises + self.network.select('actor')(
                noises, observations, goals)
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
        
        rng, time_rng, z_rng = jax.random.split(rng, 3)
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0], ), dtype=ex_observations.dtype)
        ex_zs = jax.random.normal(z_rng, shape=ex_observations.shape, dtype=ex_observations.dtype)

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
            encoders['actor_vf_state'] = encoder_module()
            encoders['actor_vf_goal'] = encoder_module()
            encoders['actor_state'] = encoder_module()
            encoders['actor_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        if config['discrete']:
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
            # actor_def = GCDiscreteActor(
            #     hidden_dims=config['actor_hidden_dims'],
            #     action_dim=action_dim,
            #     gc_encoder=encoders.get('actor'),
            # )
            raise NotImplementedError
        else:
            # actor_def = GCActor(
            #     hidden_dims=config['actor_hidden_dims'],
            #     action_dim=action_dim,
            #     layer_norm=config['actor_layer_norm'],
            #     state_dependent_std=False,
            #     const_std=config['const_std'],
            #     gc_encoder=encoders.get('actor'),
            # )

            actor_vf_def = GCFMVectorField(
                vector_dim=action_dim,
                hidden_dims=config['actor_hidden_dims'],
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor_vf_state'),
                goal_encoder=encoders.get('actor_vf_goal'),
            )

            actor_def = GCFMValue(
                hidden_dims=config['actor_hidden_dims'],
                output_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor_state'),
                goal_encoder=encoders.get('actor_goal'),
            )

        network_info = dict(
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (ex_goals, ex_times, ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations, ex_goals)),
            actor=(actor_def, (ex_actions, ex_observations, ex_goals)),
        )

        if config['distill_type'] == 'noise_div_int':
            network_info['critic_noise'] = (critic_noise_def, (ex_goals, ex_observations, ex_actions))
            network_info['critic_div'] = (critic_div_def, (ex_goals, ex_observations, ex_actions))
        else:
            network_info['critic'] = (critic_def, (ex_goals, ex_observations, ex_actions))
        # if config['distill_type'] == 'noise_div_int':
        #     network_info.update(
        #         critic_noise=(critic_noise_def, (ex_goals, ex_observations, ex_actions)),
        #     )
        #     if config['div_type'] == 'exact':
        #         network_info.update(
        #             critic_div=(critic_div_def, (ex_goals, ex_observations, ex_actions)),
        #         )
        #     else:
        #         network_info.update(
        #             critic_div=(critic_div_def, (ex_goals, ex_observations, ex_actions, ex_zs)),
        #         )
        # else:
        #     if config['div_type'] == 'exact':
        #         network_info.update(
        #             critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
        #         )
        #     else:
        #         network_info.update(
        #             critic=(critic_def, (ex_goals, ex_observations, ex_actions, ex_zs)),
        #         )
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
            agent_name='gcfac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5', 'tsit5').
            ode_adjoint_type='recursive_checkpoint',  # Type of ODE adjoint ('recursive_checkpoint', 'direct', 'back_solve').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            div_type='exact',  # Divergence estimator type ('exact', 'hutchinson_normal', 'hutchinson_rademacher').
            distill_type='none',  # Distillation type ('none', 'log_prob', 'noise_div_int').
            actor_distill_type='fwd_sample',  # Actor distillation type ('fwd_sample', 'fwd_int').
            num_hutchinson_ests=4,  # Number of random vectors for hutchinson divergence estimation.
            use_target_network=False,  # Whether to use the target critic vector field to compute the distillation loss.
            log_prob_clip=jnp.inf,
            alpha=0.1,  # BC coefficient in DDPG+BC.
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
