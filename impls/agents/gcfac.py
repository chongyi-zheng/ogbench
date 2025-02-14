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
from utils.networks import GCFMVectorField, GCFMValue, GCActorVectorField
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class GCFlowActorCriticAgent(flax.struct.PyTreeNode):
    """Goal-conditioned Flow Actor Critic (GCFAC) agent.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    vector_field_func: Any = nonpytree_field()
    # ode_term: Any = nonpytree_field()
    ode_solver: Any
    ode_adjoint: Any
    config: Any = nonpytree_field()

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
        #     # TODO (chongyi): flow_div_int is not exactly equivalent to flow_log_prob in this case
        #     flow_log_prob = flow_div_int
        flow_log_prob, flow_noise, flow_div_int = self.compute_log_likelihood(
            goals, observations,
            dataset_obs_mean, dataset_obs_var,
            q_rng, actions=actions, info=True)

        if self.config['distill_type'] == 'log_prob':
            # assert self.config['noise_type'] != 'marginal'
            log_prob_pred = self.network.select('critic')(
                goals, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(log_prob_pred - flow_log_prob).mean()
        elif self.config['distill_type'] == 'noise_div_int':
            # assert self.config['noise_type'] != 'marginal'
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

    def flow_matching_loss(self, batch, grad_params, rng):
        """Compute the flow matching loss."""

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['actor_goals']

        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        if self.config['noise_type'] == 'normal':
            noises = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        else:
            noises = jnp.roll(actions, shift=1, axis=0)
        # noises = jax.random.normal(noise_rng, shape=actions.shape, dtype=actions.dtype)
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
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

        if self.config['distill_type'] == 'log_prob':
            # assert self.config['noise_type'] != 'marginal'
            q = self.network.select('critic')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
        elif self.config['distill_type'] == 'noise_div_int':
            # assert self.config['noise_type'] != 'marginal'
            shortcut_noise_pred = self.network.select('critic_noise')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
            shortcut_div_int_pred = self.network.select('critic_div')(
                batch['actor_goals'], batch['observations'], actions=q_actions)

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
        elif self.config['distill_type'] == 'div_int':
            q = self.network.select('critic')(
                batch['actor_goals'], batch['observations'], actions=q_actions)
        else:
            rng, q_rng = jax.random.split(rng)
            if self.config['noise_type'] == 'normal':
                q = self.compute_log_likelihood(
                    batch['actor_goals'], batch['observations'],
                    batch['dataset_obs_mean'], batch['dataset_obs_var'],
                    q_rng, actions=q_actions,
                )
            else:
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

    def compute_fwd_flow_samples(self, noises, observations, goals):
        def vector_field(time, noises, carry):
            observations, goals = carry
            times = jnp.full(noises.shape[:-1], time)

            vf = self.network.select('actor_vf')(
                noises, times, observations, goals)

            return vf

        ode_term = ODETerm(vector_field)
        ode_sol = diffeqsolve(
            ode_term, self.ode_solver,
            t0=0.0, t1=1.0, dt0=1 / self.config['num_flow_steps'],
            y0=noises, args=(observations, goals),
            adjoint=self.ode_adjoint,
            throw=False,  # (chongyi): setting throw to false is important for speed
        )
        noises = ode_sol.ys[-1]

        return noises

    def compute_log_likelihood(self, goals, observations,
                               dataset_obs_mean, dataset_obs_var,
                               key, actions=None,
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

        rng, flow_matching_rng = jax.random.split(rng)
        flow_matching_loss, flow_matching_info = self.flow_matching_loss(
            batch, grad_params, flow_matching_rng)
        for k, v in flow_matching_info.items():
            info[f'flow_matching/{k}'] = v

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

        # dist = self.network.select('actor')(observations, goals, temperature=temperature)
        # seed, actor_seed = jax.random.split(seed)
        # actions = dist.sample(seed=actor_seed)
        # if not self.config['discrete']:
        #     actions = jnp.clip(actions, -1, 1)

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
            if goals is not None:
                goals = jnp.expand_dims(goals, axis=0)
        action_dim = self.network.model_def.modules['actor'].action_dim

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

            actor_vf_def = GCActorVectorField(
                action_dim=action_dim,
                hidden_dims=config['actor_hidden_dims'],
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor_vf_state'),
                goal_encoder=encoders.get('actor_vf_goal'),
            )

            actor_def = GCActorVectorField(
                action_dim=action_dim,
                hidden_dims=config['actor_hidden_dims'],
                layer_norm=config['actor_layer_norm'],
                state_encoder=encoders.get('actor_vf_state'),
                goal_encoder=encoders.get('actor_vf_goal'),
            )

        network_info = dict(
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            actor_vf=(actor_vf_def, (ex_actions, ex_times, ex_observations, ex_goals)),
            actor=(actor_def, (ex_actions, ex_observations, ex_goals)),
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
        elif config['ode_adjoint_type'] == 'back_solve':
            ode_adjoint = BacksolveAdjoint()
        else:
            raise TypeError("Unknown ode_adjoint_type: {}".format(config['ode_adjoint_type']))

        if config['div_type'] == 'exact':
            def vector_field(noise_div_int, time, observations, actions, z):
                noises, _ = noise_div_int

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

                    vf = network.select('critic_vf')(
                        noise, time, observation, action).squeeze(0)

                    return vf

                vf = network.select('critic_vf')(
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

            def vector_field(noise_div_int, time, observations, actions, z):
                noises, _ = noise_div_int

                # Forward (vf) and linearization (jac_vf_dot_z)
                times = jnp.full(noises.shape[:-1], time)

                def single_jvp(z):
                    vf, jac_vf_dot_z = jax.jvp(
                        lambda n: network.select('critic_vf')(n, times, observations, actions),
                        (noises,), (z,)
                    )

                    return vf, jac_vf_dot_z

                vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
                div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)

                vf = vf[..., 0]  # vf are the same along the final dimension
                div = div.mean(axis=-1)

                return vf, div

        return cls(rng, network=network, cond_prob_path=cond_prob_path,
                   vector_field_func=vector_field,
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
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5', 'tsit5').
            ode_adjoint_type='recursive_checkpoint',  # Type of ODE adjoint ('recursive_checkpoint', 'direct', 'back_solve').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            noise_type='normal',  # Noise distribution type ('normal', 'marginal')
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            div_type='exact',  # Divergence estimator type ('exact', 'hutchinson_normal', 'hutchinson_rademacher').
            distill_type='none',  # Distillation type ('none', 'log_prob', 'rev_int').
            actor_distill_type='fwd_sample',  # Actor distillation type ('fwd_sample', 'fwd_int').
            num_hutchinson_ests=4,  # Number of random vectors for hutchinson divergence estimation.
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
