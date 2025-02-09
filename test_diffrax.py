from diffrax import diffeqsolve, ODETerm, Dopri5, Euler

import numpy as np
import jax.numpy as jnp
import jax


def vector_field(t, eps, args):
    # $f(t, y(t), args) \mathrm{d}t$
    s, a = args

    # times = t * jnp.ones(shape=(eps.shape[0], ))
    times = jnp.full(eps.shape, t)

    return 0.5 * times * eps + s


def compute_rev_flow_samples(goals, observations, actions=None, num_flow_steps=10):
    noises = goals
    step_size = 1.0 / num_flow_steps

    def body_fn(carry, i):
        """
        carry: (noises, )
        i: current step index
        """
        (noises,) = carry

        # Time for this iteration
        # times = 1.0 - jnp.full(goals.shape[:-1], i * step_size)
        times = jnp.array(1.0 - i * step_size)

        conds = (observations, actions)
        vf = vector_field(times, noises, conds)

        # Update goals and divergence integral. We need to consider Q ensemble here.
        new_noises = noises - vf * step_size

        # Return updated carry and scan output
        return (new_noises,), None

    # Use lax.scan to iterate over num_flow_steps
    (noises,), _ = jax.lax.scan(
        body_fn, (noises,), jnp.arange(num_flow_steps))

    return noises


def diffrax_rev_flow_samples(goals, observations, actions=None, num_flow_steps=10):
    term = ODETerm(vector_field)
    solver = Dopri5()
    solution = diffeqsolve(
        term, solver,
        t0=1.0, t1=0.0, dt0=-1 / num_flow_steps,
        y0=goals, args=(observations, actions)
    )
    noises = solution.ys[-1]

    return noises

def compute_log_likelihood(goals, observations, actions, z, return_ratio=False,
                           div_type='hutchinson_normal', num_flow_steps=10):
    """
    Euler method
    """

    noisy_goals = goals
    div_int = jnp.zeros(goals.shape[:-1])
    step_size = 1.0 / num_flow_steps

    # Define the body function to be scanned
    def body_fn(carry, i):
        """
        carry: (noisy_goals, div_int, rng)
        i: current step index
        """
        noisy_goals, div_int, rng = carry

        # Time for this iteration
        # times = 1.0 - jnp.full(noisy_goals.shape[:-1], i * step_size)
        times = 1.0 - i * step_size

        if div_type == 'exact':
            def compute_exact_div(noisy_goals, times, observations, actions):
                def vf_func(noisy_goal, time, observation, action):
                    noisy_goal = jnp.expand_dims(noisy_goal, 0)
                    time = jnp.expand_dims(time, 0)
                    observation = jnp.expand_dims(observation, 0)
                    if action is not None:
                        action = jnp.expand_dims(action, 0)
                    vf = self.network.select('critic_vf')(
                        noisy_goal, time, observation, action).squeeze(0)

                    return vf

                def div_func(noisy_goal, time, observation, action):
                    jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
                    # jac = jac.reshape([noisy_goal.shape[-1], noisy_goal.shape[-1]])

                    return jnp.trace(jac, axis1=-2, axis2=-1)

                vf = self.network.select('critic_vf')(
                    noisy_goals, times, observations, actions)

                if actions is not None:
                    div = jax.vmap(div_func, in_axes=(0, 0, 0, 0), out_axes=0)(
                        noisy_goals, times, observations, actions)
                else:
                    div = jax.vmap(div_func, in_axes=(0, 0, 0, None), out_axes=0)(
                        noisy_goals, times, observations, actions)

                return vf, div

            vf, div = compute_exact_div(noisy_goals, times, observations, actions)
        else:
            def compute_hutchinson_div(noisy_goals, times, observations, actions, rng):
                # Define vf_func for jvp
                def vf_func(goals):
                    # vf = self.network.select('critic_vf')(
                    #     goals,
                    #     times,
                    #     observations,
                    #     actions=actions,
                    # )
                    vf = vector_field(times, goals, (observations, actions))

                    # return vf.reshape([-1, *vf.shape[2:]])
                    return vf

                # def vector_field(t, eps, args):
                #     # $f(t, y(t), args) \mathrm{d}t$
                #     s, a = args
                #
                #     # times = t * jnp.ones(shape=(eps.shape[0], ))
                #     times = t * jnp.ones_like(eps)
                #
                #     return 0.5 * times * eps + s

                # Split RNG and sample noise
                # z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)

                # Forward (vf) and linearization (jac_vf_dot_z)
                vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z,))
                # vf = vf.reshape([-1, *vf.shape[1:]])
                # jac_vf_dot_z = jac_vf_dot_z.reshape([-1, *jac_vf_dot_z.shape[1:]])

                # Hutchinson's trace estimator
                # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
                div = jnp.einsum("ij,ij->i", jac_vf_dot_z, z)

                return vf, div

            rng, div_rng = jax.random.split(rng)
            vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, div_rng)

        # Update goals and divergence integral. We need to consider Q ensemble here.
        # new_noisy_goals = jnp.min(noisy_goals[None] - vf * step_size, axis=0)
        # new_div_int = jnp.min(div_int[None] - div * step_size, axis=0)
        new_noisy_goals = noisy_goals - vf * step_size
        new_div_int = div_int - div * step_size

        # Return updated carry and scan output
        return (new_noisy_goals, new_div_int, rng), None

    # Use lax.scan to iterate over num_flow_steps
    (noisy_goals, div_int, rng), _ = jax.lax.scan(
        body_fn, (noisy_goals, div_int, rng), jnp.arange(num_flow_steps))

    if return_ratio:
        log_ratio = div_int
        return log_ratio
    else:
        # Finally, compute log_prob using the final noisy_goals and div_int
        gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
        log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

        return log_prob

def diffrax_compute_log_likelihood(goals, observations, actions, z, return_ratio=False,
                                   div_type='hutchinson_normal', num_flow_steps=10):

    def vf_func(times, noise_div_int, carry):
        noises, _ = noise_div_int
        observations, actions, z = carry

        # Split RNG and sample noise
        # rng, div_rng = jax.random.split(rng)
        # z = jax.random.normal(div_rng, shape=noises.shape, dtype=noises.dtype)

        # def vf_func(goals):
        #     # vf = self.network.select('critic_vf')(
        #     #     goals,
        #     #     times,
        #     #     observations,
        #     #     actions=actions,
        #     # )
        #     vf = vector_field(times, goals, (observations, actions))
        #
        #     # return vf.reshape([-1, *vf.shape[2:]])
        #     return vf

        # Forward (vf) and linearization (jac_vf_dot_z)
        vf, jac_vf_dot_z = jax.jvp(
            lambda n: vector_field(times, n, (observations, actions)),
            (noises, ), (z,)
        )
        div = jnp.einsum("ij,ij->i", jac_vf_dot_z, z)

        return (vf, div)

    if div_type == 'exact':
        raise NotImplementedError
    else:
        term = ODETerm(vf_func)
        # solver = Dopri5()
        solver = Euler()
        solution = diffeqsolve(
            term, solver,
            t0=1.0, t1=0.0, dt0=-1 / num_flow_steps,
            y0=(goals, jnp.zeros(goals.shape[:-1])),
            args=(observations, actions, z)
        )
        (noises, div_int) = jax.tree_map(
            lambda x: x[-1], solution.ys)

        if return_ratio:
            log_ratio = div_int
            return log_ratio
        else:
            # Finally, compute log_prob using the final noisy_goals and div_int
            gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
            log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

            return log_prob


def main():
    key = jax.random.PRNGKey(np.random.randint(2 ** 32 - 1))
    shape = (16, 2)

    key, goal_key, obs_key, action_key = jax.random.split(key, 4)
    goals = jax.random.normal(goal_key, shape=shape)
    observations = jax.random.normal(obs_key, shape=shape)
    actions = jax.random.normal(action_key, shape=shape)

    noises = compute_rev_flow_samples(goals, observations, actions)
    diffrax_noises = diffrax_rev_flow_samples(goals, observations, actions)

    # assert jnp.allclose(noises, diffrax_noises)

    key, log_p_key, diffrax_log_p_key = jax.random.split(key, 3)
    log_p = compute_log_likelihood(
        goals, observations, actions, log_p_key, return_ratio=True)
    diffrax_log_p = diffrax_compute_log_likelihood(
        goals, observations, actions, diffrax_log_p_key, return_ratio=True)

    assert jnp.allclose(log_p, diffrax_log_p)

    print("same!")


if __name__ == "__main__":
    main()
