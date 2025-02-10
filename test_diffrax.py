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

def compute_log_likelihood(goals, observations, actions, rng, return_ratio=False,
                           div_type='hutchinson_normal', num_flow_steps=10,
                           num_hutchinson_ests=32):
    """
    Euler method
    """

    noisy_goals = goals
    div_int = jnp.zeros(goals.shape[:-1])
    step_size = 1.0 / num_flow_steps

    # Define the body function to be scanned
    def body_fn(carry, i):
        noisy_goals, div_int, z = carry

        # Time for this iteration
        # times = 1.0 - jnp.full(noisy_goals.shape[:-1], i * step_size)
        times = 1.0 - i * step_size

        if div_type == 'exact':
            def compute_exact_div(noisy_goals, times, observations, actions):
                # def vf_func(noisy_goal, time, observation, action):
                #     noisy_goal = jnp.expand_dims(noisy_goal, 0)
                #     time = jnp.expand_dims(time, 0)
                #     observation = jnp.expand_dims(observation, 0)
                #     if action is not None:
                #         action = jnp.expand_dims(action, 0)
                #     vf = self.network.select('critic_vf')(
                #         noisy_goal, time, observation, action).squeeze(0)
                #
                #     return vf

                def vf_func(t, g, obs, a):
                    noisy_goal = jnp.expand_dims(g, 0)
                    time = jnp.expand_dims(t, 0)
                    observation = jnp.expand_dims(obs, 0)
                    if a is not None:
                        action = jnp.expand_dims(a, 0)
                    else:
                        action = a

                    vf = vector_field(time, noisy_goal, (observation, action)).squeeze(0)

                    return vf

                # def div_func(noisy_goal, time, observation, action):
                #     jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
                #
                #     # return jnp.trace(jac, axis1=-2, axis2=-1)
                #     return jac

                # vf = self.network.select('critic_vf')(
                #     noisy_goals, times, observations, actions)
                vf = vector_field(times, noisy_goals, (observations, actions))

                if actions is not None:
                    # jac = jax.vmap(jax.jacrev(vf_func), in_axes=(0, 0, 0, 0), out_axes=0)(
                    #     noisy_goals, times, observations, actions)
                    jac = jax.vmap(
                        jax.jacrev(vf_func, argnums=1),
                        in_axes=(None, 0, 0, 0), out_axes=0
                    )(times, noisy_goals, observations, actions)
                else:
                    jac = jax.vmap(
                        jax.jacrev(vf_func, argnums=1),
                        in_axes=(None, 0, 0, None), out_axes=0
                    )(times, noisy_goals, observations, actions)

                div = jnp.trace(jac, axis1=-2, axis2=-1)

                return vf, div

            vf, div = compute_exact_div(noisy_goals, times, observations, actions)
        else:
            def compute_hutchinson_div(noisy_goals, times, observations, actions, z):
                # Define vf_func for jvp
                def vf_func(goals):
                    vf = vector_field(times, goals, (observations, actions))

                    return vf

                # Split RNG and sample noise
                # z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)

                # Forward (vf) and linearization (jac_vf_dot_z)
                # vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z,))

                def single_jvp(z):
                    vf, jac_vf_dot_z = jax.jvp(
                        lambda n: vector_field(times, n, (observations, actions)),
                        (noisy_goals,), (z,)
                    )

                    return vf, jac_vf_dot_z

                # vf, jac_vf_dot_z = jax.jvp(
                #     lambda n: vector_field(times, n, (observations, actions)),
                #     (noises,), (z,)
                # )
                vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
                div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)

                # Hutchinson's trace estimator
                # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
                # div = jnp.einsum("ij,ij->i", jac_vf_dot_z, z)
                vf = vf[..., 0]
                div = div.mean(axis=-1)

                return vf, div

            # rng, div_rng = jax.random.split(rng)
            vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, z)

        # Update goals and divergence integral. We need to consider Q ensemble here.
        # new_noisy_goals = jnp.min(noisy_goals[None] - vf * step_size, axis=0)
        # new_div_int = jnp.min(div_int[None] - div * step_size, axis=0)
        new_noisy_goals = noisy_goals - vf * step_size
        new_div_int = div_int - div * step_size

        # Return updated carry and scan output
        return (new_noisy_goals, new_div_int, z), None

    # Use lax.scan to iterate over num_flow_steps
    rng, div_rng = jax.random.split(rng)
    z = jax.random.normal(div_rng, shape=(*goals.shape, num_hutchinson_ests), dtype=goals.dtype)
    (noisy_goals, div_int, _), _ = jax.lax.scan(
        body_fn, (noisy_goals, div_int, z), jnp.arange(num_flow_steps))

    if return_ratio:
        log_ratio = div_int
        return log_ratio
    else:
        # Finally, compute log_prob using the final noisy_goals and div_int
        gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
        log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

        return log_prob


def diffrax_compute_log_likelihood(goals, observations, actions, rng, return_ratio=False,
                                   div_type='hutchinson_normal', num_flow_steps=10,
                                   num_hutchinson_ests=32):

    if div_type == 'exact':
        def vf_func(times, noise_div_int, carry):
            noises, _ = noise_div_int
            observations, actions, _ = carry
            def single_vf(t, g, obs, a):
                noisy_goal = jnp.expand_dims(g, 0)
                time = jnp.expand_dims(t, 0)
                observation = jnp.expand_dims(obs, 0)
                if a is not None:
                    action = jnp.expand_dims(a, 0)
                else:
                    action = a

                vf = vector_field(time, noisy_goal, (observation, action)).squeeze(0)

                return vf

            vf = vector_field(times, noises, (observations, actions))

            if actions is not None:
                jac = jax.vmap(
                    jax.jacrev(single_vf, argnums=1),
                    in_axes=(None, 0, 0, 0), out_axes=0
                )(times, noises, observations, actions)
            else:
                jac = jax.vmap(
                    jax.jacrev(vf_func, argnums=1),
                    in_axes=(None, 0, 0, None), out_axes=0
                )(times, noises, observations, actions)

            div = jnp.trace(jac, axis1=-2, axis2=-1)

            return vf, div

    else:
        def vf_func(times, noise_div_int, carry):
            noises, _ = noise_div_int
            observations, actions, z = carry

            # Split RNG and sample noise
            # rng, div_rng = jax.random.split(rng)
            # z = jax.random.normal(div_rng, shape=noises.shape, dtype=noises.dtype)

            # Forward (vf) and linearization (jac_vf_dot_z)
            def single_jvp(z):
                vf, jac_vf_dot_z = jax.jvp(
                    lambda n: vector_field(times, n, (observations, actions)),
                    (noises,), (z,)
                )

                return vf, jac_vf_dot_z

            # vf, jac_vf_dot_z = jax.jvp(
            #     lambda n: vector_field(times, n, (observations, actions)),
            #     (noises,), (z,)
            # )
            vf, jac_vf_dot_z = jax.vmap(single_jvp, in_axes=-1, out_axes=-1)(z)
            div = jnp.einsum("ijl,ijl->il", jac_vf_dot_z, z)

            vf = vf[..., 0]
            div = div.mean(axis=-1)

            return vf, div

    term = ODETerm(vf_func)
    # solver = Dopri5()
    solver = Euler()
    rng, div_rng = jax.random.split(rng)
    z = jax.random.normal(div_rng, shape=(*goals.shape, num_hutchinson_ests), dtype=goals.dtype)
    solution = diffeqsolve(
        term, solver,
        t0=1.0, t1=0.0, dt0=-1 / num_flow_steps,
        y0=(goals, jnp.zeros(goals.shape[:-1])),
        args=(observations, actions, z)
    )
    noises, div_int = jax.tree.map(
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
        goals, observations, actions, diffrax_log_p_key, div_type='hutchinson_normal', return_ratio=True)
    diffrax_log_p = diffrax_compute_log_likelihood(
        goals, observations, actions, diffrax_log_p_key, div_type='hutchinson_normal', return_ratio=True)

    assert jnp.allclose(log_p, diffrax_log_p)

    print("same!")


if __name__ == "__main__":
    main()
