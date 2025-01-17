import time
import numpy as np

import jax
import jax.numpy as jnp


def main():
    num_ensembles = 2
    data_dims = 2
    batch_size = 512

    key = jax.random.PRNGKey(np.random.randint(2 ** 32 - 1))
    key, jac_key = jax.random.split(key)
    J = jax.random.normal(jac_key, (num_ensembles, data_dims, data_dims))
    # J = jnp.stack([jnp.diag(jnp.array([1.0, 2.0, 3.0])), jnp.diag(jnp.array([-1.0, -2.0, -3.0]))], axis=0)

    def vf_func(x):
        out = jnp.einsum('eij,kj->eki', J, x)

        return out

    def vf_batch_sum_func(goals, ensemble_idx, dim_idx):
        # vf = self.network.select(module_name + '_vf')(
        #     goals, times, observations, actions=actions)
        # vf = vf[ensemble_idx, :, dim_idx]
        vf = vf_func(goals)
        vf = vf[ensemble_idx, :, dim_idx]

        # Sum over the batch: sum_{n = 1}^N vf_{e,i}(x_n), where e is the ensemble_idx and i is the dim_idx.
        vf_sum = jnp.sum(vf)
        return vf_sum

    def compute_div(goals):
        # [∇_x vf_{e, i}(x_n)]_d, shape = (N, )
        derivative_func = lambda e, d: jax.grad(
            vf_batch_sum_func)(goals, e, d)[:, d]

        # div_func = jax.vmap(derivative_func, in_axes=(None, ), )
        # [∇_x vf_{e, 1}(x_n), ∇_x vf_{e, 2}(x_n), ..., ∇_x vf_{e, D}(x_n)], shape = (N, D)
        derivative_vec_func = lambda e: jax.vmap(
            derivative_func, in_axes=(None, 0), out_axes=1)(e, jnp.arange(goals.shape[-1]))

        derivatives = jax.vmap(derivative_vec_func)(jnp.arange(num_ensembles))  # (E, N, D)
        div = derivatives.sum(axis=-1)

        return div

    key, g_key = jax.random.split(key)
    goals = jax.random.normal(key, (batch_size, data_dims))
    start_time = time.time()
    div = compute_div(goals)
    jax.block_until_ready(div)
    end_time = time.time()
    print("div time = {}".format(end_time - start_time))

    # def new_vf_func(x, ensemble_idx):
    #     # option 1
    #     out = jnp.einsum('eij,j->ei', J, x)
    #     # out = out.reshape(-1)
    #
    #     return out[ensemble_idx]

    def new_vf_func(x):
        # option 2
        out = jnp.einsum('eij,j->ei', J, x)
        out = out.reshape(-1)

        return out

    def div_func(x):
        # jac = jax.jacrev(
        #     lambda x: new_vf_func(x, ensemble_idx)
        # )(x)

        def f(x):
            # option 1
            # jac = jax.vmap(
            #     jax.jacrev(new_vf_func),
            #     in_axes=(None, 0), out_axes=0
            # )(x, jnp.arange(num_ensembles))

            # option 2
            jac = jax.jacrev(new_vf_func)(x)
            jac = jac.reshape([num_ensembles, data_dims, data_dims])

            return jnp.trace(jac, axis1=-2, axis2=-1)

        div = jax.vmap(f, in_axes=0, out_axes=1)(x)

        return div

    # If x is a batch of samples, we vmap over them:
    # div_func = lambda e: jax.vmap(div, in_axes=(0, None))(goals, e)
    start_time = time.time()
    # div_jac = jax.vmap(div)(jnp.arange(num_ensembles))

    # out = new_vf_func(goals[0])
    # tmp_out = jnp.einsum('eij,j->ei', J, goals[0])
    # assert jnp.all(tmp_out[0] == out[:data_dims])
    # assert jnp.all(tmp_out[1] == out[data_dims:])

    div_jac = div_func(goals)
    jax.block_until_ready(div_jac)
    end_time = time.time()
    print("div_jac time = {}".format(end_time - start_time))

    div_gt = jax.vmap(jnp.diag, in_axes=0, out_axes=0)(J).sum(axis=-1)

    assert jnp.allclose(div_jac, div_gt[:, None])
    assert jnp.allclose(div, div_gt[:, None])

    print()


if __name__ == "__main__":
    main()
