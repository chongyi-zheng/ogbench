from collections import defaultdict
import tqdm
import copy
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter1d

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_observations', 4, 'Number of observations.')
flags.DEFINE_integer('num_actions', 4, 'Number of actions.')
flags.DEFINE_integer('initial_obs', 0, 'The initial observation.')
flags.DEFINE_float('discount', 0.9, 'The discount factor.')

flags.DEFINE_integer('repr_dim', 16, 'Forward backward representation dimensions.')
flags.DEFINE_integer('normalized_latent', 0, 'Whether to normalize the latents and backward representations.')
flags.DEFINE_integer('num_eval_latents', 8, 'Number of evaluation latents.')
flags.DEFINE_float('orthonorm_coef', 0.0, 'Orthonormalization regularization coefficient.')
flags.DEFINE_float('null_reward_scale', 1.0, 'Scalar for the null reward function.')


def step(observation, action):
    if observation == 0:
        next_observation = observation + action
    else:
        next_observation = observation
    return next_observation


def compute_gt_fb_reprs(behavioral_policy):
    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_sa = jnp.asarray(transition_probs_sa, dtype=behavioral_policy.dtype)

    neg_marg_sa = jnp.ones(
        [FLAGS.num_observations, FLAGS.num_actions],
        dtype=behavioral_policy.dtype
    ) / (FLAGS.num_observations * FLAGS.num_actions)

    # value iteration
    # Build T_pi: [S, A, S, A] with T_pi[s, a, s', a'] = P(s'|s,a) * pi[a'|s']
    # T_pi = jnp.einsum('ijk,kl->ijkl', transition_probs_sa, policy)  # [S, A, S, A]
    T_pi = transition_probs_sa[..., None] * behavioral_policy[None, None]  # [S, A, S, A]
    T_flat = T_pi.reshape(FLAGS.num_observations * FLAGS.num_actions,
                          FLAGS.num_observations * FLAGS.num_actions)
    I = jnp.eye(FLAGS.num_observations * FLAGS.num_actions, dtype=behavioral_policy.dtype)
    # solve (I - discount * T) X = (1-discount) * I for X
    gt_sr_sa_sa = (1.0 - FLAGS.discount) * jnp.linalg.inv(I - FLAGS.discount * T_flat)  # [S * A, S * A]
    # gt_sr_sa_sa = gt_sr_sa_sa_flat.reshape(FLAGS.num_observations, FLAGS.num_actions,
    #                                        FLAGS.num_observations, FLAGS.num_actions)

    gt_ratios = gt_sr_sa_sa / neg_marg_sa.reshape(-1)[None]  # [S * A, S * A]

    def compute_gt_occ_measure(policy):
        gt_sr_sa_sa = jnp.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions])
        for _ in range(1000):
            gt_sr_sa_sa_new = (
                (1 - FLAGS.discount) * jnp.eye(FLAGS.num_observations * FLAGS.num_actions).reshape(
            FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions)
                + FLAGS.discount * jnp.einsum('ijk,klm->ijlm', transition_probs_sa,
                                              jnp.sum(policy[..., None, None] * gt_sr_sa_sa, axis=1))
            )

            if jnp.max(jnp.abs(gt_sr_sa_sa_new - gt_sr_sa_sa)) < 1e-10:
                gt_sr_sa_sa = gt_sr_sa_sa_new
                break
            gt_sr_sa_sa = gt_sr_sa_sa_new

        return gt_sr_sa_sa

    # gt_sr_sa_sa_tmp = compute_gt_occ_measure(behavioral_policy)
    # gt_sr_sa_sa_tmp = gt_sr_sa_sa_tmp.reshape(FLAGS.num_observations * FLAGS.num_actions,
    #                                           FLAGS.num_observations * FLAGS.num_actions)

    u, s, vt = jnp.linalg.svd(gt_ratios, full_matrices=False)
    assert jnp.allclose(u @ jnp.diag(s) @ vt, gt_ratios)  # only works with float64
    recon_err = jnp.sum(jnp.abs(u @ jnp.diag(s) @ vt - gt_ratios))

    forward_reprs = (u @ jnp.diag(s))[:, :FLAGS.repr_dim]
    backward_reprs = vt[:FLAGS.repr_dim].T

    forward_reprs = forward_reprs.reshape(
        FLAGS.num_observations, FLAGS.num_actions,
        FLAGS.repr_dim
    )
    backward_reprs = backward_reprs.reshape(
        FLAGS.num_observations, FLAGS.num_actions,
        FLAGS.repr_dim
    )

    info = {
        'forward_reprs': forward_reprs,
        'backward_reprs': backward_reprs,
        'gt_sr_sa_sa': gt_sr_sa_sa,
        'neg_marg_sa': neg_marg_sa,
        'recon_err': recon_err,
    }

    return info


def main(_):
    # we use jnp.float64 to enable accurate SVD.
    jax.config.update('jax_enable_x64', True)

    # rng = jax.random.key(np.random.randint(2 ** 32))
    # rng, latent_rng = jax.random.split(rng, 2)
    # latents = jax.random.normal(latent_rng, shape=(FLAGS.num_eval_latents, FLAGS.repr_dim))
    # latents = latents / jnp.linalg.norm(latents, axis=-1, keepdims=True)
    behavioral_policy = jnp.ones([FLAGS.num_observations, FLAGS.num_actions], dtype=jnp.float64) / FLAGS.num_actions
    gt_fb_reprs = compute_gt_fb_reprs(behavioral_policy)

    backward_reprs = gt_fb_reprs['backward_reprs']
    backward_reprs = backward_reprs.reshape(
        FLAGS.num_observations * FLAGS.num_actions, FLAGS.repr_dim)

    backward_reprs = np.asarray(backward_reprs)
    null_basis = scipy.linalg.null_space(backward_reprs.T).T
    num_null_basis = null_basis.shape[0]

    print(num_null_basis)


if __name__ == "__main__":
    app.run(main)
