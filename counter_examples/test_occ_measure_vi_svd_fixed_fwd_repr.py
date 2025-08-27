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


def compute_gt_fb_reprs(latents, rng):
    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_sa = jnp.asarray(transition_probs_sa)

    neg_marg_sa = jnp.ones([FLAGS.num_observations, FLAGS.num_actions]) / (FLAGS.num_observations * FLAGS.num_actions)

    # sample the fixed forward representation for each latents
    rng, forward_repr_rng = jax.random.split(rng)
    forward_reprs = jax.random.normal(
        forward_repr_rng,
        (FLAGS.num_eval_latents, FLAGS.num_observations, FLAGS.num_actions, FLAGS.repr_dim)
    )
    forward_reprs = jax.nn.softplus(forward_reprs)
    forward_latent_inners = jnp.einsum('ijkl,il->ijk', forward_reprs, latents)

    policies = jax.nn.one_hot(forward_latent_inners.argmax(2), FLAGS.num_actions, axis=2)

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

    def compute_gt_occ_measure_solve(policy):
        S, A = policy.shape
        # Build T_pi: [S, A, S, A] with T_pi[s, a, s', a'] = P(s'|s,a) * pi[a'|s']
        # T_pi = jnp.einsum('ijk,kl->ijkl', transition_probs_sa, policy)  # [S,A,S,A]
        T_pi = transition_probs_sa[..., None] * policy[None, None]  # [S,A,S,A]
        M = S * A
        T_flat = T_pi.reshape(M, M)
        I = jnp.eye(M, dtype=policy.dtype)
        # Solve (I - discount*T) X = (1-discount) * I  for X
        # SR_flat = jnp.linalg.solve(I - FLAGS.discount * T_flat, (1.0 - FLAGS.discount) * I)
        sr_flat = (1.0 - FLAGS.discount) * jnp.linalg.inv(I - FLAGS.discount * T_flat)

        return sr_flat.reshape(S, A, S, A)

    # gt_srs_sa_sa = compute_gt_occ_measure(policies[0])
    # gt_srs_sa_sa = jax.vmap(compute_gt_occ_measure)(policies)
    gt_srs_sa_sa = jax.vmap(compute_gt_occ_measure_solve)(policies)
    gt_ratios = gt_srs_sa_sa / neg_marg_sa[None, None, None]

    # TODO (chongyiz):
    # u, s, vt = jnp.linalg.svd(gt_ratios, full_matrices=False)
    flatten_forward_reprs = forward_reprs.reshape(
        FLAGS.num_eval_latents * FLAGS.num_observations * FLAGS.num_actions,
        FLAGS.repr_dim
    )
    gt_ratios = gt_ratios.reshape(
        FLAGS.num_eval_latents * FLAGS.num_observations * FLAGS.num_actions,
        FLAGS.num_observations * FLAGS.num_actions
    )
    flatten_backward_reprs_temp = jnp.linalg.lstsq(flatten_forward_reprs, gt_ratios, rcond=None)[0]
    flatten_backward_reprs = jnp.linalg.inv(flatten_forward_reprs.T @ flatten_forward_reprs) @ flatten_forward_reprs.T @ gt_ratios
    backward_reprs = flatten_backward_reprs.reshape(
        FLAGS.num_observations, FLAGS.num_actions,
        FLAGS.num_observations, FLAGS.num_actions,
    )
    fb_error = jnp.mean(jnp.abs(flatten_forward_reprs @ flatten_backward_reprs - gt_ratios))

    info = {
        'forward_reprs': forward_reprs,
        'backward_reprs': backward_reprs,
        'gt_srs_sa_sa': gt_srs_sa_sa,
        'neg_marg_sa': neg_marg_sa,
        'fb_error': fb_error,
    }

    return info


def main(_):
    rng = jax.random.key(np.random.randint(2 ** 32))
    rng, latent_rng, gt_fb_reprs_rng = jax.random.split(rng, 3)
    latents = jax.random.normal(latent_rng, shape=(FLAGS.num_eval_latents, FLAGS.repr_dim))
    latents = latents / jnp.linalg.norm(latents, axis=-1, keepdims=True)
    gt_fb_reprs = compute_gt_fb_reprs(latents, gt_fb_reprs_rng)

    backward_reprs = gt_fb_reprs['backward_reprs']
    backward_reprs = backward_reprs.reshape(
        FLAGS.num_observations * FLAGS.num_actions,
        FLAGS.num_observations * FLAGS.num_actions
    )

    backward_reprs = np.asarray(backward_reprs)
    null_basis = scipy.linalg.null_space(backward_reprs.T)
    num_null_basis = null_basis.shape[-1]

    print(num_null_basis)


if __name__ == "__main__":
    app.run(main)
