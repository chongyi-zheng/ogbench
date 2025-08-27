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
flags.DEFINE_float('reward_scale', 1.0, 'Scalar for the reward function.')


def step(observation, action):
    if observation == 0:
        next_observation = observation + action
    else:
        next_observation = observation
    return next_observation


def get_transition_probs(dtype):
    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_sa = jnp.asarray(transition_probs_sa, dtype=dtype)
    return transition_probs_sa
    

def compute_fb_reprs(policy, transition_probs_sa):
    neg_marg_sa = jnp.ones(
        [FLAGS.num_observations, FLAGS.num_actions],
        dtype=policy.dtype
    ) / (FLAGS.num_observations * FLAGS.num_actions)

    # value iteration
    # Build P_pi: [S, A, S, A] with P_pi[s, a, s', a'] = P(s'|s,a) * pi[a'|s']
    # P_pi = jnp.einsum('ijk,kl->ijkl', transition_probs_sa, policy)  # [S, A, S, A]
    P_pi = transition_probs_sa[..., None] * policy[None, None]  # [S, A, S, A]
    P_flat = P_pi.reshape(FLAGS.num_observations * FLAGS.num_actions,
                          FLAGS.num_observations * FLAGS.num_actions)
    I = jnp.eye(FLAGS.num_observations * FLAGS.num_actions, dtype=policy.dtype)
    # solve (I - discount * T) X = (1-discount) * I for X
    gt_sr_sa_sa = (1.0 - FLAGS.discount) * jnp.linalg.inv(I - FLAGS.discount * P_flat)  # [S * A, S * A]
    # gt_sr_sa_sa = gt_sr_sa_sa_flat.reshape(FLAGS.num_observations, FLAGS.num_actions,
    #                                        FLAGS.num_observations, FLAGS.num_actions)

    gt_ratios = gt_sr_sa_sa / neg_marg_sa.reshape(-1)[None]  # [S * A, S * A]

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


def compute_qs(policy, rewards, transition_probs_sa):
    # value iteration
    # Build P_pi: [S, A, S, A] with P_pi[s, a, s', a'] = P(s'|s,a) * pi[a'|s']
    P_pi = transition_probs_sa[..., None] * policy[None, None]  # [S, A, S, A]
    P_flat = P_pi.reshape(FLAGS.num_observations * FLAGS.num_actions,
                          FLAGS.num_observations * FLAGS.num_actions)
    R = rewards.astype(policy.dtype).reshape(FLAGS.num_observations * FLAGS.num_actions)  # [S * A]
    I = jnp.eye(FLAGS.num_observations * FLAGS.num_actions, dtype=policy.dtype)  # [S * A, S * A]
    # solve (I - discount * T) X = (1-discount) * R for X
    q_flat = (1.0 - FLAGS.discount) * R @ jnp.linalg.inv(I - FLAGS.discount * P_flat)  # [S * A, S * A]
    q = q_flat.reshape(FLAGS.num_observations, FLAGS.num_actions)

    return q


def main(_):
    # we use jnp.float64 to enable accurate SVD.
    jax.config.update('jax_enable_x64', True)

    transition_probs_sa = get_transition_probs(jnp.float64)
    behavioral_policy = jnp.ones([FLAGS.num_observations, FLAGS.num_actions], dtype=jnp.float64) / FLAGS.num_actions
    gt_fb_reprs = compute_fb_reprs(behavioral_policy, transition_probs_sa)

    neg_marg_sa = gt_fb_reprs['neg_marg_sa']
    forward_reprs = gt_fb_reprs['forward_reprs']
    backward_reprs = gt_fb_reprs['backward_reprs']
    backward_reprs_flat = backward_reprs.reshape(
        FLAGS.num_observations * FLAGS.num_actions, FLAGS.repr_dim)

    null_basis = scipy.linalg.null_space(np.asarray(backward_reprs_flat).T).T
    num_null_basis = null_basis.shape[0]

    print("num_null_basis = {}".format(num_null_basis))
    if num_null_basis > 0:
        adversarial_rewards_flat = FLAGS.reward_scale * null_basis
        adversarial_rewards = adversarial_rewards_flat.reshape(
            num_null_basis, FLAGS.num_observations, FLAGS.num_actions)
        reward_latents_flat = jnp.einsum(
            'ij,ki->kj',
            backward_reprs_flat * neg_marg_sa.reshape(-1)[..., None],
            adversarial_rewards_flat,
        )
        reward_latents = reward_latents_flat.reshape(
            num_null_basis, FLAGS.repr_dim
        )
        q_beta_ests = jnp.einsum('ijk,lk->lij', forward_reprs, reward_latents)
        q_beta_gts = jax.vmap(compute_qs, in_axes=(None, 0, None))(
            behavioral_policy, adversarial_rewards, transition_probs_sa)

        avg_abs_err = jnp.mean(jnp.abs(q_beta_gts - q_beta_ests))

        print("avg_abs_err = {}".format(avg_abs_err))
    else:
        pass

    print("end")


if __name__ == "__main__":
    app.run(main)
