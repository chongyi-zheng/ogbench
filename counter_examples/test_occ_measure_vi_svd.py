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


def compute_gt_fb_reprs(latents):
    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_sa = jnp.asarray(transition_probs_sa)

    neg_marg_sa = jnp.ones([FLAGS.num_observations, FLAGS.num_actions]) / (FLAGS.num_observations * FLAGS.num_actions)

    def compute_gt_occ_measure(latent):
        gt_sr_sa_sa = jnp.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions])
        for _ in range(1000):
            ratios = gt_sr_sa_sa / neg_marg_sa[None, None]
            ratios = jnp.reshape(ratios, [FLAGS.num_observations * FLAGS.num_actions,
                                          FLAGS.num_observations * FLAGS.num_actions])

            # TODO (chongyi): do we specify the rank here?
            u, s, vh = jnp.linalg.svd(ratios)
            forward_reprs = u * s
            backward_reprs = vh
            forward_latent_inners = jnp.einsum('ik,k->i', forward_reprs, latent)

            forward_latent_inners = jnp.reshape(forward_latent_inners,
                                                [FLAGS.num_observations, FLAGS.num_actions])
            policy = jax.nn.one_hot(forward_latent_inners.argmax(1), FLAGS.num_actions, axis=1)

            gt_sr_sa_sa = (
                (1 - FLAGS.discount) * jnp.eye(FLAGS.num_observations * FLAGS.num_actions).reshape(
            FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions)
                + FLAGS.discount * jnp.einsum('ijk,klm->ijlm', transition_probs_sa,
                                              jnp.sum(policy[..., None, None] * gt_sr_sa_sa, axis=1))
            )

        forward_reprs = jnp.reshape(forward_reprs, [FLAGS.num_observations, FLAGS.num_actions,
                                                    FLAGS.num_observations * FLAGS.num_actions])
        backward_reprs = jnp.reshape(backward_reprs, [FLAGS.num_observations, FLAGS.num_actions,
                                                      FLAGS.num_observations * FLAGS.num_actions])

        return forward_reprs, backward_reprs, gt_sr_sa_sa

    forward_reprs, backward_reprs, gt_sr_sa_sa = jax.vmap(compute_gt_occ_measure)(latents)

    info = {
        'forward_reprs': forward_reprs,
        'backward_reprs': backward_reprs,
        'gt_sr_sa_sa': gt_sr_sa_sa,
    }

    return info


def main(_):
    rng = jax.random.key(np.random.randint(2 ** 32))
    rng, latent_rng = jax.random.split(rng, 2)
    latents = jax.random.normal(latent_rng, shape=(FLAGS.num_eval_latents, FLAGS.repr_dim))
    latents = latents / jnp.linalg.norm(latents, axis=-1, keepdims=True)
    gt_fb_reprs = compute_gt_fb_reprs(latents)

    backward_reprs = gt_fb_reprs['backward_reprs']
    backward_reprs = jnp.reshape(backward_reprs,
                                 [FLAGS.num_observations * FLAGS.num_actions,
                                  FLAGS.num_observations * FLAGS.num_actions])

    backward_reprs = np.asarray(backward_reprs)
    null_basis = scipy.linalg.null_space(backward_reprs.T)
    num_null_basis = null_basis.shape[-1]

    print(num_null_basis)


if __name__ == "__main__":
    app.run(main)
