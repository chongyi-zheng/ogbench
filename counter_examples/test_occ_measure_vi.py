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

flags.DEFINE_integer('num_training_steps', 1_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 1_000, 'Evaluation interval.')
flags.DEFINE_integer('batch_size', 256, 'The batch size.')
flags.DEFINE_integer('hidden_dim', 32, 'Forward backward network hidden dimensions.')
flags.DEFINE_integer('log_linear', 0, 'Whether to use the log linear parameterization for FB networks.')
flags.DEFINE_integer('normalized_latent', 0, 'Whether to normalize the latents and backward representations.')
flags.DEFINE_integer('num_eval_latents', 16, 'Number of evaluation latents.')
flags.DEFINE_float('orthonorm_coef', 0.0, 'Orthonormalization regularization coefficient.')
flags.DEFINE_float('null_reward_scale', 1.0, 'Scalar for the null reward function.')


class FBCritic(nn.Module):
    repr_dim: int = 4
    hidden_dim: int = 32
    log_linear: bool = False
    normalized_backward_reprs: bool = False

    def setup(self):
        self.forward_mlp = nn.Sequential([
            # nn.Dense(self.hidden_dim),
            # nn.gelu,
            # nn.Dense(self.hidden_dim),
            # nn.gelu,
            # nn.Dense(self.repr_dim),
            nn.Dense(self.repr_dim, use_bias=False),
        ])
        self.backward_mlp = nn.Sequential([
            # nn.Dense(self.hidden_dim),
            # nn.gelu,
            # nn.Dense(self.hidden_dim),
            # nn.gelu,
            # nn.Dense(self.repr_dim),
            nn.Dense(self.repr_dim, use_bias=False),
        ])

    @nn.compact
    # def __call__(self, observations, actions, latents, future_observations, future_actions):
    def __call__(self, observations, actions, future_observations, future_actions):
        observations = jax.nn.one_hot(observations, FLAGS.num_observations)
        actions = jax.nn.one_hot(actions, FLAGS.num_actions)
        future_observations = jax.nn.one_hot(future_observations, FLAGS.num_observations)
        future_actions = jax.nn.one_hot(future_actions, FLAGS.num_actions)
        forward_repr_inputs = jnp.concatenate([observations, actions], axis=-1)
        backward_repr_inputs = jnp.concatenate([future_observations, future_actions], axis=-1)

        forward_reprs = self.forward_mlp(forward_repr_inputs)
        backward_reprs = self.backward_mlp(backward_repr_inputs)

        if self.normalized_backward_reprs:
            backward_reprs = backward_reprs / jnp.linalg.norm(
                backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.repr_dim)
        prob_ratios = jnp.einsum('ik,jk->ij', forward_reprs, backward_reprs)

        if self.log_linear:
            prob_ratios = jnp.exp(prob_ratios)

        return prob_ratios

    # @nn.compact
    # def forward_reprs(self, observations, actions, latents):
    #     observations = jax.nn.one_hot(observations, FLAGS.num_observations)
    #     actions = jax.nn.one_hot(actions, FLAGS.num_actions)
    #     forward_repr_inputs = jnp.concatenate([observations, actions, latents], axis=-1)
    #
    #     forward_reprs = self.forward_mlp(forward_repr_inputs)
    #
    #     return forward_reprs
    @nn.compact
    def forward_reprs(self, observations, actions):
        observations = jax.nn.one_hot(observations, FLAGS.num_observations)
        actions = jax.nn.one_hot(actions, FLAGS.num_actions)
        forward_repr_inputs = jnp.concatenate([observations, actions], axis=-1)

        forward_reprs = self.forward_mlp(forward_repr_inputs)

        return forward_reprs

    @nn.compact
    def backward_reprs(self, future_observations, future_actions):
        future_observations = jax.nn.one_hot(future_observations, FLAGS.num_observations)
        future_actions = jax.nn.one_hot(future_actions, FLAGS.num_actions)
        backward_repr_inputs = jnp.concatenate([future_observations, future_actions], axis=-1)

        backward_reprs = self.backward_mlp(backward_repr_inputs)
        if self.normalized_backward_reprs:
            backward_reprs = backward_reprs / jnp.linalg.norm(
                backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.repr_dim)

        return backward_reprs


def step(observation, action):
    if observation == 0:
        next_observation = observation + action
    else:
        next_observation = observation
    return next_observation


def compute_gt_fb_reprs(latents, repr_dim=4):
    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_sa = jnp.asarray(transition_probs_sa)

    @jax.jit
    def get_fb_reprs(latent):
        rng = jax.random.key(np.random.randint(2 ** 32))
        rng, init_rng = jax.random.split(rng, 2)
        fb_critic = FBCritic(repr_dim=repr_dim, hidden_dim=FLAGS.hidden_dim,
                             log_linear=FLAGS.log_linear, normalized_backward_reprs=FLAGS.normalized_latent)
        n_observations = jnp.repeat(
            jnp.arange(FLAGS.num_observations)[:, None],
            FLAGS.num_actions,
            axis=1,
        ).reshape(-1)
        n_actions = jnp.repeat(
            jnp.arange(FLAGS.num_actions)[None, :],
            FLAGS.num_observations,
            axis=0,
        ).reshape(-1)
        params = fb_critic.init(
            init_rng,
            n_observations, n_actions,
            n_observations, n_actions,
        )

        @jax.jit
        def loss_fn(params):
            forward_reprs = fb_critic.apply(params, n_observations, n_actions, method='forward_reprs')
            backward_reprs = fb_critic.backward_reprs(n_observations, n_actions, method='forward_reprs')

            forward_latent_inners = jnp.einsum('ik,k->i', forward_reprs, latent)
            forward_latent_inners = forward_latent_inners.reshape(
                FLAGS.num_observations, FLAGS.num_actions)
            policy = jax.nn.one_hot(forward_latent_inners.argmax(axis=1), FLAGS.num_actions, axis=1)

            gt_sr_sa_sa = jnp.einsum('ijk,lmk->ijlm', forward_reprs, backward_reprs)
            gt_sr_sa_sa_targets = (
                    (1 - FLAGS.discount) * jnp.eye(FLAGS.num_observations * FLAGS.num_actions).reshape(
                FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions)
                    + FLAGS.discount * jnp.einsum('ijk,klm->ijlm', transition_probs_sa,
                                                  jnp.sum(policy[..., None, None] * gt_sr_sa_sa, axis=1))
            )

            loss = jnp.mean((gt_sr_sa_sa - gt_sr_sa_sa_targets) ** 2)

            return loss, {
                'loss': loss,
            }

        optimizer = optax.adam(learning_rate=3e-4)
        opt_state = optimizer.init(params)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def update_fn(params, opt_state):
            (_, info), grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, info

        for _ in range(FLAGS.num_training_steps):
            params, opt_state, info = update_fn(
                params, opt_state)

        return params

    neg_marg_sa = jnp.ones([FLAGS.num_observations * FLAGS.num_actions]) / (FLAGS.num_observations * FLAGS.num_actions)
    gt_ratios = gt_sr_sa_sa / neg_marg_sa[None, None]

    return gt_ratios


def main():
    pass


if __name__ == "__main__":
    main()
