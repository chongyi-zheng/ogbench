from collections import Counter, defaultdict
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

flags.DEFINE_integer('dataset_size', 1_000, 'Size of the dataset.')
flags.DEFINE_integer('num_trajectories',  100, 'The number of trajectories to collect.')
flags.DEFINE_integer('max_episode_length', 11, 'The maximum length of an episode.')

flags.DEFINE_integer('num_training_steps', 10_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 1_000, 'Evaluation interval.')
flags.DEFINE_float('tau', 0.005, 'Target network update rate.')
flags.DEFINE_integer('batch_size', 256, 'The batch size.')
flags.DEFINE_integer('repr_dim', 4, 'Forward backward representation dimensions.')
flags.DEFINE_integer('hidden_dim', 32, 'Forward backward network hidden dimensions.')
flags.DEFINE_integer('log_linear', 0, 'Whether to use the log linear parameterization for FB networks.')
flags.DEFINE_integer('normalized_latent', 0, 'Whether to normalize the latents and backward representations.')
flags.DEFINE_float('orthonorm_coef', 0.0, 'Orthonormalization regularization coefficient.')


class FBCritic(nn.Module):
    repr_dim: int = 4
    hidden_dim: int = 32
    log_linear: bool = False
    normalized_backward_reprs: bool = False

    def setup(self):
        self.forward_mlp = nn.Sequential([
            nn.Dense(self.repr_dim, use_bias=False),
        ])
        self.backward_mlp = nn.Sequential([
            nn.Dense(self.repr_dim, use_bias=False),
        ])

    @nn.compact
    def __call__(self, observations, actions, future_observations, future_actions):
        # observations = jax.nn.one_hot(observations, FLAGS.num_observations)
        # actions = jax.nn.one_hot(actions, FLAGS.num_actions)
        # future_observations = jax.nn.one_hot(future_observations, FLAGS.num_observations)
        # future_actions = jax.nn.one_hot(future_actions, FLAGS.num_actions)
        forward_repr_input_indices = jnp.ravel_multi_index(
            jnp.stack([observations, actions], axis=0),
            (FLAGS.num_observations, FLAGS.num_actions),
            mode='clip'  # JIT compatible
        )
        forward_repr_inputs = jax.nn.one_hot(
            forward_repr_input_indices,
            FLAGS.num_observations * FLAGS.num_actions,
            axis=-1
        )
        backward_repr_input_indices = jnp.ravel_multi_index(
            jnp.stack([future_observations, future_actions], axis=0),
            (FLAGS.num_observations, FLAGS.num_actions),
            mode='clip'  # JIT compatible
        )
        backward_repr_inputs = jax.nn.one_hot(
            backward_repr_input_indices,
            FLAGS.num_observations * FLAGS.num_actions,
            axis=-1
        )

        forward_reprs = self.forward_mlp(forward_repr_inputs)
        backward_reprs = self.backward_mlp(backward_repr_inputs)

        if self.normalized_backward_reprs:
            backward_reprs = backward_reprs / jnp.linalg.norm(
                backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.repr_dim)
        prob_ratios = jnp.einsum('ik,jk->ij', forward_reprs, backward_reprs)

        if self.log_linear:
            prob_ratios = jnp.exp(prob_ratios)

        return prob_ratios

    @nn.compact
    def forward_reprs(self, observations, actions):
        # observations = jax.nn.one_hot(observations, FLAGS.num_observations)
        # actions = jax.nn.one_hot(actions, FLAGS.num_actions)
        # forward_repr_inputs = jnp.concatenate([observations, actions, latents], axis=-1)
        forward_repr_input_indices = jnp.ravel_multi_index(
            jnp.stack([observations, actions], axis=0),
            (FLAGS.num_observations, FLAGS.num_actions),
            mode='clip'  # JIT compatible
        )
        forward_repr_inputs = jax.nn.one_hot(
            forward_repr_input_indices,
            FLAGS.num_observations * FLAGS.num_actions,
            axis=-1
        )

        forward_reprs = self.forward_mlp(forward_repr_inputs)

        return forward_reprs

    @nn.compact
    def backward_reprs(self, future_observations, future_actions):
        # future_observations = jax.nn.one_hot(future_observations, FLAGS.num_observations)
        # future_actions = jax.nn.one_hot(future_actions, FLAGS.num_actions)
        # backward_repr_inputs = jnp.concatenate([future_observations, future_actions], axis=-1)
        backward_repr_input_indices = jnp.ravel_multi_index(
            jnp.stack([future_observations, future_actions], axis=0),
            (FLAGS.num_observations, FLAGS.num_actions),
            mode='clip'  # JIT compatible
        )
        backward_repr_inputs = jax.nn.one_hot(
            backward_repr_input_indices,
            FLAGS.num_observations * FLAGS.num_actions,
            axis=-1
        )

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


def collect_dataset(behavioral_policy):
    assert FLAGS.num_trajectories * (FLAGS.max_episode_length - 1) == FLAGS.dataset_size

    traj_dataset = defaultdict(list)
    for traj_idx in range(FLAGS.num_trajectories):
        # random sample initial observations
        obs = FLAGS.initial_obs
        for t in range(FLAGS.max_episode_length):
            action = np.random.choice(FLAGS.num_actions, p=behavioral_policy[obs])

            next_obs = step(obs, action)

            traj_dataset['observations'].append(obs)
            traj_dataset['actions'].append(action)
            traj_dataset['terminals'].append((t == FLAGS.max_episode_length - 1))

            obs = next_obs

    for k, v in traj_dataset.items():
        if k in ['observations', 'actions']:
            traj_dataset[k] = np.array(v, dtype=np.int32)
        else:
            traj_dataset[k] = np.array(v, dtype=np.float32)

    masks = (traj_dataset['terminals'] == 0)
    next_masks = np.concatenate([[False], masks[:-1]])
    new_terminals = np.concatenate([traj_dataset['terminals'][1:], [1.0]])

    dataset = dict(
        observations=traj_dataset['observations'][masks],
        actions=traj_dataset['actions'][masks],
        next_observations=traj_dataset['observations'][next_masks],
        next_actions=traj_dataset['actions'][next_masks],
        terminals=new_terminals[masks],
    )

    for k, v in dataset.items():
        assert len(v) == FLAGS.dataset_size

    (terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
    dataset.update(dict(
        terminal_locs=terminal_locs,
    ))

    return dataset


def compute_successor_reprs(dataset, policy):
    # beta = np.ones([num_observations, num_actions])
    # beta /= num_actions
    assert policy.shape == (FLAGS.num_observations, FLAGS.num_actions)

    transition_probs_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    for obs in range(FLAGS.num_observations):
        for action in range(FLAGS.num_actions):
            next_obs = step(obs, action)
            transition_probs_sa[obs, action, next_obs] += 1
    transition_probs_sa = transition_probs_sa / np.sum(transition_probs_sa, axis=-1, keepdims=True)
    transition_probs_s = np.sum(transition_probs_sa * policy[..., None], axis=1)
    # transition_probs_sa = jnp.array(transition_probs_sa)
    # transition_probs_s = jnp.array(transition_probs_s)

    # ground truth successor representations
    gt_sr_sa_s = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations])
    gt_sr_s_s = np.zeros([FLAGS.num_observations, FLAGS.num_observations])
    for _ in tqdm.trange(1000):
        gt_sr_s_s = (
            (1 - FLAGS.discount) * np.eye(FLAGS.num_observations)
            + FLAGS.discount * np.einsum(
                'ij,jk->ik',
                transition_probs_s,
                gt_sr_s_s
            )
        )
        gt_sr_sa_s = (
            (1 - FLAGS.discount) * np.eye(FLAGS.num_observations)[:, None]
            + FLAGS.discount * np.einsum(
                'ijk,kl->ijl',
                transition_probs_sa,
                np.sum(policy[..., None] * gt_sr_sa_s, axis=1)
            )
        )
    # gt_sr_s_sa = gt_sr_s_s[..., None] * policy[None]
    # gt_sr_sa_sa_tmp = gt_sr_sa_s[..., None] * policy[None, None]

    gt_sr_sa_sa = np.zeros([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions])
    for _ in tqdm.trange(1000):
        gt_sr_sa_sa = (
            (1 - FLAGS.discount) * np.eye(FLAGS.num_observations * FLAGS.num_actions).reshape([FLAGS.num_observations, FLAGS.num_actions, FLAGS.num_observations, FLAGS.num_actions])
            + FLAGS.discount * np.einsum(
                'ijk,klm->ijlm',
                transition_probs_sa,
                np.sum(policy[..., None, None] * gt_sr_sa_sa, axis=1)
            )
        )

    # empirical state marginal
    # counts = Counter(dataset['observations'])
    # emp_marg_s_beta = np.array([counts[obs] for obs in range(FLAGS.num_observations)])
    # emp_marg_s_beta = emp_marg_s_beta / np.sum(emp_marg_s_beta)
    # marginal_s_beta = jnp.array(marginal_s_beta)
    emp_marg_s_beta = np.zeros(FLAGS.num_observations)
    for s in range(FLAGS.num_observations):
        emp_marg_s_beta[s] = np.sum(dataset['observations'] == s)
    emp_marg_s_beta /= np.sum(emp_marg_s_beta, keepdims=True)

    emp_marg_sa_beta = np.zeros([FLAGS.num_observations, FLAGS.num_actions])
    for s in range(FLAGS.num_observations):
        for a in range(FLAGS.num_actions):
            emp_marg_sa_beta[s, a] = np.sum(np.logical_and(dataset['observations'] == s, dataset['actions'] == a))
    emp_marg_sa_beta /= np.sum(emp_marg_sa_beta, keepdims=True)

    srs = dict(gt_sr_sa_s=gt_sr_sa_s, gt_sr_s_s=gt_sr_s_s,
               gt_sr_s_sa=gt_sr_s_sa, gt_sr_sa_sa=gt_sr_sa_sa,
               emp_marg_s_beta=emp_marg_s_beta, emp_marg_sa_beta=emp_marg_sa_beta)

    return srs


def compute_null_basis(matrix, atol=0.0, rtol=None):
    num_rows, num_cols = matrix.shape
    u, s, v_transpose = jnp.linalg.svd(matrix, full_matrices=True)
    if rtol is None:
        # heuristic like NumPy: eps * max(m, n)
        rtol = jnp.finfo(matrix.dtype).eps * max(num_rows, num_cols)
    tol = jnp.maximum(atol, rtol * s[0])
    rank = jnp.sum(s > tol)  # columns of V corresponding to zero singular values

    return v_transpose[rank:].T  # shape (n, n - rank)


def sample_batch(dataset, batch_size):
    idxs = np.random.randint(FLAGS.dataset_size, size=(batch_size,))
    rand_idxs = np.random.randint(FLAGS.dataset_size, size=(batch_size,))

    offsets = np.random.geometric(p=1 - FLAGS.discount, size=(batch_size,)) - 1  # in [0, inf)
    final_state_idxs = dataset['terminal_locs'][np.searchsorted(dataset['terminal_locs'], idxs)]
    future_idxs = np.minimum(idxs + offsets, final_state_idxs)

    batch = dict(
        observations=dataset['observations'][idxs],
        actions=dataset['actions'][idxs],
        next_observations=dataset['next_observations'][idxs],
        next_actions=dataset['next_actions'][idxs],
        future_observations=dataset['observations'][future_idxs],
        future_actions=dataset['actions'][future_idxs],
        random_observations=dataset['observations'][rand_idxs],
        random_actions=dataset['actions'][rand_idxs],
    )

    return batch


def plot_metrics(metrics, f_axes=None, logyscale_stats=[], title='', label=None):
    # learning curves
    if f_axes is None:
        nrows = np.ceil(len(metrics) / 4).astype(int)
        ncols = 4
        f, axes = plt.subplots(nrows=nrows, ncols=ncols)
        if nrows == 1:
            axes = np.array([axes])
        f.set_figheight(3 * nrows)
        f.set_figwidth(3 * ncols)
    else:
        f, axes = f_axes

    for idx, (name, val) in enumerate(metrics.items()):
        v = np.array(val)
        if len(v) == 0:
            continue

        x, y = v[:, 0], v[:, 1]
        ax = axes[idx // 4, idx % 4]

        if 'train' in name:
            y = gaussian_filter1d(y, 100)
        ax.plot(x, y, label=label)
        if name in logyscale_stats:
            ax.set_yscale('log')
        ax.set_title(name)

        # ax.legend()
        ax.grid()

    f.suptitle(title)

    return f, axes


def train_and_eval(dataset, num_training_steps=10_000, eval_interval=1_000):
    rng = jax.random.key(np.random.randint(2 ** 32))
    rng, init_rng = jax.random.split(rng, 2)

    fb_critic = FBCritic(repr_dim=FLAGS.repr_dim, hidden_dim=FLAGS.hidden_dim,
                         log_linear=FLAGS.log_linear, normalized_backward_reprs=FLAGS.normalized_latent)
    example_batch = sample_batch(dataset, 2)
    params = fb_critic.init(
        init_rng,
        example_batch['observations'], example_batch['actions'],
        example_batch['future_observations'], example_batch['future_actions']
    )

    behavioral_policy = np.ones([FLAGS.num_observations, FLAGS.num_actions]) / FLAGS.num_actions
    srs = compute_successor_reprs(dataset, behavioral_policy)
    gt_ratios = jnp.asarray(srs['gt_sr_sa_sa'] / srs['emp_marg_sa_beta'][None, None])

    @jax.jit
    def fb_loss_fn(params, batch, rng):
        observations = batch['observations']
        actions = batch['actions']
        random_observations = batch['random_observations']
        random_actions = batch['random_actions']
        ratio_labels = gt_ratios[observations, actions][:, random_observations, random_actions]

        random_prob_ratios = fb_critic.apply(
            params, observations, actions,
            random_observations, random_actions
        )

        loss = jnp.mean((random_prob_ratios - ratio_labels) ** 2)

        info = dict(
            loss=loss,
            random_prob_ratios=jnp.mean(jnp.diag(random_prob_ratios)),
        )

        return loss, info

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(fb_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(params, opt_state, batch, rng):
        rng, loss_rng = jax.random.split(rng, 2)

        (_, info), grad = grad_fn(params, batch, loss_rng)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, rng, info

    def evaluate_fn(params):
        observations = jnp.repeat(
            jnp.expand_dims(jnp.arange(FLAGS.num_observations), axis=1),
            FLAGS.num_actions,
            axis=1,
        ).reshape(-1)
        actions = jnp.repeat(
            jnp.expand_dims(jnp.arange(FLAGS.num_actions), axis=0),
            FLAGS.num_observations,
            axis=0
        ).reshape(-1)
        prob_ratios = fb_critic.apply(params, observations, actions, observations, actions)

        backward_reprs = fb_critic.apply(params, observations, actions, method='backward_reprs')
        backward_repr_norms = jnp.linalg.norm(backward_reprs, axis=-1, keepdims=True)
        backward_reprs = np.asarray(backward_reprs)
        backward_repr_norms = np.asarray(backward_repr_norms)

        behavioral_policy = np.ones([FLAGS.num_observations, FLAGS.num_actions]) / FLAGS.num_actions
        srs = compute_successor_reprs(dataset, behavioral_policy)
        gt_ratios = srs['gt_sr_sa_sa'] / srs['emp_marg_sa_beta'][None, None]
        prob_ratios = prob_ratios.reshape(FLAGS.num_observations, FLAGS.num_actions,
                                          FLAGS.num_observations, FLAGS.num_actions)
        prob_ratios = np.asarray(prob_ratios)

        prob_ratio_error = np.nanmean(np.abs(prob_ratios - gt_ratios))

        # (chongyi): The matrix used to compute the nullspace is (repr_dim, |S||A|)
        # null_basis = compute_null_basis(backward_reprs)
        null_basis = scipy.linalg.null_space(backward_reprs.T)
        num_null_basis = null_basis.shape[-1]

        info = dict(
            prob_ratio_error=prob_ratio_error,
            num_null_basis=num_null_basis,
            backward_repr_norms=np.mean(backward_repr_norms),
        )

        return info

    metrics = defaultdict(list)
    for step in tqdm.trange(num_training_steps):
        rng, update_rng = jax.random.split(rng)
        batch = sample_batch(dataset, FLAGS.batch_size)
        params, opt_state, rng, info = update_fn(
            params, opt_state, batch, update_rng)

        for k, v in info.items():
            metrics['train/' + k].append(
                np.array([step, v])
            )

        if step == 1 or step % eval_interval == 0:
            eval_info = evaluate_fn(params)
            for k, v in eval_info.items():
                metrics['eval/' + k].append(
                    np.array([step, v])
                )

    for k, v in metrics.items():
        metrics[k] = np.asarray(v)

    return metrics


def main(_):
    # sanity check
    # not full rank
    A = jnp.array([
        [1., 2., 3.],
        [2., 4., 6.]
    ], dtype=jnp.float32)
    # null_basis = compute_null_basis(A)
    scipy_null_basis = scipy.linalg.null_space(A)
    assert np.allclose(np.abs(A @ scipy_null_basis), 0.0, atol=1e-6)

    # full rank
    A = jnp.array([
        [2., 1.],
        [5., 3.],
    ])
    # null_basis = compute_null_basis(A)
    scipy_null_basis = scipy.linalg.null_space(A)
    assert scipy_null_basis.shape[-1] == 0

    behavioral_policy = np.ones([FLAGS.num_observations, FLAGS.num_actions]) / FLAGS.num_actions
    dataset = collect_dataset(behavioral_policy)

    # TODO (chongyi): use different loss type
    # loss_types = ['mc_lsif', 'td_lsif']
    metrics = train_and_eval(dataset, num_training_steps=FLAGS.num_training_steps, eval_interval=FLAGS.eval_interval)

    f, axes = plot_metrics(metrics, f_axes=None, logyscale_stats=['eval/prob_ratio_error'])
    f.tight_layout()
    f.suptitle('repr_dim = {}'.format(FLAGS.repr_dim))
    f.savefig("figures/sl_fb_counter_examples_no_latent_repr_dim={}.png".format(
        FLAGS.repr_dim), dpi=300)
    print()


if __name__ == '__main__':
    app.run(main)
