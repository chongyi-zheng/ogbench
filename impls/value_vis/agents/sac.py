import copy
from typing import Any
from tqdm import tqdm

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax

from impls.utils.evaluation import supply_rng
from impls.utils.networks import ensemblize, TransformedWithMode, LogParam
from impls.value_vis.utils import default_init, explore, evaluate


class QValue(nn.Module):
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, observations, actions):
        q = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(1, kernel_init=self.kernel_init),
        ])(jnp.concatenate([observations, actions], axis=-1))

        q = q.squeeze(axis=-1)

        return q


class Actor(nn.Module):
    action_dim: int
    kernel_init: Any = default_init()
    final_fc_init_scale: float = 1e-2
    log_std_min: float = -5
    log_std_max: float = 2

    @nn.compact
    def __call__(self, observations, temperature=1.0):
        outputs = nn.Sequential([
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(512, kernel_init=self.kernel_init),
            nn.relu,
        ])(observations)
        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(self.final_fc_init_scale))(outputs)
        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init(self.final_fc_init_scale))(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


def get_batch(dataset, batch_size, discount=0.99):
    dataset_size = dataset['observations'].shape[0]
    terminal_locs = np.nonzero(dataset['terminals'] > 0)[0]

    idxs = np.random.randint(dataset_size, size=batch_size)
    batch = jax.tree_util.tree_map(lambda arr: arr[idxs], dataset)

    # sample future observation from truncated geometric distribution
    final_obs_idxs = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    offsets = np.random.geometric(p=1.0 - discount, size=(batch_size, ))  # in [1, inf)
    future_obs_idxs = np.minimum(idxs + offsets, final_obs_idxs)
    batch['future_observations'] = jax.tree_util.tree_map(
        lambda arr: arr[future_obs_idxs], dataset['observations'])

    return batch


def train_and_eval_online_sac(env, key,
                              discount=0.99, tau=0.005,
                              num_ensembles=2, target_entropy_multiplier=0.5,
                              batch_size=256, learning_rate=3e-4,
                              num_training_steps=50_000, expl_interval=1000, eval_interval=10_000):
    target_entropy = -target_entropy_multiplier * env.action_space.shape[0]

    def alpha_loss_fn(alpha_params, actor_params, alpha_fn, actor_fn, batch, key):
        key, action_key = jax.random.split(key)

        dist = actor_fn.apply(actor_params, batch['observations'])
        _, log_probs = dist.sample_and_log_prob(seed=action_key)

        alpha = alpha_fn.apply(alpha_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - target_entropy)).mean()

        return alpha_loss, {
            'alpha_loss': alpha_loss,
            'alpha': alpha,
        }

    def actor_loss_fn(actor_params, q_params, alpha_params,
                      actor_fn, q_value_fn, alpha_fn, batch, key):
        """Compute the actor loss."""
        key, action_key = jax.random.split(key)
        observations = batch['observations']

        dist = actor_fn.apply(actor_params, observations)
        q_actions, log_probs = dist.sample_and_log_prob(seed=action_key)

        # Compute actor loss
        qs = q_value_fn.apply(q_params, observations, q_actions)
        q = qs.min(axis=0)

        alpha = alpha_fn.apply(alpha_params)
        actor_loss = (log_probs * alpha - q).mean()

        # for logging
        action_std = dist._distribution.stddev()

        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'std': action_std.mean(),
        }

    def critic_loss_fn(q_params, target_q_params, actor_params, alpha_params,
                       q_value_fn, actor_fn, alpha_fn, batch, key):
        """Compute the Q-learning loss."""
        key, next_action_key = jax.random.split(key)

        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        next_dist = actor_fn.apply(actor_params, next_observations)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_action_key)

        next_qs = q_value_fn.apply(target_q_params, next_observations, next_actions)
        next_q = next_qs.min(axis=0)
        target_q = rewards + discount * (1.0 - terminals) * next_q
        target_q = target_q - discount * (1.0 - terminals) * next_log_probs * alpha_fn.apply(alpha_params)

        qs = q_value_fn.apply(q_params, observations, actions)
        critic_loss = jnp.square(target_q - qs).mean()

        # for logging
        q = qs.min(axis=0)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    key, obs_key, action_key = jax.random.split(key, 3)
    example_observations = np.stack([
        env.observation_space.sample(), env.observation_space.sample()], axis=0)
    example_actions = np.stack([
        env.action_space.sample(), env.action_space.sample()], axis=0)

    key, alpha_key, actor_key, q_value_key = jax.random.split(key, 4)

    alpha_fn = LogParam()
    actor_fn = Actor(env.action_space.shape[0])
    if num_ensembles > 1:
        q_value_fn = ensemblize(QValue, num_ensembles)()
    else:
        q_value_fn = QValue()

    alpha_params = alpha_fn.init(alpha_key)
    actor_params = actor_fn.init(
        actor_key, example_observations)
    q_params = q_value_fn.init(
        q_value_key, example_observations, example_actions)
    target_q_params = copy.deepcopy(q_params)

    alpha_optimizer = optax.adam(learning_rate=learning_rate)
    alpha_opt_state = alpha_optimizer.init(alpha_params)
    actor_optimizer = optax.adam(learning_rate=learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    q_optimizer = optax.adam(learning_rate=learning_rate)
    q_opt_state = q_optimizer.init(q_params)

    alpha_grad_fn = jax.value_and_grad(alpha_loss_fn, has_aux=True)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)

    @jax.jit
    def update_fn(q_params, target_q_params, actor_params, alpha_params,
                  q_opt_state, actor_opt_state, alpha_opt_state, batch, key):
        key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        info = dict()

        (_, critic_info), critic_grads = critic_grad_fn(
            q_params, target_q_params, actor_params, alpha_params,
            q_value_fn, actor_fn, alpha_fn,
            batch, critic_key
        )

        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        (_, actor_info), actor_grads = actor_grad_fn(
            actor_params, q_params, alpha_params,
            actor_fn, q_value_fn, alpha_fn,
            batch, actor_key
        )

        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        (_, alpha_info), alpha_grads = alpha_grad_fn(
            alpha_params, actor_params,
            alpha_fn, actor_fn,
            batch, alpha_key
        )

        for k, v in alpha_info.items():
            info[f'alpha/{k}'] = v

        q_updates, q_opt_state = q_optimizer.update(
            critic_grads, q_opt_state)
        q_params = optax.apply_updates(q_params, q_updates)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)
        alpha_updates, alpha_opt_state = alpha_optimizer.update(
            alpha_grads, alpha_opt_state)
        alpha_params = optax.apply_updates(alpha_params, alpha_updates)

        target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            target_q_params, q_params)

        return (q_params, target_q_params, actor_params, alpha_params,
                q_opt_state, actor_opt_state, alpha_opt_state, info)

    @jax.jit
    def sample_actions(params, observations, seed=None, temperature=1.0):
        dist = actor_fn.apply(params, observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)

        return actions

    key, expl_key, eval_key = jax.random.split(key, 3)
    expl_sample_action_fn = supply_rng(sample_actions, rng=expl_key)
    eval_sample_action_fn = supply_rng(sample_actions, rng=eval_key)

    metrics = dict()
    dataset = dict()
    for step in tqdm(range(num_training_steps + 1), desc="sac training"):
        if step % expl_interval == 0:
            key, sample_key = jax.random.split(key)
            expl_info, expl_episodes = explore(expl_sample_action_fn, env, actor_params, desc='sac exploration')

            for k, v in expl_episodes.items():
                data_key = k + 's'
                if k not in dataset:
                    dataset[data_key] = v
                else:
                    dataset[data_key] = np.concatenate([
                        dataset[data_key], v], axis=0)

            for k, v in expl_info.items():
                eval_k = 'expl/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

        key, train_key = jax.random.split(key)

        batch = get_batch(dataset, batch_size)
        (q_params, target_q_params, actor_params, alpha_params,
         q_opt_state, actor_opt_state, alpha_opt_state, info) = update_fn(
            q_params, target_q_params, actor_params, alpha_params,
            q_opt_state, actor_opt_state, alpha_opt_state, batch, train_key)

        for k, v in info.items():
            train_k = 'train/' + k
            metric = np.array([[step, v]])
            if train_k not in metrics:
                metrics[train_k] = metric
            else:
                metrics[train_k] = np.concatenate([metrics[train_k], metric], axis=0)

        if step % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_info = evaluate(eval_sample_action_fn, env, actor_params, desc='sac evaluation')

            for k, v in eval_info.items():
                eval_k = 'eval/' + k
                metric = np.array([[step, v]])
                if eval_k not in metrics:
                    metrics[eval_k] = metric
                else:
                    metrics[eval_k] = np.concatenate([metrics[eval_k], metric], axis=0)

    return metrics, dataset

