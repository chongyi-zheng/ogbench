import jax
import tqdm
import time
import flax
import distrax
import chex
import wandb
import optax
import pickle
import random
import ogbench
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections

import ogbench
from impls.agents import agents
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.wrappers import OfflineObservationNormalizer
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.flax_utils import restore_agent, save_agent
from impls.utils.evaluation import supply_rng


def main():
    env_name = "pointmaze-medium-navigate-v0"
    seed = np.random.randint(2**32 - 1)
    batch_size = 1024

    random.seed(seed)
    np.random.seed(seed)

    from impls.agents.crl_infonce import get_config
    config = get_config()

    # load datasets and env
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, compact_dataset=False)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    env.reset()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("obs_dim = {}".format(obs_dim))
    print("action_dim = {}".format(action_dim))

    # Discretize the training dataset
    observations = train_dataset['observations']
    next_observations = train_dataset['next_observations']

    rounded_obs_next_obs = np.round(
        jnp.concatenate([observations, next_observations], axis=0),
        decimals=0
    )
    state_xys, state_next_states = np.unique(rounded_obs_next_obs, axis=0, return_inverse=True)

    states = state_next_states[:observations.shape[0]]
    next_states = state_next_states[observations.shape[0]:]

    # Compute analytical and empirical discounted state occupancy measure
    num_states = states.max() + 1

    s_transition_prob = np.zeros((num_states, num_states))
    np.add.at(s_transition_prob, (states, next_states), 1)
    s_transition_prob /= np.sum(s_transition_prob, axis=-1, keepdims=True)

    # starts from the current time step
    # inv = np.linalg.inv(np.eye(num_states) - config['discount'] * s_transition_prob)
    # s_future_state_prob = (1 - config['discount']) * inv
    # starts from the next time step
    inv = np.linalg.inv(np.eye(num_states) - config['discount'] * s_transition_prob)
    s_future_state_prob = (1 - config['discount']) * inv @ s_transition_prob
    s_future_state_prob = np.clip(s_future_state_prob, 0.0, 1.0)

    s_marginal_prob = np.zeros(num_states)
    np.add.at(s_marginal_prob, states, 1)
    s_marginal_prob /= np.sum(s_marginal_prob, axis=-1, keepdims=True)

    s_future_marginal_prob = np.sum(s_future_state_prob * s_marginal_prob[:, None], axis=0)
    s_future_marginal_prob /= np.sum(s_future_marginal_prob, axis=-1, keepdims=True)

    # Collect trajectory dataset
    terminal_locs = np.nonzero(train_dataset['terminals'] > 0)[0]
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])

    final_state_idxs = terminal_locs[np.searchsorted(terminal_locs, np.arange(len(train_dataset['terminals'])))]
    initial_state_idxs = initial_locs[
        np.searchsorted(initial_locs, np.arange(len(train_dataset['terminals'])), side='right') - 1]

    traj_start_idxs = np.unique(initial_state_idxs)
    traj_end_idxs = np.unique(final_state_idxs)

    traj_len = traj_end_idxs[0] - traj_start_idxs[0] + 1

    traj_datasets = {
        'observations': [],
        'next_observations': [],
        'actions': [],
    }
    for idx, (traj_start_idx, traj_end_idx) in enumerate(zip(traj_start_idxs, traj_end_idxs)):
        if traj_start_idx == traj_end_idx:
            continue

        traj_observations = train_dataset['observations'][traj_start_idx:traj_end_idx + 1]
        traj_next_observations = train_dataset['next_observations'][traj_start_idx:traj_end_idx + 1]
        traj_actions = train_dataset['actions'][traj_start_idx:traj_end_idx + 1]

        traj_datasets['observations'].append(traj_observations)
        traj_datasets['next_observations'].append(traj_next_observations)
        traj_datasets['actions'].append(traj_actions)

    for k, v in traj_datasets.items():
        traj_datasets[k] = np.asarray(v)

    rounded_traj_observations = np.round(
        traj_datasets['observations'].reshape([-1, traj_datasets['observations'].shape[-1]]), decimals=0)
    _, traj_states = np.unique(rounded_traj_observations, axis=0, return_inverse=True)
    rounded_traj_next_observations = np.round(
        traj_datasets['next_observations'].reshape([-1, traj_datasets['next_observations'].shape[-1]]), decimals=0)
    _, traj_next_states = np.unique(rounded_traj_next_observations, axis=0, return_inverse=True)

    assert traj_states.max() + 1 == num_states

    traj_datasets['states'] = traj_states.reshape(*traj_datasets['observations'].shape[:-1])
    traj_datasets['next_states'] = traj_next_states.reshape(*traj_datasets['next_observations'].shape[:-1])

    # compute the empirical discounted state occupancy measure
    def compute_s_future_state_t_counts(traj_states, num_states):
        """
        vectorized implementation of the following code (generated by chatgpt):

        s_future_state_t_counts = jnp.zeros([traj_len, num_states, num_states], dtype=int)
        for t in range(traj_len):
          states = traj_datasets['states'][:, t]
          future_states = traj_datasets['states'][:, t:]

          s_future_state_t_counts = s_future_state_t_counts.at[jnp.arange(traj_len - t)[None, :], states[:, None], future_states].add(1)

        """
        traj_len = traj_states.shape[1]  # Number of time steps in a trajectory

        # Create indices for current and future time steps
        t_indices = jnp.arange(traj_len)
        t_future_indices = jnp.arange(traj_len)

        # Compute all possible time differences
        delta_t_matrix = t_future_indices[None, :] - t_indices[:, None]
        valid_mask = (delta_t_matrix >= 0) & (delta_t_matrix < traj_len)

        # Flatten the indices and mask
        delta_t_flat = delta_t_matrix.flatten()
        t_flat = jnp.broadcast_to(t_indices[:, None], (traj_len, traj_len)).flatten()
        t_future_flat = jnp.broadcast_to(t_future_indices[None, :], (traj_len, traj_len)).flatten()
        valid_mask_flat = valid_mask.flatten()

        # Select valid indices based on the mask
        delta_t_valid = delta_t_flat[valid_mask_flat]
        t_valid = t_flat[valid_mask_flat]
        t_future_valid = t_future_flat[valid_mask_flat]

        # Gather the states for current and future time steps
        s_valid = traj_states[:, t_valid]  # Shape: (N, num_valid_pairs)
        s_future_valid = traj_states[:, t_future_valid]  # Shape: (N, num_valid_pairs)

        # Repeat delta_t_valid for all trajectories
        delta_t_valid = jnp.broadcast_to(delta_t_valid, s_valid.shape)

        # Flatten the arrays to create indices
        delta_t_indices = delta_t_valid.flatten()
        s_indices = s_valid.flatten()
        s_future_indices = s_future_valid.flatten()

        # Initialize the counts array
        s_future_state_t_counts = jnp.zeros((traj_len, num_states, num_states), dtype=jnp.int32)

        # Use JAX's advanced indexing to accumulate counts
        s_future_state_t_counts = s_future_state_t_counts.at[
            delta_t_indices, s_indices, s_future_indices
        ].add(1)

        return s_future_state_t_counts

    s_future_state_t_counts = compute_s_future_state_t_counts(
        traj_datasets['states'], num_states=num_states)  # (traj_len, num_states, num_states)
    s_future_state_t_counts = s_future_state_t_counts.astype(jnp.float32)
    s_future_state_t_prob = s_future_state_t_counts / (jnp.sum(s_future_state_t_counts, axis=-1, keepdims=True) + 1e-12)

    # starts from the next time step
    discounts = config['discount'] ** jnp.arange(traj_len - 1)
    discounts = discounts.at[-1].set(discounts[-1] * (1 + config['discount']))  # for truncated geometric distribution

    empirical_s_future_state_prob = (1 - config['discount']) * jnp.sum(
        discounts[:, None, None] * s_future_state_t_prob[1:], axis=0)
    empirical_s_future_state_prob /= (1 - config['discount'] ** traj_len)

    empirical_future_s_marginal_prob = jnp.sum(empirical_s_future_state_prob * s_marginal_prob[:, None], axis=0)
    empirical_future_s_marginal_prob /= jnp.sum(empirical_future_s_marginal_prob, axis=-1, keepdims=True)

    # Planning with posterior sampling
    # locally optimal scripted policy
    def scripted_policy(state_xys, observations, goals):
        obs_xys = state_xys[observations]
        goal_xys = state_xys[goals]
        actions = jnp.clip(0.25 * (goal_xys - obs_xys), -1.0, 1.0)

        return actions

    class Policy:
        def __init__(self,
                     state_xys,
                     s_future_state_prob,
                     s_future_marginal_prob,
                     dist_type='analytical_crl_ratio',
                     planning_per_steps=1,
                     num_states=104,
                     num_waypoints_in_sequence=10,
                     num_waypoint_sequences=1024,
                     num_cem_updates=20,
                     cem_top_k=256,
                     cem_mixing_coef=0.25,
                     use_scripted_policy=True):
            self.dist_type = dist_type
            self.planning_per_steps = planning_per_steps
            self.num_states = num_states
            self.num_waypoints_in_sequence = num_waypoints_in_sequence
            self.num_waypoint_sequences = num_waypoint_sequences
            self.num_cem_updates = num_cem_updates
            self.cem_top_k = cem_top_k
            self.cem_mixing_coef = cem_mixing_coef
            self.use_scripted_policy = use_scripted_policy

            assert self.cem_top_k <= self.num_waypoint_sequences

            self.state_xys = jnp.asarray(state_xys)
            self.s_future_state_prob = jnp.asarray(s_future_state_prob)
            self.s_future_marginal_prob = jnp.asarray(s_future_marginal_prob)

        def get_action(self, obs, goal, key, t, last_waypoint_seqs, last_waypoint_seq_idx, last_waypoint):

            def planning_f(key):
                probs = self.s_future_marginal_prob[None, :].repeat(self.num_waypoints_in_sequence, axis=0)

                def cross_entropy_method_fn(carry, key):
                    *_, probs = carry

                    c = probs.cumsum(axis=-1)
                    u = jax.random.uniform(
                        key, shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence, 1))
                    waypoint_seqs = (u < c[None]).argmax(axis=-1)

                    if self.dist_type == 'analytical_crl_ratio':
                        ratio = jnp.log(self.s_future_state_prob) - jnp.log(self.s_future_marginal_prob[None, :])
                    elif self.dist_type == 'empirical_crl_ratio':
                        raise NotImplementedError

                    sw_logits = ratio[obs, waypoint_seqs[:, 0]]  # (N, ), log p(w_1 | s) - log p(w_1)
                    ww_logits = ratio[waypoint_seqs[:, :-1], waypoint_seqs[:,
                                                             1:]]  # (N, T - 1), log p(w_{t + 1} | w_t) - log p(w_{t + 1})
                    wg_logits = ratio[waypoint_seqs[:, -1], goal]  # (N, )  log p(g | w_T) - log p(g)
                    sg_logits = ratio[obs, goal]
                    gg_logits = ratio[goal, goal]

                    log_scores = jnp.concatenate([sw_logits[:, None], ww_logits, wg_logits[:, None]],
                                                 axis=-1)  # (N, T + 1)
                    waypoint_seq_probs = jax.vmap(
                        lambda seq: probs[jnp.arange(self.num_waypoints_in_sequence), seq], in_axes=0, out_axes=0
                    )(waypoint_seqs)
                    waypoint_seq_marg_probs = jax.vmap(
                        lambda seq: self.s_future_marginal_prob[None, :].repeat(self.num_waypoints_in_sequence, axis=0)[
                            jnp.arange(self.num_waypoints_in_sequence), seq], in_axes=0, out_axes=0
                    )(waypoint_seqs)
                    log_importance_weights = jnp.log(waypoint_seq_marg_probs) - jnp.log(waypoint_seq_probs)
                    log_importance_weights = jnp.sum(log_importance_weights, axis=-1)
                    log_scores = jnp.sum(log_scores, axis=-1)
                    # log_scores += jnp.log(importance_weights)

                    # cross entropy method: update sampling probabilities
                    top_k_scores, top_k_indices = jax.lax.top_k(log_scores, self.cem_top_k)
                    top_k_waypoint_seqs = waypoint_seqs[top_k_indices]
                    new_probs = jax.nn.one_hot(top_k_waypoint_seqs, num_classes=num_states)
                    new_probs = jnp.mean(new_probs, axis=0)

                    probs = (1 - self.cem_mixing_coef) * new_probs + self.cem_mixing_coef * probs

                    new_carry = (waypoint_seqs, log_scores, sg_logits, gg_logits, log_importance_weights, probs)

                    return new_carry, None

                key, *seq_keys = jax.random.split(key, num=self.num_cem_updates + 1)
                (waypoint_seqs, log_scores, sg_logits, gg_logits, log_importance_weights, probs), _ = jax.lax.scan(
                    cross_entropy_method_fn,
                    init=(jnp.zeros((self.num_waypoint_sequences, self.num_waypoints_in_sequence), dtype=jnp.int32),
                          jnp.zeros(self.num_waypoint_sequences),
                          jnp.zeros(()), jnp.zeros(()), jnp.zeros((self.num_waypoint_sequences,)), probs),
                    xs=jnp.asarray(seq_keys),
                )

                log_scores += log_importance_weights  # this is important
                scores = jax.nn.softmax(log_scores)

                key, score_key = jax.random.split(key)
                w_seq_idx = jax.random.choice(score_key, self.num_waypoint_sequences, p=scores)

                waypoint = jax.lax.select(
                    # log_scores[w_seq_idx] <= (sg_logits + self.num_waypoints_in_sequence * gg_logits),
                    obs == goal,
                    goal,
                    waypoint_seqs[w_seq_idx, 0]
                )
                # waypoint = waypoint_seqs[w_seq_idx, 0]

                return waypoint_seqs, w_seq_idx, waypoint

            key, planning_key = jax.random.split(key)
            waypoint_seqs, waypoint_seq_idx, waypoint = jax.lax.cond(
                t % self.planning_per_steps == 0,
                planning_f,
                lambda k: (last_waypoint_seqs, last_waypoint_seq_idx, last_waypoint),
                planning_key,
            )

            last_waypoint_seqs = waypoint_seqs
            last_waypoint_seq_idx = waypoint_seq_idx
            last_waypoint = waypoint

            if self.use_scripted_policy:
                return scripted_policy(self.state_xys, obs, waypoint), waypoint_seqs, waypoint_seq_idx, waypoint
            else:
                raise NotImplementedError

    NUM_EPISODES = 100
    max_episode_steps = 500

    trajs = {
        'obs': [],
        'action': [],
        'reward': [],
        'done': [],
        'discrete_success': [],
        'success': [],
        'waypoint_seqs': [],
        'waypoint_seq_idxs': [],
        'waypoints': [],
        'goal': [],
    }

    num_waypoints_in_sequence = 3
    num_waypoint_sequences = 10_000
    num_cem_updates = 10
    cem_top_k = 2500
    cem_mixing_coef = 0.25
    policy = Policy(state_xys,
                    s_future_state_prob,
                    s_future_marginal_prob,
                    dist_type='analytical_crl_ratio',
                    planning_per_steps=3,
                    num_states=num_states,
                    num_waypoints_in_sequence=num_waypoints_in_sequence,
                    num_waypoint_sequences=num_waypoint_sequences,
                    num_cem_updates=num_cem_updates,
                    cem_top_k=cem_top_k,
                    cem_mixing_coef=cem_mixing_coef,
                    use_scripted_policy=True)
    policy.get_action = jax.jit(policy.get_action)

    for episode_idx in tqdm.tqdm(range(NUM_EPISODES)):
        obs, info = env.reset(seed=episode_idx)
        goal = info['goal']

        rounded_obs = np.round(obs, decimals=0)
        obs = np.where(np.all(state_xys == rounded_obs, axis=-1))[0][0]

        rounded_goal = np.round(goal, decimals=0)
        goal = np.where(np.all(state_xys == rounded_goal, axis=-1))[0][0]

        trajs['goal'].append(goal)

        key = jax.random.PRNGKey(episode_idx)
        episode = {}
        for k in trajs.keys():
            if k not in ['goal']:
                episode[k] = []

        waypoint_seqs = -jnp.ones((num_waypoint_sequences, num_waypoints_in_sequence), dtype=jnp.int32)
        waypoint_seq_idx = -jnp.ones((), dtype=jnp.int32)
        waypoint = -jnp.ones((), dtype=jnp.int32)
        for t in range(max_episode_steps):
            key, policy_key = jax.random.split(key)
            action, waypoint_seqs, waypoint_seq_idx, waypoint = policy.get_action(
                obs, goal, policy_key, t, waypoint_seqs, waypoint_seq_idx, waypoint)
            next_obs, reward, _, done, info = env.step(action)

            rounded_next_obs = np.round(next_obs, decimals=0)
            next_obs = np.where(np.all(state_xys == rounded_next_obs, axis=-1))[0][0]

            episode['obs'].append(obs)
            episode['action'].append(action)
            episode['reward'].append(reward)
            episode['done'].append(done)
            episode['discrete_success'].append(float(obs == goal))
            episode['success'].append(info['success'])
            episode['waypoint_seqs'].append(waypoint_seqs)
            episode['waypoint_seq_idxs'].append(waypoint_seq_idx)
            episode['waypoints'].append(waypoint)

            obs = next_obs

        for k in trajs.keys():
            if k not in ['goal']:
                trajs[k].append(episode[k])

    for k, v in trajs.items():
        trajs[k] = np.array(v)

    discrete_success_rate = np.mean(np.any(trajs['discrete_success'] > 0, axis=-1))
    success_rate = np.mean(np.any(trajs['success'] > 0, axis=-1))

    print()
    print("discrete success rate = {}".format(discrete_success_rate))  # 0.07
    print("success rate = {}".format(success_rate))  # 0.08

    # plot trajectories
    obs = trajs['obs']
    goal = trajs['goal']
    waypoint_seqs = trajs['waypoint_seqs']
    waypoint_seq_idxs = trajs['waypoint_seq_idxs']
    waypoints = trajs['waypoints']

    NUM_PLOTS = 8

    fig, axes = plt.subplots(nrows=1, ncols=NUM_PLOTS)
    fig.set_figheight((3 + 1.5) * 1)
    fig.set_figwidth(3 * NUM_PLOTS)

    size = 10.0
    for episode_idx in range(NUM_PLOTS):
        obs_xys = state_xys[obs[episode_idx]]

        # waypoint_seq_idx = waypoint_seq_idxs[episode_idx]
        # sampled_waypoint_seq = waypoint_seqs[episode_idx][np.arange(max_episode_steps), waypoint_seq_idx]
        # sampled_waypoint_seq_xys = env.state2xys(sampled_waypoint_seq)
        # print(sampled_waypoint_seq_xys.shape)
        waypoint_xys = state_xys[waypoints[episode_idx]]
        goal_xy = state_xys[goal[episode_idx]]

        ax = axes[episode_idx]
        ax.scatter(state_xys[:, 0], state_xys[:, 1], s=size, marker='s', color='tab:blue')
        ax.scatter(obs_xys[0, 0], obs_xys[0, 1], s=5 * size, marker='x', color='red', label=r'$s_0$')
        ax.plot(obs_xys[:, 0], obs_xys[:, 1], color='k', label=r'$s_t$')
        ax.scatter(waypoint_xys[:, 0], waypoint_xys[:, 1], s=size, marker='s', color='orange', label='$w$')
        ax.scatter(waypoint_xys[0, 0], waypoint_xys[0, 1], s=size, marker='s', color='tab:purple', label='first $w$')
        ax.scatter(goal_xy[0], goal_xy[1], s=5 * size, marker='*', color='green', label=r'$g$')

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim([-1.5, 21.5])
        ax.set_ylim([-1.5, 21.5])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax.set_title(r"Analytical $\log\frac{p(g | s)}{p(g)}$ posterior" + "\nsampling planner + scripted policy")

    plt.tight_layout()
    plt.savefig("./discrete_pointmaze_cem_posterior_sampling.png")


if __name__ == "__main__":
    main()
