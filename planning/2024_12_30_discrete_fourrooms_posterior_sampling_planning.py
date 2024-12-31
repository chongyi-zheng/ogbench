import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import tqdm
from IPython import display
from scipy.ndimage import gaussian_filter1d
import gymnasium as gym

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import common_utils
import optax


WALLS = {
    "FourRooms": np.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]
    ),
}


def resize_walls(walls, factor):
    """Increase the environment by rescaling.

    Args:
      walls: 0/1 array indicating obstacle locations.
      factor: (int) factor by which to rescale the environment.
    """
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class DiscretePointEnv:
    """Abstract class for 2D navigation environments."""

    def __init__(
            self,
            walls='FourRooms',
            random_initial_state=True,
            gamma=0.95,
    ):
        """Initialize the point environment.

        Args:
          walls: (str) Name of one of the maps defined above.
          random_initial_state: (bool) Whether the initial state should be chosen
            uniformly across the state space, or fixed at (0, 0).
          gamma: (float) The discount factor.
        """
        self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._random_initial_state = random_initial_state
        self._gamma = gamma

        self.candidate_states = np.where(self._walls == 0)

        self.state_dim = 1
        self.action_dim = 1
        self.action_space = gym.spaces.Discrete(5)  # null - 0, up - 1, left - 2, down - 3, right - 4
        self.observation_space = gym.spaces.Discrete(len(self.candidate_states[0]))

        self.state = 0
        self.goal = self.candidate_states[-1]

    def _sample_empty_state(self):
        state = int(self.observation_space.sample())
        return state

    def state2coordinates(self, state):
        row = self.candidate_states[0][state]
        col = self.candidate_states[1][state]

        return row, col

    def state2xys(self, state):
        row, col = self.state2coordinates(state)

        return np.stack([col, row], axis=-1)

    def coordinates2state(self, row, col):
        state = int(np.sum(self._walls.flatten()[:row * self._width + col] == 0))

        return state

    def compute_reward(self, state, goal=None):
        if goal is None:
            goal = self.goal
        if state == goal:
            return (1 - self._gamma)
        else:
            return 0

    def get_next_state(self, state, action):
        curr_state = self.state
        self.state = state
        next_state = self.step(action)[0]
        self.state = curr_state

        return next_state

    def reset(self, state=None, goal=None):
        if self._random_initial_state:
            self.state = self._sample_empty_state()
        else:
            self.state = state if state is not None else 0
        self.goal = goal if goal is not None else self._sample_empty_state()

        info = dict(goal=self.goal)
        return self.state, info

    def step(self, action):
        reward = self.compute_reward(self.state, self.goal)

        row, col = self.state2coordinates(self.state)
        if action == 0:  # noop
            pass
        elif action == 1:  # up
            if (row - 1 >= 0) and (not self._walls[row - 1, col]):
                self.state = self.coordinates2state(row - 1, col)
        elif action == 2:  # left
            if (col - 1 >= 0) and (not self._walls[row, col - 1]):
                self.state = self.coordinates2state(row, col - 1)
        elif action == 3:  # down
            if (row + 1 <= self._height - 1) and (not self._walls[row + 1, col]):
                self.state = self.coordinates2state(row + 1, col)
        elif action == 4:  # right
            if (col + 1 <= self._width - 1) and (not self._walls[row, col + 1]):
                self.state = self.coordinates2state(row, col + 1)
        else:
            raise RuntimeError("Invalid action: {}".format(action))

        done = float(self.state == self.goal)
        success = float(self.state == self.goal)
        info = dict(goal=self.goal, success=success)

        return self.state, reward, done, info


def main():
    # Collect dataset
    gamma = 0.95
    env = DiscretePointEnv(walls='FourRooms', gamma=gamma)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    goal = num_states // 2
    beta = np.ones((num_states, num_actions)) / num_actions
    num_trajs = 1_000  # 100K dataset
    max_episode_steps = 100

    traj_dataset = []

    for traj_idx in tqdm.trange(num_trajs):
        traj = []
        # randomized the initial state
        s, _ = env.reset(goal=goal)
        for t in range(max_episode_steps):
            a = np.random.choice(num_actions, p=beta[s])
            next_s, r, d, _ = env.step(a)

            traj.append((s, a, r, d, next_s, traj_idx, t))
            s = next_s
        traj_dataset.append(traj)

    traj_dataset = np.array(traj_dataset)  # (num_traj, max_episode_steps, 5)
    traj_dataset_dict = {
        's': traj_dataset[..., 0].astype(np.int64),
        'a': traj_dataset[..., 1].astype(np.int64),
        'r': traj_dataset[..., 2],
        'd': traj_dataset[..., 3],
        'next_s': traj_dataset[..., 4].astype(np.int64),
        'traj_idx': traj_dataset[..., 5].astype(np.int64),
        't': traj_dataset[..., 6].astype(np.int64),
    }

    dataset_dict = dict(traj_dataset_dict)
    for k, v in dataset_dict.items():
        dataset_dict[k] = v[:, :-1].flatten()
    dataset_dict['next_a'] = traj_dataset_dict['a'][:, 1:].flatten()

    # Compute the ground truth future state distribution
    sa_transition_prob = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            next_s = env.get_next_state(s, a)
            sa_transition_prob[s, a, next_s] = 1
    s_transition_prob = np.sum(sa_transition_prob * beta[:, :, None], axis=1)

    inv = np.linalg.inv(np.eye(num_states) - gamma * s_transition_prob)

    # start from the current timestep
    # s_future_state_prob = (1 - gamma) * inv
    # start from the next timestep
    s_future_state_prob = (1 - gamma) * inv @ s_transition_prob
    s_future_state_prob = np.clip(s_future_state_prob, 0.0, 1.0)

    # the marginal state distribution is uniform
    s_marginal = np.ones(num_states)
    s_marginal /= np.sum(s_marginal, axis=-1, keepdims=True)

    # future state marginal distribution
    s_future_marginal = np.sum(s_transition_prob * s_marginal[:, None], axis=0)

    # locally optimal scripted policy
    candidate_states = jnp.array(env.candidate_states)

    # Define planner
    def state2xys(state):
        row = candidate_states[0, state]
        col = candidate_states[1, state]

        return jnp.stack([col, row], axis=-1)

    def scripted_policy(state, goal):
        state_xy, goal_xy = state2xys([state, goal])
        dcol = goal_xy[0] - state_xy[0]
        drow = goal_xy[1] - state_xy[1]

        action = jnp.select(condlist=[dcol > 0, dcol < 0, drow > 0, drow < 0],
                            choicelist=[4, 2, 3, 1],
                            default=0)

        return action

    # posterior sampling planner
    class Policy:
        def __init__(self,
                     s_future_state_prob,
                     s_future_marginal,
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

            self.s_future_state_prob = jnp.asarray(s_future_state_prob)
            self.s_future_marginal = jnp.asarray(s_future_marginal)

        def get_action(self, obs, goal, key, t, last_waypoint_seqs, last_waypoint_seq_idx, last_waypoint):
            # def planning_f(key):
            #     key, marginal_key, score_key = jax.random.split(key, num=3)
            #     waypoint_seqs = jax.random.choice(
            #         marginal_key, num_states,
            #         shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence),
            #         p=self.s_future_marginal,
            #     )
            #
            #     if self.dist_type == 'analytical_crl_ratio':
            #         ratio = jnp.log(self.s_future_state_prob) - jnp.log(self.s_future_marginal[None, :])
            #     elif self.dist_type == 'empirical_crl_ratio':
            #         # ratio = jnp.log(empirical_s_future_state_prob) - jnp.log(empirical_future_s_marginal_prob[None, :])
            #         raise NotImplementedError
            #
            #     sw_logits = ratio[obs, waypoint_seqs[:, 0]]  # (N, ), log p(w_1 | s) - log p(w_1)
            #     ww_logits = ratio[waypoint_seqs[:, :-1], waypoint_seqs[:, 1:]]  # (N, T - 1), log p(w_{t + 1} | w_t) - log p(w_{t + 1})
            #     wg_logits = ratio[waypoint_seqs[:, -1], goal]  # (N, )  log p(g | w_T) - log p(g)
            #
            #     log_scores = jnp.concatenate([sw_logits[:, None], ww_logits, wg_logits[:, None]], axis=-1)  # (N, T + 1)
            #     log_scores = jnp.sum(log_scores, axis=-1)
            #     scores = jax.nn.softmax(log_scores)
            #
            #     w_seq_idx = jax.random.choice(score_key, self.num_waypoint_sequences, p=scores)
            #     # w_seq_idx = jnp.argmax(scores)
            #     waypoint = waypoint_seqs[w_seq_idx, 0]
            #
            #     return waypoint_seqs, w_seq_idx, waypoint

            def planning_f(key):
                # key, seq_key, score_key = jax.random.split(key, num=3)

                # Cross entropy method
                probs = self.s_future_marginal[None, :].repeat(self.num_waypoints_in_sequence, axis=0)

                # for _ in range(self.num_cem_updates):
                #     key, seq_key = jax.random.split(key)
                #     # waypoint_seqs = jax.random.categorical(
                #     #     seq_key, jnp.log(probs), shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence))
                #
                #     c = probs.cumsum(axis=-1)
                #     u = jax.random.uniform(
                #         seq_key, shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence, 1))
                #     waypoint_seqs = (u < c[None]).argmax(axis=-1)
                #
                #     if self.dist_type == 'analytical_crl_ratio':
                #         ratio = jnp.log(self.s_future_state_prob) - jnp.log(self.s_future_marginal[None, :])
                #     elif self.dist_type == 'empirical_crl_ratio':
                #         # ratio = jnp.log(empirical_s_future_state_prob) - jnp.log(empirical_future_s_marginal_prob[None, :])
                #         raise NotImplementedError
                #
                #     sw_logits = ratio[obs, waypoint_seqs[:, 0]]  # (N, ), log p(w_1 | s) - log p(w_1)
                #     ww_logits = ratio[waypoint_seqs[:, :-1], waypoint_seqs[:, 1:]]  # (N, T - 1), log p(w_{t + 1} | w_t) - log p(w_{t + 1})
                #     wg_logits = ratio[waypoint_seqs[:, -1], goal]  # (N, )  log p(g | w_T) - log p(g)
                #     sg_logits = ratio[obs, goal]
                #     gg_logits = ratio[goal, goal]
                #
                #     log_scores = jnp.concatenate([sw_logits[:, None], ww_logits, wg_logits[:, None]], axis=-1)  # (N, T + 1)
                #     waypoint_seq_probs = jax.vmap(
                #         lambda seq: probs[jnp.arange(self.num_waypoints_in_sequence), seq], in_axes=0, out_axes=0
                #     )(waypoint_seqs)
                #     waypoint_seq_marg_probs = jax.vmap(
                #         lambda seq: self.s_future_marginal[None, :].repeat(self.num_waypoints_in_sequence, axis=0)[
                #             jnp.arange(self.num_waypoints_in_sequence), seq], in_axes=0, out_axes=0
                #     )(waypoint_seqs)
                #     importance_weights = waypoint_seq_marg_probs / waypoint_seq_probs
                #     importance_weights = jnp.prod(importance_weights, axis=-1)
                #     log_scores = jnp.sum(log_scores, axis=-1)
                #     log_scores += jnp.log(importance_weights)
                #     # scores = jax.nn.softmax(log_scores)
                #
                #     # cross entropy method: update sampling probabilities
                #     top_k_scores, top_k_indices = jax.lax.top_k(log_scores, self.cem_top_k)
                #     top_k_waypoint_seqs = waypoint_seqs[top_k_indices]
                #     new_probs = jax.nn.one_hot(top_k_waypoint_seqs, num_classes=num_states)
                #     new_probs = jnp.mean(new_probs, axis=0)
                #
                #     probs = (1 - self.cem_mixing_coef) * new_probs + self.cem_mixing_coef * probs

                def cross_entropy_method_fn(carry, key):
                    *_, probs = carry

                    # waypoint_seqs = jax.random.categorical(
                    #     key, jnp.log(probs), shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence))

                    c = probs.cumsum(axis=-1)
                    u = jax.random.uniform(
                        key, shape=(self.num_waypoint_sequences, self.num_waypoints_in_sequence, 1))
                    waypoint_seqs = (u < c[None]).argmax(axis=-1)

                    if self.dist_type == 'analytical_crl_ratio':
                        ratio = jnp.log(self.s_future_state_prob) - jnp.log(self.s_future_marginal[None, :])
                    elif self.dist_type == 'empirical_crl_ratio':
                        raise NotImplementedError

                    sw_logits = ratio[obs, waypoint_seqs[:, 0]]  # (N, ), log p(w_1 | s) - log p(w_1)
                    ww_logits = ratio[waypoint_seqs[:, :-1], waypoint_seqs[:, 1:]]  # (N, T - 1), log p(w_{t + 1} | w_t) - log p(w_{t + 1})
                    wg_logits = ratio[waypoint_seqs[:, -1], goal]  # (N, )  log p(g | w_T) - log p(g)
                    sg_logits = ratio[obs, goal]
                    gg_logits = ratio[goal, goal]

                    log_scores = jnp.concatenate([sw_logits[:, None], ww_logits, wg_logits[:, None]],
                                                 axis=-1)  # (N, T + 1)
                    waypoint_seq_probs = jax.vmap(
                        lambda seq: probs[jnp.arange(self.num_waypoints_in_sequence), seq], in_axes=0, out_axes=0
                    )(waypoint_seqs)
                    waypoint_seq_marg_probs = jax.vmap(
                        lambda seq: self.s_future_marginal[None, :].repeat(self.num_waypoints_in_sequence, axis=0)[
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
                          jnp.zeros(()), jnp.zeros(()), jnp.zeros((self.num_waypoint_sequences, )),
                          probs),
                    xs=jnp.asarray(seq_keys),
                )

                log_scores += log_importance_weights  # this is important
                scores = jax.nn.softmax(log_scores)

                key, score_key = jax.random.split(key)
                w_seq_idx = jax.random.choice(score_key, self.num_waypoint_sequences, p=scores)
                # w_seq_idx = jnp.argmax(scores)

                # if log_scores[w_seq_idx] <= (sg_logits + self.num_waypoints_in_sequence * gg_logits):
                #     waypoint = goal
                # else:
                #     waypoint = waypoint_seqs[w_seq_idx, 0]
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

            # waypoint_seqs, waypoint_seq_idx, waypoint = planning_f(planning_key)

            last_waypoint_seqs = waypoint_seqs
            last_waypoint_seq_idx = waypoint_seq_idx
            last_waypoint = waypoint

            if self.use_scripted_policy:
                return scripted_policy(obs, waypoint), waypoint_seqs, waypoint_seq_idx, waypoint
            else:
                raise NotImplementedError

    # planning
    NUM_EPISODES = 100
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
    policy = Policy(s_future_state_prob,
                    s_future_marginal,
                    dist_type='analytical_crl_ratio',
                    planning_per_steps=2,
                    num_states=num_states,
                    num_waypoints_in_sequence=num_waypoints_in_sequence,
                    num_waypoint_sequences=num_waypoint_sequences,
                    num_cem_updates=num_cem_updates,
                    cem_top_k=cem_top_k,
                    cem_mixing_coef=cem_mixing_coef,
                    use_scripted_policy=True)
    policy.get_action = jax.jit(policy.get_action)

    for episode_idx in tqdm.tqdm(range(NUM_EPISODES)):
        obs, info = env.reset()
        goal = info['goal']
        trajs['goal'].append(goal)

        key = jax.random.PRNGKey(episode_idx)
        done = False
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
            next_obs, reward, done, info = env.step(action)

            episode['obs'].append(obs)
            episode['action'].append(action)
            episode['reward'].append(reward)
            episode['done'].append(done)
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

    success_rate = np.mean(np.any(trajs['success'] > 0, axis=-1))

    print("success rate = {}".format(success_rate))
    print()

    # 2D contours
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
        states = np.arange(num_states)
        state_xys = env.state2xys(states)

        # print()
        # print(waypoints[episode_idx])

        obs_xys = env.state2xys(obs[episode_idx])
        waypoint_seq_idx = waypoint_seq_idxs[episode_idx]
        sampled_waypoint_seq = waypoint_seqs[episode_idx][np.arange(max_episode_steps), waypoint_seq_idx]
        sampled_waypoint_seq_xys = env.state2xys(sampled_waypoint_seq)
        # print(sampled_waypoint_seq_xys.shape)
        waypoint_xys = env.state2xys(waypoints[episode_idx])
        goal_xy = env.state2xys(goal[episode_idx])

        ax = axes[episode_idx]
        ax.scatter(state_xys[:, 0], state_xys[:, 1], s=size, marker='s', color='tab:blue')
        ax.scatter(obs_xys[0, 0], obs_xys[0, 1], s=5 * size, marker='x', color='red', label=r'$s_0$')
        ax.plot(obs_xys[:, 0], obs_xys[:, 1], color='k', label=r'$s_t$')
        ax.scatter(waypoint_xys[:, 0], waypoint_xys[:, 1], s=size, marker='s', color='orange', label='$w$')
        # ax.scatter(waypoint_seq_xys[0, 0, :, 0], waypoint_seq_xys[0, 0, :, 1], s=size, marker='s', color='orange', label='$w$')
        # ax.scatter(sampled_waypoint_seq_xys[0, :, 0], sampled_waypoint_seq_xys[0, :, 1], s=20, marker='s', color='orange', label='$w$')
        # ax.scatter(sampled_waypoint_seq_xys[0, 0, 0], sampled_waypoint_seq_xys[0, 0, 1], s=size, marker='s',
        #            color='tab:purple', label='first $w$')
        ax.scatter(waypoint_xys[0, 0], waypoint_xys[0, 1], s=size, marker='s', color='tab:purple', label='first $w$')
        ax.scatter(goal_xy[0], goal_xy[1], s=5 * size, marker='*', color='green', label=r'$g$')

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim([-1.1, 11.1])
        ax.set_ylim([-1.1, 11.1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax.set_title(r"Analytical $\log\frac{p(g | s)}{p(g)}$ posterior" + "\nsampling planner + scripted policy")

    plt.tight_layout()
    plt.savefig("./cem_posterior_sampling.png")


if __name__ == "__main__":
    main()
