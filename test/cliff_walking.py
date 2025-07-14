from collections import defaultdict

import numpy as np
import gymnasium
from gymnasium.envs.toy_text.cliffwalking import (
    UP, RIGHT, DOWN, LEFT, POSITION_MAPPING
)
import tqdm


class CliffWalkingEnv(gymnasium.Wrapper):
    def __init__(self, random_init_state=False, max_episode_steps=1000, 
                 render_mode="rgb_array", **kwargs):
        env = gymnasium.make(
            "CliffWalking-v1",
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            **kwargs
        )
        super().__init__(env)

        self.nS = self.env.get_wrapper_attr('nS')
        self.nA = self.env.get_wrapper_attr('nA')
        self.shape = self.env.get_wrapper_attr('shape')

        # The original transition probabilities for absorbing states are not correct.
        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            P[s][UP] = self._calculate_transition_prob(position, UP)
            P[s][RIGHT] = self._calculate_transition_prob(position, RIGHT)
            P[s][DOWN] = self._calculate_transition_prob(position, DOWN)
            P[s][LEFT] = self._calculate_transition_prob(position, LEFT)
        self.env.set_wrapper_attr('P', P)

        if random_init_state:
            initial_state_distrib = np.ones(self.nS)
            cliff_positions = np.asarray(np.where(env.get_wrapper_attr('_cliff')))
            cliff_states = np.ravel_multi_index(cliff_positions, self.shape)
            initial_state_distrib[cliff_states] = 0.0
            initial_state_distrib[47] = 0.0
            initial_state_distrib /= np.sum(initial_state_distrib, keepdims=True)
            self.env.set_wrapper_attr('initial_state_distrib', initial_state_distrib)

        # Calculate transition probabilities and rewards
        rewards = np.full((self.nS, self.nA, self.nS), np.nan)
        transition_probs = np.zeros((self.nS, self.nA, self.nS))
        masks = np.zeros((self.nS, self.nA, self.nS))
        for state in range(self.nS):
            for action in range(self.nA):
                _, next_state, reward, terminated = self.env.get_wrapper_attr('P')[state][action][0]
                rewards[state, action, next_state] = reward
                transition_probs[state, action, next_state] += 1.0
                masks[state, action, next_state] = float(not terminated)
        transition_probs /= np.sum(transition_probs, axis=-1, keepdims=True)
        assert np.all(np.sum(transition_probs, axis=-1) == 1.0)
        reward_max, reward_min = np.nanmax(rewards), np.nanmin(rewards)
        rewards[np.isnan(rewards)] = reward_min
        assert np.all((reward_min <= rewards) & (rewards <= reward_max))

        self._orig_reward_min, self._orig_reward_max = reward_min, reward_max
        self.orig_rewards = rewards
        self.rewards = (rewards - reward_min) / (reward_max - reward_min)
        self.transition_probs = transition_probs
        self.masks = masks

    def _calculate_transition_prob(self, current, move):
        """Determine the outcome for an action. Transition Prob is always 1.0.
        
        The original transition probabilities for absorbing states are not correct.
        """
        if not self.env.get_wrapper_attr('is_slippery'):
            deltas = [POSITION_MAPPING[move]]
        else:
            deltas = [
                POSITION_MAPPING[act] for act in [(move - 1) % 4, move, (move + 1) % 4]
            ]
        outcomes = []

        # the single absorbing state is the goal
        goal_position = np.asarray([self.shape[0] - 1, self.shape[1] - 1])
        goal_state = np.ravel_multi_index(goal_position, self.shape)
        current_position = np.array(current)
        current_state = np.ravel_multi_index(tuple(current_position), self.shape)
        for delta in deltas:
            if current_state == goal_state:
                new_state = current_state
                reward = 0
                is_terminated = True
            else:
                new_position = current_position + np.array(delta)
                new_position = self.env.get_wrapper_attr('_limit_coordinates')(new_position).astype(int)
                new_state = np.ravel_multi_index(tuple(new_position), self.shape)
                if self.env.get_wrapper_attr('_cliff')[tuple(new_position)]:
                    reward = -100
                    new_state = self.env.get_wrapper_attr('start_state_index')
                else:
                    reward = -1
                is_terminated = (new_state == goal_state)
            outcomes.append((1 / len(deltas), new_state, reward, is_terminated))
        return outcomes

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.env.set_wrapper_attr('start_state_index', obs)

        return obs, info

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = super().step(action)
        reward = (orig_reward - self._orig_reward_min) / (self._orig_reward_max - self._orig_reward_min)

        return obs, reward, terminated, truncated, info


class AugmentedCliffWalkingEnv(CliffWalkingEnv):
    def __init__(self, discount=0.99, random_init_state=False, max_episode_steps=1000,
                 render_mode="rgb_array", **kwargs):
        self.discount = discount

        super().__init__(random_init_state, max_episode_steps, render_mode=render_mode, **kwargs)

        # add s+ and s- into the state space
        self.nS += 2
        self.observation_space = gymnasium.spaces.Discrete(self.nS)

        rewards = np.full((self.nS, self.nA, self.nS), -1.0)
        transition_probs = np.zeros((self.nS, self.nA, self.nS))
        masks = np.zeros((self.nS, self.nA, self.nS))

        # transiting into s+ and staying at s+ give 0.0 reward, otherwise the reward is -1.
        # rewards[..., self.nS - 2] = 0.0
        rewards[self.nS - 2] = 0.0
        assert np.all((-1.0 <= rewards) & (rewards <= 1.0))

        transition_probs[:self.nS - 2, :, :self.nS - 2] = discount * self.transition_probs
        transition_probs[:self.nS - 2, :, self.nS - 2] = (1.0 - discount) * np.sum(
            self.rewards * self.transition_probs, axis=-1)
        transition_probs[:self.nS - 2, :, self.nS - 1] = (1.0 - discount) * (1.0 - np.sum(
            self.rewards * self.transition_probs, axis=-1))
        transition_probs[self.nS - 2, :, self.nS - 2] = 1.0
        transition_probs[self.nS - 1, :, self.nS - 1] = 1.0
        assert np.all(np.sum(transition_probs, axis=-1) == 1.0)

        # (chongyi): the original absorbing states becomes non-absorbing states
        masks[:self.nS - 2, :, :self.nS - 2] = np.ones_like(self.masks)
        masks[..., self.nS - 2] = 0.0
        masks[..., self.nS - 1] = 0.0
        masks[self.nS - 2] = 0.0
        masks[self.nS - 1] = 0.0

        self.aug_rewards = rewards
        self.aug_transition_probs = transition_probs
        self.aug_masks = masks

        self.last_aug_s = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_aug_s = obs

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        aug_reward = -1.0
        aug_info = {"prob": 1}
        aug_terminated = True
        if self.last_aug_s == self.nS - 2:
            aug_obs = int(self.nS - 2)
            aug_reward = 0.0
        elif self.last_aug_s == self.nS - 1:
            aug_obs = int(self.nS - 1)
        else:
            if np.random.rand() < self.discount:
                aug_obs = obs
                aug_info = info
                aug_terminated = False
            else:
                if np.random.rand() < reward:
                    aug_obs = int(self.nS - 2)
                else:
                    aug_obs = int(self.nS - 1)
        self.last_aug_s = aug_obs

        return aug_obs, aug_reward, aug_terminated, truncated, aug_info


def main():
    discount = 0.95
    max_episode_steps = 100

    env = CliffWalkingEnv(random_init_state=True, max_episode_steps=max_episode_steps)
    # env.reset()[0]
    # img = env.render()

    # value iteration to find the optimal Q
    # assert np.all(env.rewards == ((env.orig_rewards + 100) / (-1 + 100)))
    rewards = env.rewards
    transition_probs = env.transition_probs
    masks = env.masks

    opt_q = np.zeros([env.nS, env.nA], dtype=np.float32)
    for _ in range(10_000):
        opt_q = (
            (1 - discount) * np.sum(transition_probs * rewards, axis=-1) +
            discount * np.einsum('ijk,k->ij', transition_probs, np.max(opt_q, axis=-1))
        )
    # deterministic optimal policy
    opt_policy = np.zeros([env.nS, env.nA])
    opt_policy[np.arange(env.nS), np.argmax(opt_q, axis=-1)] = 1.0

    print(np.sum(rewards * transition_probs, axis=-1))

    # value iteration to find the optimal Q in the original MDP using gamma ** 2
    opt_q_square_discount = np.zeros([env.nS, env.nA], dtype=np.float32)
    for _ in range(10_000):
        opt_q_square_discount = (
            (1 - discount ** 2) * np.sum(transition_probs * rewards, axis=-1)
            + discount ** 2 * np.einsum('ijk,k->ij', transition_probs, np.max(opt_q_square_discount, axis=-1))
        )

    aug_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_steps=max_episode_steps)
    # value iteration to find the optimal discounted state occupancy measure in the GCMDP using gamma
    rewards = aug_env.aug_rewards
    transition_probs = aug_env.aug_transition_probs
    masks = aug_env.aug_masks

    opt_d_sa = np.zeros([aug_env.nS, aug_env.nA, aug_env.nS], dtype=np.float32)
    for _ in range(10_000):
        opt_a = np.argmax(opt_d_sa[..., aug_env.nS - 2], axis=-1)
        opt_pi = np.zeros([aug_env.nS, aug_env.nA])
        opt_pi[np.arange(aug_env.nS), opt_a] = 1.
        opt_d_sa = (1.0 - discount) * np.eye(aug_env.nS)[:, None] + discount * np.einsum(
            'ijk,kl->ijl', transition_probs, np.sum(opt_d_sa * opt_pi[..., None], axis=1))
    scaled_opt_d_sa_splus = opt_d_sa[:aug_env.nS - 2, :, aug_env.nS - 2] * (1 + discount) / discount
    assert np.allclose(scaled_opt_d_sa_splus, opt_q_square_discount)

    # value iteration to compute the optimal single goal-conditioned Q in the GCMDP using gamma
    opt_gc_q = np.zeros([aug_env.nS, aug_env.nA], dtype=np.float32)
    for _ in range(10_000):
        opt_gc_q = (
            (1.0 - discount) * np.sum(rewards * transition_probs, axis=-1)
            + discount * np.einsum('ijk,k->ij', transition_probs, np.max(opt_gc_q, axis=-1))
        )
    # indicators = np.zeros((aug_env.nS, aug_env.nA))
    # indicators[aug_env.nS - 2] = 1.
    # scaled_opt_gc_q = (opt_gc_q[:aug_env.nS - 2] + 1) * discount + indicators * (1 - discount)
    # scaled_opt_gc_q = scaled_opt_gc_q * (1 + discount) / discount
    scaled_opt_gc_q = (opt_gc_q[:aug_env.nS - 2] + 1) * (1 + discount) / discount

    assert np.allclose(scaled_opt_d_sa_splus, scaled_opt_gc_q)
    assert np.allclose(opt_q_square_discount, scaled_opt_gc_q)

    dataset = defaultdict(list)
    num_episodes = 1_000
    num_transitions = 0
    for _ in tqdm.trange(num_episodes):
        obs, info = aug_env.reset()
        for _ in range(max_episode_steps):
            action = aug_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = aug_env.step(action)
            done = terminated or truncated

            num_transitions += 1
            dataset['observations'].append(obs)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['next_observations'].append(next_obs)
            dataset['masks'].append(not terminated)  # for absorbing states
            dataset['terminals'].append(done)  # for the end of trajectory

            obs = next_obs

    for k, v in dataset.items():
        if k in ['observations', 'actions', 'next_observations']:
            dtype = np.int32
        elif k in ['masks', 'terminals']:
            dtype = bool
        else:
            dtype = np.float32
        dataset[k] = np.array(v, dtype=dtype)

    # compute empirical transitions
    rewards = np.zeros((aug_env.nS, aug_env.nA, aug_env.nS))
    transition_probs = np.zeros((aug_env.nS, aug_env.nA, aug_env.nS))
    # for state in range(aug_env.nS):
    #     for action in range(aug_env.nA):
    #         rewards[state, action, next_state] = reward
    #         transition_probs[state, action, next_state] += 1.0
    for i in tqdm.trange(dataset['observations'].shape[0]):
        state = dataset['observations'][i]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        next_state = dataset['next_observations'][i]

        rewards[state, action, next_state] += reward
        transition_probs[state, action, next_state] += 1.0
    rewards /= (transition_probs + 1e-8)
    assert np.all((0.0 <= rewards) & (rewards <= 1.0))
    transition_probs /= np.sum(transition_probs, axis=-1, keepdims=True)
    # assert np.all(np.sum(transition_probs, axis=-1) == 1.0)
    rewards *= transition_probs

    print()


if __name__ == "__main__":
    main()
