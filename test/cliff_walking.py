from collections import defaultdict

import numpy as np
import gymnasium
import tqdm


class CliffWalkingEnv(gymnasium.Wrapper):
    def __init__(self, random_init_state=False, max_episode_step=1000, **kwargs):
        env = gymnasium.make(
            "CliffWalking-v1",
            max_episode_steps=max_episode_step,
            **kwargs
        )

        self.nS = env.get_wrapper_attr('nS')
        self.nA = env.get_wrapper_attr('nA')
        self.shape = env.get_wrapper_attr('shape')
        if random_init_state:
            initial_state_distrib = np.ones(self.nS)
            cliff = np.zeros(self.shape, dtype=bool)
            cliff[3, 1:-1] = True
            cliff_positions = np.asarray(np.where(cliff))
            cliff_states = np.ravel_multi_index(cliff_positions, self.shape)
            initial_state_distrib[cliff_states] = 0.0
            initial_state_distrib[47] = 0.0
            env.unwrapped.initial_state_distrib = initial_state_distrib

        rewards = np.full((self.nS, self.nA, self.nS), np.nan)
        transition_probs = np.zeros((self.nS, self.nA, self.nS))
        for state in range(self.nS):
            for action in range(self.nA):
                _, next_state, reward, _ = env.get_wrapper_attr('P')[state][action][0]
                rewards[state, action, next_state] = reward
                transition_probs[state, action, next_state] += 1.0
        transition_probs /= np.sum(transition_probs, axis=-1, keepdims=True)
        assert np.all(np.sum(transition_probs, axis=-1) == 1.0)
        reward_max, reward_min = np.nanmax(rewards), np.nanmin(rewards)
        rewards[np.isnan(rewards)] = reward_min
        assert np.all((reward_min <= rewards) & (rewards <= reward_max))

        self._orig_reward_min, self._orig_reward_max = reward_min, reward_max
        self.orig_rewards = rewards
        self.rewards = (rewards - reward_min) / (reward_max - reward_min)
        self.transition_probs = transition_probs

        super().__init__(env)

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = super().step(action)
        reward = (orig_reward - self._orig_reward_min) / (self._orig_reward_max - self._orig_reward_min)

        return obs, reward, terminated, truncated, info


class AugmentedCliffWalkingEnv(CliffWalkingEnv):
    def __init__(self, discount=0.99, random_init_state=False, max_episode_step=1000, **kwargs):
        self.discount = discount

        super().__init__(random_init_state, max_episode_step, **kwargs)

        # add s+ and s- into the state space
        self.nS += 2
        self.observation_space = gymnasium.spaces.Discrete(self.nS)

        rewards = np.zeros((self.nS, self.nA, self.nS))
        transition_probs = np.zeros((self.nS, self.nA, self.nS))

        # transit into s+ gives 1.0 reward, otherwise the reward is 0.
        rewards[..., self.nS - 2] = 1.0
        assert np.all((0.0 <= rewards) & (rewards <= 1.0))

        transition_probs[:self.nS - 2, :, :self.nS - 2] = discount * self.transition_probs
        transition_probs[:self.nS - 2, :, self.nS - 2] = (1.0 - discount) * np.sum(
            self.rewards * self.transition_probs, axis=-1)
        transition_probs[:self.nS - 2, :, self.nS - 1] = (1.0 - discount) * (1.0 - np.sum(
            self.rewards * self.transition_probs, axis=-1))
        transition_probs[self.nS - 2, :, self.nS - 2] = 1.0
        transition_probs[self.nS - 1, :, self.nS - 1] = 1.0
        assert np.all(np.sum(transition_probs, axis=-1) == 1.0)

        self.aug_rewards = rewards
        self.aug_transition_probs = transition_probs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if np.random.rand() < self.discount:
            aug_obs = obs
            aug_reward = 0.0
            aug_terminated = terminated
        else:
            if np.random.rand() < reward:
                aug_obs = self.nS - 2
                aug_reward = 1.0
                aug_terminated = True
            else:
                aug_obs = self.nS - 1
                aug_reward = 0.0
                aug_terminated = True

        return aug_obs, aug_reward, aug_terminated, truncated, info


def main():
    discount = 0.95
    max_episode_step = 1000

    env = CliffWalkingEnv(random_init_state=True, max_episode_step=max_episode_step)
    aug_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_step=max_episode_step)

    aug_rewards = aug_env.aug_rewards
    aug_transition_probs = aug_env.aug_transition_probs

    dataset = defaultdict(list)
    num_episodes = 1_001
    num_transitions = 0
    for _ in tqdm.trange(num_episodes):
        done = False
        obs, info = aug_env.reset()
        while not done:
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
    transition_probs /= np.sum(transition_probs, axis=-1, keepdims=True)
    assert np.all(np.abs(transition_probs[:37] - aug_transition_probs[:37]) < 5e-2)
    # assert np.all(np.sum(transition_probs, axis=-1) == 1.0)
    rewards *= transition_probs
    assert np.all((0 <= rewards) & (rewards <= 1))

    print()


if __name__ == "__main__":
    main()
