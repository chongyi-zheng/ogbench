from typing import Any

import numpy as np
import flax


class PositiveRewardShaping(flax.struct.PyTreeNode):
    agent: flax.struct.PyTreeNode
    min_reward: float
    epsilon: float = 1e-6

    def _shaping(self, rewards):
        return rewards - self.min_reward + self.epsilon

    def update(self, batch):
        batch['rewards'] = self._shaping(batch['rewards'])
        agent, info = self.agent.update(batch)

        return self.replace(agent=agent), info

    @classmethod
    def create(
        cls,
        agent,
        dataset,
        epsilon=1e-6,
    ):
        if hasattr(dataset, 'dataset'):
            rewards = dataset.dataset['rewards']
        else:
            rewards = dataset['rewards']
        min_reward = np.min(rewards)

        return cls(agent=agent, min_reward=min_reward, epsilon=epsilon)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.agent, name)


class OfflineObservationNormalizer(flax.struct.PyTreeNode):
    """
    This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    """

    agent: Any
    normalizer_type: str
    mean: np.ndarray
    var: np.ndarray
    max: np.ndarray
    min: np.ndarray
    epsilon: float = 1e-8

    def normalize(self, observations):
        if observations is None:
            return None
        else:
            if self.normalizer_type == 'normal':
                return (observations - self.mean) / np.sqrt(
                    self.var + self.epsilon
                )
            elif self.normalizer_type == 'bounded':
                return 2 * (observations - self.min) / (
                    self.max - self.min
                ) - 1.0
            else:
                raise TypeError("Unsupported normalizer type: {}".format(
                    self.normalizer_type))

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        observations = self.normalize(observations)
        goals = self.normalize(goals)

        try:
            return self.agent.sample_actions(observations, goals, seed=seed, temperature=temperature)
        except TypeError as e:
            return self.agent.sample_actions(observations, seed=seed, temperature=temperature)

    def update(self, batch):
        batch['observations'] = self.normalize(batch['observations'])
        batch['next_observations'] = self.normalize(batch['next_observations'])
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalize(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalize(batch['actor_goals'])
        agent, info = self.agent.update(batch)

        return self.replace(agent=agent), info

    def total_loss(self, batch, grad_params, rng=None):
        batch['observations'] = self.normalize(batch['observations'])
        batch['next_observations'] = self.normalize(batch['next_observations'])
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalize(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalize(batch['actor_goals'])

        return self.agent.total_loss(batch, grad_params, rng)

    @classmethod
    def create(
        cls,
        agent,
        dataset,
        normalizer_type='normal',
        epsilon=1e-8,
    ):
        if hasattr(dataset, 'dataset'):
            observations = dataset.dataset['observations']
        else:
            observations = dataset['observations']
        dataset_mean = np.mean(observations, axis=0)
        dataset_var = np.var(observations, axis=0)
        dataset_max = np.max(observations, axis=0)
        dataset_min = np.min(observations, axis=0)

        return cls(agent=agent, normalizer_type=normalizer_type,
                   mean=dataset_mean, var=dataset_var, max=dataset_max, min=dataset_min,
                   epsilon=epsilon)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.agent, name)
