from typing import Any
from functools import partial

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
    normalized_max: np.ndarray
    normalized_min: np.ndarray
    epsilon: float = 1e-8

    @staticmethod
    def normalize(observations, obs_mean, obs_var, obs_max, obs_min,
                  normalizer_type='none', epsilon=1e-8):
        if observations is None:
            return None
        else:
            if normalizer_type == 'normal':
                return (observations - obs_mean) / np.sqrt(
                    obs_var + epsilon
                )
            elif normalizer_type == 'bounded':
                return 2 * (observations - obs_min) / (
                    obs_max - obs_min
                ) - 1.0
            elif normalizer_type == 'none':
                return observations
            else:
                raise TypeError("Unsupported normalizer type: {}".format(
                    normalizer_type))

    def normalized_func(self, observations):
        return self.normalize(observations,
                              obs_mean=self.mean, obs_var=self.var,
                              obs_max=self.max, obs_min=self.min,
                              normalizer_type=self.normalizer_type, epsilon=self.epsilon)

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        observations = self.normalized_func(observations)
        goals = self.normalized_func(goals)

        try:
            return self.agent.sample_actions(observations, goals, seed=seed, temperature=temperature)
        except TypeError as e:
            return self.agent.sample_actions(observations, seed=seed, temperature=temperature)

    def pretrain(self, batch):
        batch['observations'] = self.normalized_func(batch['observations'])
        batch['next_observations'] = self.normalized_func(batch['next_observations'])
        batch['observation_min'] = self.normalized_min
        batch['observation_max'] = self.normalized_max
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalized_func(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalized_func(batch['actor_goals'])
        agent, info = self.agent.pretrain(batch)

        return self.replace(agent=agent), info

    def finetune(self, batch, full_update=True):
        batch['observations'] = self.normalized_func(batch['observations'])
        batch['next_observations'] = self.normalized_func(batch['next_observations'])
        batch['observation_min'] = self.normalized_min
        batch['observation_max'] = self.normalized_max
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalized_func(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalized_func(batch['actor_goals'])
        agent, info = self.agent.finetune(batch, full_update=full_update)

        return self.replace(agent=agent), info

    def update(self, batch):
        batch['observations'] = self.normalized_func(batch['observations'])
        batch['next_observations'] = self.normalized_func(batch['next_observations'])
        batch['observation_min'] = self.normalized_min
        batch['observation_max'] = self.normalized_max
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalized_func(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalized_func(batch['actor_goals'])
        agent, info = self.agent.update(batch)

        return self.replace(agent=agent), info

    def total_loss(self, batch, grad_params, rng=None):
        batch['observations'] = self.normalized_func(batch['observations'])
        batch['next_observations'] = self.normalized_func(batch['next_observations'])
        if 'value_goals' in batch:
            batch['value_goals'] = self.normalized_func(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self.normalized_func(batch['actor_goals'])

        return self.agent.total_loss(batch, grad_params, rng)

    def update_dataset(self, dataset):
        if self.normalizer_type == 'none':
            dataset_mean, dataset_var, dataset_max, dataset_min = None, None, None, None
        else:
            if hasattr(dataset, 'dataset'):
                observations = dataset.dataset['observations']
            else:
                observations = dataset['observations']
            dataset_mean = np.mean(observations, axis=0)
            dataset_var = np.var(observations, axis=0)
            dataset_max = np.max(observations, axis=0)
            dataset_min = np.min(observations, axis=0)

        normalized_dataset_max = self.normalize(
            dataset_max, dataset_mean, dataset_var,
            dataset_max, dataset_min,
            self.normalizer_type, self.epsilon
        )
        normalized_dataset_min = self.normalize(
            dataset_min, dataset_mean, dataset_var,
            dataset_max, dataset_min,
            self.normalizer_type, self.epsilon
        )

        return self.replace(
            mean=dataset_mean, var=dataset_var,
            max=dataset_max, min=dataset_min,
            normalized_max=normalized_dataset_max, normalized_min=normalized_dataset_min,
        )

    @classmethod
    def create(
        cls,
        agent,
        dataset,
        normalizer_type='normal',
        epsilon=1e-8,
    ):
        if normalizer_type == 'none':
            dataset_mean, dataset_var, dataset_max, dataset_min = None, None, None, None
        else:
            if hasattr(dataset, 'dataset'):
                observations = dataset.dataset['observations']
            else:
                observations = dataset['observations']
            dataset_mean = np.mean(observations, axis=0)
            dataset_var = np.var(observations, axis=0)
            dataset_max = np.max(observations, axis=0)
            dataset_min = np.min(observations, axis=0)

        normalized_dataset_max = cls.normalize(
            dataset_max, dataset_mean, dataset_var,
            dataset_max, dataset_min,
            normalizer_type, epsilon
        )
        normalized_dataset_min = cls.normalize(
            dataset_min, dataset_mean, dataset_var,
            dataset_max, dataset_min,
            normalizer_type, epsilon
        )

        return cls(agent=agent, normalizer_type=normalizer_type,
                   mean=dataset_mean, var=dataset_var, max=dataset_max, min=dataset_min,
                   normalized_max=normalized_dataset_max, normalized_min=normalized_dataset_min,
                   epsilon=epsilon)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.agent, name)
