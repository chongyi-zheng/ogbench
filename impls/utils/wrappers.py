
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

    agent: flax.struct.PyTreeNode
    mean: np.ndarray
    var: np.ndarray
    epsilon: float = 1e-8

    def _normalize(self, observations):
        if observations is None:
            return None
        else:
            return (observations - self.mean) / np.sqrt(
                self.var + self.epsilon
            )

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        observations = self._normalize(observations)
        goals = self._normalize(goals)

        try:
            return self.agent.sample_actions(observations, goals, seed=seed, temperature=temperature)
        except TypeError as e:
            return self.agent.sample_actions(observations, seed=seed, temperature=temperature)

    def update(self, batch):
        batch['observations'] = self._normalize(batch['observations'])
        batch['next_observations'] = self._normalize(batch['next_observations'])
        if 'value_goals' in batch:
            batch['value_goals'] = self._normalize(batch['value_goals'])
        if 'actor_goals' in batch:
            batch['actor_goals'] = self._normalize(batch['actor_goals'])
        agent, info = self.agent.update(batch)

        return self.replace(agent=agent), info

    @classmethod
    def create(
        cls,
        agent,
        dataset,
        epsilon=1e-8,
    ):
        if hasattr(dataset, 'dataset'):
            observations = dataset.dataset['observations']
        else:
            observations = dataset['observations']
        mean = np.mean(observations, axis=0)
        var = np.var(observations, axis=0)

        return cls(agent=agent, mean=mean, var=var, epsilon=epsilon)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.agent, name)
