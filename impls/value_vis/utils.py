from collections import defaultdict

import flax.linen as nn
import numpy as np
from tqdm import tqdm


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def evaluate(actor_fn, env, params, num_eval_episodes=20, desc='evaluation'):
    stats = defaultdict(list)
    for _ in tqdm(range(num_eval_episodes), desc=desc, position=0, leave=True):
        observation, info = env.reset()
        done = False
        episode_length = 0
        reward_sum = 0.0

        while not done:
            action = actor_fn(params=params, observations=observation)
            action = np.array(action)
            assert env.action_space.contains(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward_sum += reward
            episode_length += 1

            observation = next_observation

        stats['episode_length'].append(episode_length)
        stats['episode_return'].append(reward_sum)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
