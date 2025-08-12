#!/usr/bin/env python3
"""
Converted from mdp2gcmdp_CliffWalking_online.ipynb
Online Reinforcement Learning on CliffWalking Environment
Implements Q-Learning, CRL + Binary NCE, and GC TD InfoNCE algorithms
"""

import copy
import os
from collections import defaultdict

import gymnasium
import numpy as np
import tqdm
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from gymnasium.envs.toy_text.cliffwalking import (
    UP, RIGHT, DOWN, LEFT, POSITION_MAPPING
)

# Set matplotlib backend for non-interactive plotting
plt.ioff()

# Create output directory for figures
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
print(f"Saving figures to: {FIGURE_DIR}")


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
                reward = 100
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


# Experience Replay Buffer for Online Learning
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = defaultdict(list)
        self.position = 0
        self.size = 0
    
    def push(self, obs, action, reward, next_obs, mask, terminal):
        if self.size < self.capacity:
            # Add new transitions
            self.buffer['observations'].append(obs)
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['next_observations'].append(next_obs)
            self.buffer['masks'].append(mask)
            self.buffer['terminals'].append(terminal)
            self.size += 1
        else:
            # Overwrite old transitions (circular buffer)
            self.buffer['observations'][self.position] = obs
            self.buffer['actions'][self.position] = action
            self.buffer['rewards'][self.position] = reward
            self.buffer['next_observations'][self.position] = next_obs
            self.buffer['masks'][self.position] = mask
            self.buffer['terminals'][self.position] = terminal
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        
        idxs = np.random.randint(self.size, size=batch_size)
        batch = {}
        for k in self.buffer.keys():
            if k in ['observations', 'actions', 'next_observations']:
                dtype = np.int32
            elif k == 'terminals':
                dtype = bool
            else:
                dtype = np.float32
            batch[k] = np.array([self.buffer[k][i] for i in idxs], dtype=dtype)
        return batch
    
    def __len__(self):
        return self.size


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

        # augmented transitions
        aug_info = {"prob": 1}
        aug_terminated = True
        if self.last_aug_s == self.nS - 2:
            aug_obs = int(self.nS - 2)
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
        aug_reward = -float(aug_obs == (self.nS - 2))

        return aug_obs, aug_reward, aug_terminated, truncated, aug_info


# Augmented Replay Buffer for Augmented Environment
class AugmentedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
    
    def sample_with_goals(self, batch_size, p_curgoal=0.2, p_trajgoal=0.5, relabel_reward=False):
        """Sample batch with goal relabeling for goal-conditioned RL"""
        if self.size < batch_size:
            return None
        
        # Get terminal locations for trajectory boundaries
        terminal_array = np.array(self.buffer['terminals'][:self.size])
        terminal_locs = np.where(terminal_array)[0]
        if len(terminal_locs) == 0:
            return self.sample(batch_size)  # fallback to regular sampling
        
        idxs = np.random.randint(self.size, size=batch_size)
        batch = {}
        for k in self.buffer.keys():
            if k in ['observations', 'actions', 'next_observations']:
                dtype = np.int32
            elif k == 'terminals':
                dtype = bool
            else:
                dtype = np.float32
            batch[k] = np.array([self.buffer[k][i] for i in idxs], dtype=dtype)
        
        # Goal relabeling logic
        final_state_idxs = terminal_locs[np.searchsorted(terminal_locs, idxs, side='right') - 1]
        final_state_idxs = np.clip(final_state_idxs, 0, self.size - 1)
        
        # Generate different types of goals
        offsets = np.random.geometric(p=1 - discount, size=batch_size)
        traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        random_goal_idxs = np.random.randint(self.size, size=batch_size)
        
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), 
            traj_goal_idxs, random_goal_idxs
        )
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        
        batch['goals'] = np.array([self.buffer['observations'][i] for i in goal_idxs], dtype=np.int32)
        
        if relabel_reward:
            successes = (idxs == goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - 1.0  # 0 for goal and -1 for other states
        
        return batch


# Online Experience Collection and Exploration
def collect_experience(env, policy, buffer, num_episodes, epsilon=0.1, use_epsilon_greedy=True):
    """Collect experience using epsilon-greedy exploration"""
    collected_transitions = 0
    successful_episodes = 0
    
    # if max_ep_length is None:
    #     max_ep_length = max_episode_steps
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        # episode_length = 0
        
        while not done:
            if use_epsilon_greedy and np.random.rand() < epsilon:
                # Random action for exploration
                action = np.random.randint(env.nA)
            else:
                # Use policy
                if policy is None:
                    action = np.random.randint(env.nA)  # Random policy
                else:
                    action = np.random.choice(np.arange(env.nA), p=policy[obs])
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
            done = truncated
            
            # Store transition in replay buffer
            buffer.push(obs, action, reward, next_obs, not terminated, truncated)
            collected_transitions += 1
            
            # Check if reached goal
            if next_obs == 47:  # Goal state
                successful_episodes += 1
            
            obs = next_obs
            # episode_length += 1
    
    return collected_transitions, successful_episodes


# Function to get epsilon-greedy policy from Q-values
def get_epsilon_greedy_policy(q_values, epsilon=0.1):
    """Convert Q-values to epsilon-greedy policy"""
    nS, nA = q_values.shape
    policy = np.ones((nS, nA)) * epsilon / nA
    best_actions = np.argmax(q_values, axis=1)
    for s in range(nS):
        policy[s, best_actions[s]] += 1.0 - epsilon
    return policy


# Experience collection for augmented environment
def collect_augmented_experience(aug_env, policy, buffer, num_episodes, epsilon=0.1, use_epsilon_greedy=True):
    """Collect experience using epsilon-greedy exploration for augmented environment"""
    collected_transitions = 0
    
    for episode in range(num_episodes):
        obs, info = aug_env.reset()
        done = False
        episode_length = 0
        
        while not done and episode_length < max_episode_steps:
            if use_epsilon_greedy and np.random.rand() < epsilon:
                # Random action for exploration
                action = np.random.randint(aug_env.nA)
            else:
                # Use policy
                if policy is None:
                    action = np.random.randint(aug_env.nA)  # Random policy
                else:
                    action = np.random.choice(np.arange(aug_env.nA), p=policy[obs])
            
            next_obs, reward, terminated, truncated, info = aug_env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            buffer.push(obs, action, reward, next_obs, not terminated, truncated)
            collected_transitions += 1
            
            obs = next_obs
            episode_length += 1
    
    return collected_transitions


def plot_metrics(metrics, logyscale_stats=[], title='', save_path=None):
    """Plot training metrics and save figure"""
    # learning curves
    nrows = np.ceil(len(metrics) / 4).astype(int)
    ncols = 4
    f, axes = plt.subplots(nrows=nrows, ncols=ncols)
    if nrows == 1:
        axes = np.array([axes])
    f.set_figheight(3 * nrows)
    f.set_figwidth(3 * ncols)

    for idx, (name, val) in enumerate(metrics.items()):
        v = np.array(val)
        if len(v) == 0:
            continue

        x, y = v[:, 0], v[:, 1]
        ax = axes[idx // 4, idx % 4]

        if 'train' in name:
            y = gaussian_filter1d(y, 100)
        ax.plot(x, y)
        if name in logyscale_stats:
            ax.set_yscale('log')
        ax.set_title(name)
        ax.grid()

    f.suptitle(title)
    
    if save_path:
        f.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.close(f)  # Close figure to save memory
    return f


### Q-Learning Implementation

class QCritic(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        obs = jax.nn.one_hot(obs, env.nS)
        action = jax.nn.one_hot(action, env.nA)
        inputs = jnp.concatenate([obs, action], axis=-1)

        qs = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(512),
            nn.gelu,
            nn.Dense(1),
        ])(inputs)
        
        qs = qs.squeeze(-1)
        return qs


def run_q_learning():
    """Run Online Q-Learning Algorithm"""
    print("=" * 50)
    print("Running Online Q-Learning")
    print("=" * 50)
    
    # Parameters
    batch_size = 1024
    tau = 0.005
    num_iterations = 100_000  
    eval_interval = 1_000
    log_interval = 1_000
    experience_collection_interval = 10
    experience_episodes_per_collection = 5
    initial_experience_episodes = 1000
    min_buffer_size = 10000

    # Exploration parameters
    initial_epsilon = 1.0
    final_epsilon = 0.1
    epsilon_decay_steps = 30000

    key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    key, critic_key = jax.random.split(key)

    # Initialize critic with dummy batch
    dummy_obs = jnp.array([0, 1])
    dummy_actions = jnp.array([0, 1])
    critic = QCritic()
    critic_params = critic.init(critic_key, dummy_obs, dummy_actions)
    target_critic_params = copy.deepcopy(critic_params)

    def loss_fn(params, target_params, batch):
        q = critic.apply(params, batch['observations'], batch['actions'])
        
        next_observations = batch['next_observations'][:, None].repeat(env.nA, axis=1).reshape(-1)
        next_actions = jnp.arange(env.nA)[None, :].repeat(len(batch['observations']), axis=0).reshape(-1)
        next_qs = critic.apply(target_params, next_observations, next_actions)
        next_qs = next_qs.reshape([len(batch['observations']), env.nA])
        next_q = jnp.max(next_qs, axis=-1)
        # target_q = batch['rewards'] + discount * batch['masks'] * next_q
        target_q = batch['rewards'] + discount * next_q

        loss = jnp.mean((q - target_q) ** 2)
        
        info = {
            'loss': loss,
            'q': q.mean(),
        }
        
        return loss, info

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(critic_params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def update_fn(params, target_params, opt_state, batch):
        (loss, info), grads = grad_fn(params, target_params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            params, target_params,
        )
        
        return params, target_params, opt_state, loss, info

    def compute_success_rate(params, num_eval_episodes=100, max_episode_steps=100):
        eval_env = CliffWalkingEnv(random_init_state=True, max_episode_steps=max_episode_steps)
        aug_eval_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_steps=max_episode_steps)
        
        # compute the pi
        obs = jnp.arange(aug_eval_env.nS)[:, None].repeat(aug_eval_env.nA, axis=1).reshape(-1)
        actions = jnp.arange(aug_eval_env.nA)[None, :].repeat(aug_eval_env.nS, axis=0).reshape(-1)
        q = critic.apply(params, obs, actions)
        q = q.reshape([aug_eval_env.nS, aug_eval_env.nA])
        a = jnp.argmax(q, axis=-1)
        pi = jax.nn.one_hot(a, aug_eval_env.nA)
        pi = np.asarray(pi)

        # evaluation on original environment
        successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = eval_env.reset()
            # episode_length = 0
            while not done:
                action = np.random.choice(np.arange(eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                # episode_length += 1
                if obs == 47:  # reached goal
                    successes.append(True)
                    break
            else:
                successes.append(False)
        sr = np.mean(successes)

        # evaluation on augmented environment
        aug_successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = aug_eval_env.reset()
            # episode_length = 0
            while not done:
                action = np.random.choice(np.arange(aug_eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = aug_eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                # episode_length += 1
                if obs == aug_eval_env.nS - 2:  # reached s+
                    aug_successes.append(True)
                    break
            else:
                aug_successes.append(False)
        aug_sr = np.mean(aug_successes)

        return sr, aug_sr

    # Initial experience collection with random policy
    print("Collecting initial experience with random policy...")
    collect_experience(env, None, replay_buffer, initial_experience_episodes, epsilon=1.0)
    print(f"Collected {len(replay_buffer)} initial transitions")

    metrics = defaultdict(list)
    current_epsilon = initial_epsilon

    print("Starting online Q-Learning training...")
    for i in tqdm.trange(1, num_iterations + 1):
        # Update epsilon for exploration
        if i <= epsilon_decay_steps:
            current_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * i / epsilon_decay_steps
        else:
            current_epsilon = final_epsilon
        
        # Collect new experience periodically
        if i % experience_collection_interval == 0 and len(replay_buffer) >= min_buffer_size:
            # Get current Q-values for policy
            obs = jnp.arange(env.nS)[:, None].repeat(env.nA, axis=1).reshape(-1)
            actions = jnp.arange(env.nA)[None, :].repeat(env.nS, axis=0).reshape(-1)
            q = critic.apply(critic_params, obs, actions)
            q = q.reshape([env.nS, env.nA])
            current_policy = get_epsilon_greedy_policy(np.array(q), current_epsilon)
            
            # Collect experience
            collect_experience(env, current_policy, replay_buffer,
                               experience_episodes_per_collection,
                               epsilon=current_epsilon)
        
        # Training step
        if len(replay_buffer) >= min_buffer_size:
            batch = replay_buffer.sample(batch_size)
            if batch is not None:
                critic_params, target_critic_params, opt_state, loss, info = update_fn(
                    critic_params, target_critic_params, opt_state, batch)

                for k, v in info.items():
                    metrics['train/' + k].append(np.array([i, v]))
                
                metrics['train/epsilon'].append(np.array([i, current_epsilon]))
                metrics['train/buffer_size'].append(np.array([i, len(replay_buffer)]))

        if i == 1 or i % eval_interval == 0:
            if len(replay_buffer) >= min_buffer_size:
                sr, aug_sr = compute_success_rate(critic_params)
                metrics['eval/success_rate'].append(np.array([i, sr]))
                metrics['eval/augmented_success_rate'].append(np.array([i, aug_sr]))

        if i == 1 or i % log_interval == 0:
            save_path = os.path.join(FIGURE_DIR, f"q_learning_metrics.png")
            plot_metrics(metrics, logyscale_stats=[], title="Q-Learning Training Progress", save_path=save_path)

    return metrics


### CRL + Binary NCE Implementation

class CRLCritic(nn.Module):
    repr_dim: int = 512

    @nn.compact
    def __call__(self, obs, action, future_obs):
        obs = jax.nn.one_hot(obs, aug_env.nS)
        action = jax.nn.one_hot(action, aug_env.nA)
        future_obs = jax.nn.one_hot(future_obs, aug_env.nS)
        phi_inputs = jnp.concatenate([obs, action], axis=-1)
        psi_inputs = future_obs

        phi = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(512),
            nn.gelu,
            nn.Dense(self.repr_dim),
        ])(phi_inputs)
        psi = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(512),
            nn.gelu,
            nn.Dense(self.repr_dim),
        ])(psi_inputs)
        
        logits = jnp.einsum('ik,jk->ij', phi, psi)
        logits = logits / jnp.sqrt(self.repr_dim)
        
        return logits


def estimate_goal_marginals(aug_buffer, aug_env_nS):
    """Estimate goal marginals from current replay buffer"""
    if len(aug_buffer) < 1000:
        # Use uniform distribution if not enough data
        return jnp.ones(aug_env_nS) / aug_env_nS
    
    # Sample a large batch to estimate marginals
    sample_size = min(50000, len(aug_buffer))
    batch = aug_buffer.sample_with_goals(sample_size, p_curgoal=0.0, p_trajgoal=1.0)
    
    goal_marg = np.zeros(aug_env_nS)
    if batch is not None and 'goals' in batch:
        for state in range(aug_env_nS):
            goal_marg[state] = np.sum(batch['goals'] == state) / len(batch['goals'])
    else:
        goal_marg = np.ones(aug_env_nS) / aug_env_nS
    
    return jnp.asarray(goal_marg)


def run_crl_binary_nce():
    """Run Online CRL + Binary NCE Algorithm"""
    print("=" * 50)
    print("Running Online CRL + Binary NCE")
    print("=" * 50)
    
    # Parameters
    batch_size = 1024
    tau = 0.005
    num_iterations = 100_000
    eval_interval = 1_000
    log_interval = 1_000
    experience_collection_interval = 10
    experience_episodes_per_collection = 5
    initial_experience_episodes = 1000
    min_buffer_size = 10000

    # Exploration parameters
    initial_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_decay_steps = 30000

    key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    key, critic_key = jax.random.split(key)

    # Initialize critic with dummy batch
    dummy_obs = jnp.array([0, 1])
    dummy_actions = jnp.array([0, 1])
    dummy_future_obs = jnp.array([0, 1])
    critic = CRLCritic(repr_dim=32)
    critic_params = critic.init(critic_key, dummy_obs, dummy_actions, dummy_future_obs)

    def loss_fn(params, batch, goal_marg):
        logits = critic.apply(
            params, batch['observations'], batch['actions'], batch['goals'])
        
        I = jnp.eye(len(batch['observations']))
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=I).mean()
        
        plus_logits = critic.apply(params, batch['observations'], batch['actions'], 
                                   jnp.full_like(batch['observations'], (aug_env.nS - 2)))
        plus_probs = jnp.exp(jnp.diag(plus_logits)) * goal_marg[aug_env.nS - 2]
        q = (1 + discount) * plus_probs
        
        info = {
            'loss': loss,
            'q': q.mean(),
        }
        
        return loss, info

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(critic_params)

    def update_fn(params, opt_state, batch, goal_marg):
        def _loss_fn(params):
            return loss_fn(params, batch, goal_marg)
        
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, info

    def compute_success_rate_crl(params, goal_marg, num_eval_episodes=100, max_episode_steps=100):
        eval_env = CliffWalkingEnv(random_init_state=True, max_episode_steps=max_episode_steps)
        aug_eval_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_steps=max_episode_steps)
        
        # compute the pi
        obs = jnp.arange(aug_eval_env.nS)[:, None].repeat(aug_eval_env.nA, axis=1).reshape(-1)
        actions = jnp.arange(aug_eval_env.nA)[None, :].repeat(aug_eval_env.nS, axis=0).reshape(-1)
        plus_logits = critic.apply(params, obs, actions, 
                                   jnp.full_like(obs, (aug_eval_env.nS - 2)))
        plus_logits = jnp.diag(plus_logits)
        plus_logits = plus_logits.reshape([aug_eval_env.nS, aug_eval_env.nA])
        a = jnp.argmax(plus_logits, axis=-1)
        pi = jax.nn.one_hot(a, aug_eval_env.nA)
        pi = np.asarray(pi)

        # evaluation on original environment
        successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = eval_env.reset()
            episode_length = 0
            while not done and episode_length < max_episode_steps:
                action = np.random.choice(np.arange(eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                episode_length += 1
                if obs == 47:  # reached goal
                    successes.append(True)
                    break
            else:
                successes.append(False)
        sr = np.mean(successes)
        
        # evaluation on augmented environment
        aug_successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = aug_eval_env.reset()
            episode_length = 0
            while not done and episode_length < max_episode_steps:
                action = np.random.choice(np.arange(aug_eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = aug_eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                episode_length += 1
                if obs == aug_eval_env.nS - 2:  # reached s+
                    aug_successes.append(True)
                    break
            else:
                aug_successes.append(False)
        aug_sr = np.mean(aug_successes)

        return sr, aug_sr

    # Initial experience collection with random policy
    print("Collecting initial experience with random policy for CRL...")
    collect_augmented_experience(aug_env, None, aug_replay_buffer, initial_experience_episodes, epsilon=1.0)
    print(f"Collected {len(aug_replay_buffer)} initial transitions")

    metrics = defaultdict(list)
    current_epsilon = initial_epsilon
    current_goal_marg = estimate_goal_marginals(aug_replay_buffer, aug_env.nS)

    print("Starting online CRL + Binary NCE training...")
    for i in tqdm.trange(1, num_iterations + 1):
        # Update epsilon for exploration
        if i <= epsilon_decay_steps:
            current_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * i / epsilon_decay_steps
        else:
            current_epsilon = final_epsilon
        
        # Collect new experience periodically
        if i % experience_collection_interval == 0 and len(aug_replay_buffer) >= min_buffer_size:
            # Get current policy for experience collection
            obs = jnp.arange(aug_env.nS)[:, None].repeat(aug_env.nA, axis=1).reshape(-1)
            actions = jnp.arange(aug_env.nA)[None, :].repeat(aug_env.nS, axis=0).reshape(-1)
            plus_logits = critic.apply(critic_params, obs, actions, 
                                       jnp.full_like(obs, (aug_env.nS - 2)))
            plus_logits = jnp.diag(plus_logits)
            plus_logits = plus_logits.reshape([aug_env.nS, aug_env.nA])
            current_policy = get_epsilon_greedy_policy(np.array(plus_logits), current_epsilon)
            
            # Collect experience
            collect_augmented_experience(aug_env, current_policy, aug_replay_buffer, 
                                        experience_episodes_per_collection, 
                                        epsilon=current_epsilon)
            
            # Update goal marginals
            current_goal_marg = estimate_goal_marginals(aug_replay_buffer, aug_env.nS)
        
        # Training step
        if len(aug_replay_buffer) >= min_buffer_size:
            batch = aug_replay_buffer.sample_with_goals(batch_size, p_curgoal=0.0, p_trajgoal=1.0)
            if batch is not None:
                critic_params, opt_state, loss, info = update_fn(
                    critic_params, opt_state, batch, current_goal_marg)

                for k, v in info.items():
                    metrics['train/' + k].append(np.array([i, v]))
                
                metrics['train/epsilon'].append(np.array([i, current_epsilon]))
                metrics['train/buffer_size'].append(np.array([i, len(aug_replay_buffer)]))

        if i == 1 or i % eval_interval == 0:
            if len(aug_replay_buffer) >= min_buffer_size:
                sr, aug_sr = compute_success_rate_crl(critic_params, current_goal_marg)
                metrics['eval/success_rate'].append(np.array([i, sr]))
                metrics['eval/augmented_success_rate'].append(np.array([i, aug_sr]))

        if i == 1 or i % log_interval == 0:
            save_path = os.path.join(FIGURE_DIR, f"crl_binary_nce_metrics_iter.png")
            plot_metrics(metrics, logyscale_stats=[], title="CRL + Binary NCE Training Progress", save_path=save_path)

    return metrics


### GC TD InfoNCE Implementation

class LogParam(nn.Module):
    """Scalar parameter module with log scale."""
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_value


class InfoNCECritic(nn.Module):
    repr_dim: int = 512
    log_temp: LogParam = LogParam()

    @nn.compact
    def __call__(self, obs, action, goals, future_obs):
        obs = jax.nn.one_hot(obs, aug_env.nS)
        action = jax.nn.one_hot(action, aug_env.nA)
        goals = jax.nn.one_hot(goals, aug_env.nS)
        future_obs = jax.nn.one_hot(future_obs, aug_env.nS)
        phi_inputs = jnp.concatenate([obs, action, goals], axis=-1)
        psi_inputs = future_obs

        phi = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(512),
            nn.gelu,
            nn.Dense(self.repr_dim),
        ])(phi_inputs)
        psi = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(512),
            nn.gelu,
            nn.Dense(self.repr_dim),
        ])(psi_inputs)
        
        phi = phi / jnp.linalg.norm(phi, axis=-1, keepdims=True) * jnp.sqrt(self.repr_dim)
        psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True) * jnp.sqrt(self.repr_dim)
        logits = jnp.einsum('ik,jk->ij', phi, psi) / jnp.exp(self.log_temp())
        
        return logits


def run_gc_td_infonce():
    """Run Online GC TD InfoNCE Algorithm"""
    print("=" * 50)
    print("Running Online GC TD InfoNCE")
    print("=" * 50)
    
    # Parameters
    batch_size = 1024
    tau = 0.005
    num_iterations = 100_000
    eval_interval = 1_000
    log_interval = 1_000
    experience_collection_interval = 10
    experience_episodes_per_collection = 5
    initial_experience_episodes = 1000
    min_buffer_size = 10000

    # Exploration parameters
    initial_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_decay_steps = 30000

    key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    key, critic_key = jax.random.split(key)

    # Initialize critic with dummy batch
    dummy_obs = jnp.array([0, 1])
    dummy_actions = jnp.array([0, 1])
    dummy_goals = jnp.array([0, 1])
    dummy_future_obs = jnp.array([0, 1])
    critic = InfoNCECritic(repr_dim=32)
    critic_params = critic.init(critic_key, dummy_obs, dummy_actions, dummy_goals, dummy_future_obs)
    target_critic_params = copy.deepcopy(critic_params)

    def loss_fn(params, target_params, batch, s_plus_marg):
        logits = critic.apply(
            params, batch['observations'], batch['actions'], batch['goals'], batch['next_observations'])
        
        random_states = jnp.roll(batch['next_observations'], -1, axis=0)
        random_logits = critic.apply(
            params, batch['observations'], batch['actions'], batch['goals'], random_states)
        
        I = jnp.eye(len(batch['observations']))
        logits = I * logits + (1 - I) * random_logits
        cur_loss = optax.softmax_cross_entropy(logits=logits, labels=I)
        
        # Compute next actions for TD target
        next_observations = batch['next_observations'][:, None].repeat(aug_env.nA, axis=1).reshape(-1)
        next_actions = jnp.arange(aug_env.nA)[None, :].repeat(len(batch['observations']), axis=0).reshape(-1)
        goals = batch['goals'][:, None].repeat(aug_env.nA, axis=1).reshape(-1)
        
        future_logits = critic.apply(
            params, next_observations, next_actions, goals, random_states)
        plus_logits = critic.apply(params, next_observations, next_actions, 
                                   goals, goals)
        plus_log_probs = jnp.diag(plus_logits) - jax.nn.logsumexp(future_logits, axis=-1)
        plus_log_probs = plus_log_probs.reshape([len(batch['observations']), aug_env.nA])
        next_actions = jnp.argmax(plus_log_probs, axis=-1)
        next_actions = jax.lax.stop_gradient(next_actions)

        w_logits = critic.apply(
            target_params, batch['next_observations'], next_actions, batch['goals'], random_states)
        w = jax.nn.softmax(w_logits, axis=-1)
        w = jax.lax.stop_gradient(w) * len(batch['observations']) * I
        
        future_loss = optax.softmax_cross_entropy(logits=random_logits, labels=w)
        
        loss = (1 - discount) * cur_loss + discount * future_loss
        loss = jnp.mean(loss)
        
        # Compute Q-values for logging
        plus_logits = critic.apply(params, batch['observations'], batch['actions'], 
                                   jnp.full_like(batch['next_observations'], (aug_env.nS - 2)),
                                   jnp.full_like(batch['next_observations'], (aug_env.nS - 2)))
        random_logits = critic.apply(params, batch['observations'], batch['actions'],
                                     jnp.full_like(batch['next_observations'], (aug_env.nS - 2)),
                                     random_states)
        plus_log_probs = jnp.diag(plus_logits) - jax.nn.logsumexp(random_logits, axis=-1)
        probs = jnp.exp(plus_log_probs) * len(batch['observations']) * s_plus_marg
        q = probs * (1 + discount)
        
        info = {
            'loss': loss,
            'cur_loss': cur_loss.mean(),
            'future_loss': future_loss.mean(),
            'w_max': w.max(),
            'q': q.mean(),
        }
        
        return loss, info

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(critic_params)

    def update_fn(params, target_params, opt_state, batch, s_plus_marg):
        def _loss_fn(params):
            return loss_fn(params, target_params, batch, s_plus_marg)
        
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            params, target_params,
        )
        
        return params, target_params, opt_state, loss, info

    def compute_success_rate_infonce(params, batch, num_eval_episodes=100, max_episode_steps=100):
        eval_env = CliffWalkingEnv(random_init_state=True, max_episode_steps=max_episode_steps)
        aug_eval_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_steps=max_episode_steps)
        
        # compute the pi
        obs = jnp.arange(aug_eval_env.nS)[:, None].repeat(aug_eval_env.nA, axis=1).reshape(-1)
        actions = jnp.arange(aug_eval_env.nA)[None, :].repeat(aug_eval_env.nS, axis=0).reshape(-1)
        random_states = jnp.roll(batch['next_observations'], -1, axis=0)
        plus_logits = critic.apply(params, obs, actions, 
                                   jnp.full_like(obs, (aug_eval_env.nS - 2)),
                                   jnp.full_like(obs, (aug_eval_env.nS - 2)))
        random_logits = critic.apply(params, obs, actions, 
                                     jnp.full_like(obs, (aug_eval_env.nS - 2)),
                                     random_states)
        plus_log_probs = jnp.diag(plus_logits) - jax.nn.logsumexp(random_logits, axis=-1)
        plus_log_probs = plus_log_probs.reshape([aug_eval_env.nS, aug_eval_env.nA])
        
        a = jnp.argmax(plus_log_probs, axis=-1)
        pi = jax.nn.one_hot(a, aug_eval_env.nA)
        pi = np.asarray(pi)
        
        # evaluation on original environment
        successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = eval_env.reset()
            episode_length = 0
            while not done and episode_length < max_episode_steps:
                action = np.random.choice(np.arange(eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                episode_length += 1
                if obs == 47:  # reached goal
                    successes.append(True)
                    break
            else:
                successes.append(False)
        sr = np.mean(successes)
        
        # evaluation on augmented environment
        aug_successes = []
        for _ in range(num_eval_episodes):
            done = False
            obs, _ = aug_eval_env.reset()
            episode_length = 0
            while not done and episode_length < max_episode_steps:
                action = np.random.choice(np.arange(aug_eval_env.nA), p=pi[obs])
                next_obs, reward, terminated, truncated, _ = aug_eval_env.step(action)
                done = terminated or truncated
                obs = next_obs
                episode_length += 1
                if obs == aug_eval_env.nS - 2:  # reached s+
                    aug_successes.append(True)
                    break
            else:
                aug_successes.append(False)
        aug_sr = np.mean(aug_successes)

        return sr, aug_sr

    def estimate_s_plus_marginal(aug_buffer, aug_env_nS):
        """Estimate s+ marginal from current replay buffer"""
        if len(aug_buffer) < 100:
            return 0.1  # default value
        
        sample_size = min(10000, len(aug_buffer))
        batch = aug_buffer.sample(sample_size)
        if batch is not None:
            return jnp.sum(batch['next_observations'] == aug_env_nS - 2) / len(batch['next_observations'])
        else:
            return 0.1

    # Initial experience collection with random policy
    print("Collecting initial experience with random policy for InfoNCE...")
    collect_augmented_experience(aug_env, None, aug_replay_buffer, initial_experience_episodes, epsilon=1.0)
    print(f"Collected {len(aug_replay_buffer)} initial transitions")

    metrics = defaultdict(list)
    current_epsilon = initial_epsilon

    print("Starting online GC TD InfoNCE training...")
    for i in tqdm.trange(1, num_iterations + 1):
        # Update epsilon for exploration
        if i <= epsilon_decay_steps:
            current_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * i / epsilon_decay_steps
        else:
            current_epsilon = final_epsilon
        
        # Collect new experience periodically
        if i % experience_collection_interval == 0 and len(aug_replay_buffer) >= min_buffer_size:
            # Get current policy for experience collection
            dummy_batch = aug_replay_buffer.sample_with_goals(batch_size)
            if dummy_batch is not None:
                obs = jnp.arange(aug_env.nS)[:, None].repeat(aug_env.nA, axis=1).reshape(-1)
                actions = jnp.arange(aug_env.nA)[None, :].repeat(aug_env.nS, axis=0).reshape(-1)
                random_states = jnp.roll(dummy_batch['next_observations'], -1, axis=0)
                plus_logits = critic.apply(critic_params, obs, actions, 
                                           jnp.full_like(obs, (aug_env.nS - 2)),
                                           jnp.full_like(obs, (aug_env.nS - 2)))
                random_logits = critic.apply(critic_params, obs, actions, 
                                             jnp.full_like(obs, (aug_env.nS - 2)),
                                             random_states)
                plus_log_probs = jnp.diag(plus_logits) - jax.nn.logsumexp(random_logits, axis=-1)
                plus_log_probs = plus_log_probs.reshape([aug_env.nS, aug_env.nA])
                current_policy = get_epsilon_greedy_policy(np.array(plus_log_probs), current_epsilon)
                
                # Collect experience
                collect_augmented_experience(aug_env, current_policy, aug_replay_buffer, 
                                            experience_episodes_per_collection, 
                                            epsilon=current_epsilon)
        
        # Training step
        if len(aug_replay_buffer) >= min_buffer_size:
            batch = aug_replay_buffer.sample_with_goals(batch_size)
            if batch is not None:
                s_plus_marg = estimate_s_plus_marginal(aug_replay_buffer, aug_env.nS)
                critic_params, target_critic_params, opt_state, loss, info = update_fn(
                    critic_params, target_critic_params, opt_state, batch, s_plus_marg)

                for k, v in info.items():
                    metrics['train/' + k].append(np.array([i, v]))
                
                metrics['train/epsilon'].append(np.array([i, current_epsilon]))
                metrics['train/buffer_size'].append(np.array([i, len(aug_replay_buffer)]))

        if i == 1 or i % eval_interval == 0:
            if len(aug_replay_buffer) >= min_buffer_size:
                eval_batch = aug_replay_buffer.sample_with_goals(batch_size)
                if eval_batch is not None:
                    sr, aug_sr = compute_success_rate_infonce(critic_params, eval_batch)
                    metrics['eval/success_rate'].append(np.array([i, sr]))
                    metrics['eval/augmented_success_rate'].append(np.array([i, aug_sr]))

        if i == 1 or i % log_interval == 0:
            save_path = os.path.join(FIGURE_DIR, f"gc_td_infonce_metrics.png")
            plot_metrics(metrics, logyscale_stats=[], title="GC TD InfoNCE Training Progress", save_path=save_path)

    return metrics


def plot_comparison(q_learning_metrics, crl_metrics, td_infonce_metrics):
    """Plot comparison of all algorithms"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))

    metric_name = 'eval/success_rate'
    ax = axes[0]
    for algo, metrics in zip(['Online Q-learning', 'Online CRL + BNCE', 'Online TD InfoNCE'], 
                             [q_learning_metrics, crl_metrics, td_infonce_metrics]):
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            metrics_arr = np.asarray(metrics[metric_name])
            ax.plot(metrics_arr[:, 0], metrics_arr[:, 1], label=algo, linewidth=2)
    ax.set_ylabel('Success rate in\nthe original env', fontsize=14)
    ax.set_xlabel('Training iterations', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
     
    metric_name = 'eval/augmented_success_rate'
    ax = axes[1]
    for algo, metrics in zip(['Online Q-learning', 'Online CRL + BNCE', 'Online TD InfoNCE'], 
                             [q_learning_metrics, crl_metrics, td_infonce_metrics]):
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            metrics_arr = np.asarray(metrics[metric_name])
            ax.plot(metrics_arr[:, 0], metrics_arr[:, 1], label=algo, linewidth=2)
    ax.set_ylabel('Success rate in\nthe augmented env', fontsize=14)
    ax.set_xlabel('Training iterations', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    
    # Save comparison plot
    save_path = os.path.join(FIGURE_DIR, "algorithms_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure: {save_path}")
    plt.close(fig)


def main():
    """Main function to run all experiments"""
    print("Starting Online RL Experiments on CliffWalking Environment")
    print("=" * 60)
    
    # Global parameters
    global discount, max_episode_steps, env, aug_env, replay_buffer, aug_replay_buffer
    
    discount = 0.95
    max_episode_steps = 100
    
    # Initialize environments
    env = CliffWalkingEnv(random_init_state=True, max_episode_steps=max_episode_steps)
    aug_env = AugmentedCliffWalkingEnv(discount=discount, random_init_state=True, max_episode_steps=max_episode_steps)
    
    # Initialize replay buffers
    replay_buffer_capacity = 100000
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    aug_replay_buffer = AugmentedReplayBuffer(replay_buffer_capacity)
    
    print(f"Environment setup complete:")
    print(f"  Original env states: {env.nS}, actions: {env.nA}")
    print(f"  Augmented env states: {aug_env.nS}, actions: {aug_env.nA}")
    print(f"  Replay buffer capacity: {replay_buffer_capacity}")
    print(f"  Discount factor: {discount}")
    print()
    
    # Run all algorithms
    try:
        # Q-Learning
        q_learning_metrics = run_q_learning()
        print(f"Q-Learning final results:")
        if 'eval/success_rate' in q_learning_metrics and len(q_learning_metrics['eval/success_rate']) > 0:
            print(f"  Original env success rate: {q_learning_metrics['eval/success_rate'][-1][1]:.3f}")
        if 'eval/augmented_success_rate' in q_learning_metrics and len(q_learning_metrics['eval/augmented_success_rate']) > 0:
            print(f"  Augmented env success rate: {q_learning_metrics['eval/augmented_success_rate'][-1][1]:.3f}")
        print()
        
        # CRL + Binary NCE
        crl_metrics = run_crl_binary_nce()
        print(f"CRL + Binary NCE final results:")
        if 'eval/success_rate' in crl_metrics and len(crl_metrics['eval/success_rate']) > 0:
            print(f"  Original env success rate: {crl_metrics['eval/success_rate'][-1][1]:.3f}")
        if 'eval/augmented_success_rate' in crl_metrics and len(crl_metrics['eval/augmented_success_rate']) > 0:
            print(f"  Augmented env success rate: {crl_metrics['eval/augmented_success_rate'][-1][1]:.3f}")
        print()
        
        # GC TD InfoNCE
        td_infonce_metrics = run_gc_td_infonce()
        print(f"GC TD InfoNCE final results:")
        if 'eval/success_rate' in td_infonce_metrics and len(td_infonce_metrics['eval/success_rate']) > 0:
            print(f"  Original env success rate: {td_infonce_metrics['eval/success_rate'][-1][1]:.3f}")
        if 'eval/augmented_success_rate' in td_infonce_metrics and len(td_infonce_metrics['eval/augmented_success_rate']) > 0:
            print(f"  Augmented env success rate: {td_infonce_metrics['eval/augmented_success_rate'][-1][1]:.3f}")
        print()
        
        # Plot comparison
        plot_comparison(q_learning_metrics, crl_metrics, td_infonce_metrics)
        
        print("=" * 60)
        print("All experiments completed successfully!")
        print(f"Training figures saved to: {FIGURE_DIR}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()