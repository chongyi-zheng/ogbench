import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.env_utils import compute_reward
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCActor, Value, GCMetricValue, GCValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class HILPAgent(flax.struct.PyTreeNode):
    """HILP agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        if self.config['reward_type'] == 'state_action':
            actions = batch['actions']
        else:
            actions = None
        rewards = batch['rewards']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)
        reward_preds = self.network.select('reward')(
            observations, actions=actions,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def skill_value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_skill_critic')(batch['observations'], batch['skills'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('skill_value')(batch['observations'], batch['skills'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def skill_critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('skill_value')(batch['next_observations'], batch['skills'])

        q1, q2 = self.network.select('skill_critic')(
            batch['observations'], batch['skills'], batch['actions'], params=grad_params
        )
        q = batch['skill_rewards'] + self.config['discount'] * batch['masks'] * next_v
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def skill_actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        v = self.network.select('skill_value')(batch['observations'], batch['skills'])
        q1, q2 = self.network.select('skill_critic')(batch['observations'], batch['skills'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('skill_actor')(batch['observations'], batch['skills'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info


    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, skill_rng = jax.random.split(rng, 2)

        # Train HILP representation.
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        # Compute skill information
        batch_size = batch['observations'].shape[0]
        batch['phis'] = self.get_phis(batch['observations'])
        batch['next_phis'] = self.get_phis(batch['next_observations'])
        random_skills = np.random.randn(batch_size, self.config['latent_dim'])
        batch['skills'] = random_skills / jnp.linalg.norm(random_skills, axis=1, keepdims=True) * jnp.sqrt(config['latent_dim'])
        rewards = ((batch['next_phis'] - batch['phis']) * batch['skills']).sum(axis=1)
        batch['skill_rewards'] = rewards

        # Train skill policies.
        skill_value_loss, skill_value_info = self.skill_value_loss(batch, grad_params)
        for k, v in skill_value_info.items():
            info[f'skill_value/{k}'] = v

        skill_critic_loss, skill_critic_info = self.skill_critic_loss(batch, grad_params)
        for k, v in skill_critic_info.items():
            info[f'skill_critic/{k}'] = v

        skill_actor_loss, skill_actor_info = self.skill_actor_loss(batch, grad_params)
        for k, v in skill_actor_info.items():
            info[f'skill_actor/{k}'] = v

        loss = value_loss + skill_value_loss + skill_critic_loss + skill_actor_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, flow_matching_rng, actor_rng = jax.random.split(rng, 4)

        # Reward prediction model is currently unused.
        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        batch['skills'] = batch['chosen_skills']
        batch['skill_rewards'] = batch['rewards']

        # Continue training skill policies.
        skill_value_loss, skill_value_info = self.skill_value_loss(batch, grad_params)
        for k, v in skill_value_info.items():
            info[f'skill_value/{k}'] = v

        skill_critic_loss, skill_critic_info = self.skill_critic_loss(batch, grad_params)
        for k, v in skill_critic_info.items():
            info[f'skill_critic/{k}'] = v

        skill_actor_loss, skill_actor_info = self.skill_actor_loss(batch, grad_params)
        for k, v in skill_actor_info.items():
            info[f'skill_actor/{k}'] = v

        loss = reward_loss + skill_value_loss + skill_critic_loss + skill_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'skill_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.finetune_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'skill_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        skills,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('skill_actor')(observations, skills, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)

        return actions

    @jax.jit
    def get_phis(
            self,
            observations,
    ):
        """Return phi(s)."""
        _, phis, _ = self.network.select('value')(observations, observations, info=True)
        return phis[0]

    @classmethod
    def create(
            cls,
            seed,
            ex_observations,
            ex_actions,
            config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        # obs_dim = ex_observations.shape[-1]
        # action_dim = ex_actions.shape[-1]
        # ex_orig_observations = ex_observations
        # ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],), dtype=ex_actions.dtype)
        if config['use_reward_func']:
            assert config['reward_env_info'] is not None
        ex_orig_observations = ex_observations
        ex_skills = np.zeros((1, config['latent_dim']), dtype=ex_observations.dtype)

        ex_times = ex_actions[..., 0]
        obs_dims = ex_observations.shape[1:]
        obs_dim = obs_dims[-1]
        action_dim = ex_actions.shape[-1]
        action_dtype = ex_actions.dtype

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            raise NotImplementedError
            encoder_module = encoder_modules[config['encoder']]
            if 'mlp_hidden_dims' in encoder_module.keywords:
                obs_dim = encoder_module.keywords['mlp_hidden_dims'][-1]
            else:
                obs_dim = encoder_modules['impala'].mlp_hidden_dims[-1]
            rng, obs_rng = jax.random.split(rng, 2)
            ex_observations = jax.random.normal(
                obs_rng, shape=(ex_observations.shape[0], obs_dim), dtype=action_dtype)

            encoders['critic'] = encoder_module()
            encoders['critic_vf'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define value and actor networks.
        value_def = GCMetricValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            latent_dim=config['latent_dim'],
            num_ensembles=2,
            encoder=encoders.get('value'),
        )

        skill_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            gc_encoder=encoders.get('skill_value'),
        )
        skill_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('skill_value'),
        )
        skill_actor_def = GCActor(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('skill_actor'),
        )

        reward_def = Value(
            network_type=config['network_type'],
            num_residual_blocks=config['num_residual_blocks'],
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_observations)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
            skill_critic=(skill_critic_def, (ex_observations, ex_skills, ex_actions)),
            target_skill_critic=(copy.deepcopy(skill_critic_def), (ex_observations, ex_skills, ex_actions)),
            skill_value=(skill_value_def, (ex_observations, ex_skills)),
            skill_actor=(skill_actor_def, (ex_observations, ex_skills)),
        )
        if config['reward_type'] == 'state':
            network_info.update(
                reward=(reward_def, (ex_observations, )),
                target_reward=(copy.deepcopy(reward_def), (ex_observations, )),
            )
        else:
            network_info.update(
                reward=(reward_def, (ex_observations, ex_actions)),
                target_reward=(copy.deepcopy(reward_def), (ex_observations, ex_actions)),
            )
        if config['encoder'] is not None:
            network_info['critic_vf_encoder'] = (encoders.get('critic_vf'), (ex_orig_observations,))
            network_info['target_critic_vf_encoder'] = (copy.deepcopy(encoders.get('critic_vf')), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        if config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_value'] = params['modules_value']
        params['modules_target_skill_critic'] = params['modules_skill_critic']

        config['obs_dims'] = obs_dims
        config['action_dim'] = action_dim
        config['action_dtype'] = action_dtype

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hilp',  # Agent name.
            obs_dims=ml_collections.config_dict.placeholder(tuple),
            # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            action_dtype=ml_collections.config_dict.placeholder(np.dtype),
            # Action data type (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            network_type='mlp',  # Type of the network
            num_residual_blocks=1,  # Number of residual blocks for simba network.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            latent_dim=512,  # HILP latent dimension.
            reward_layer_norm=True,  # Whether to use layer normalization for the reward.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            q_agg='mean',  # Aggregation method for target Q values.
            critic_noise_type='normal',  # Critic noise type. ('marginal_state', 'normal').
            critic_fm_loss_type='sarsa_squared', # Type of critic flow matching loss. ('naive_sarsa', 'coupled_sarsa', 'sarsa_squared')
            num_flow_goals=1,  # Number of future flow goals for the compute target q.
            clip_flow_goals=False,  # Whether to clip the flow goals.
            use_terminal_masks=False,  # Whether to use the terminal masks.
            ode_solver_type='euler',  # Type of ODE solver ('euler', 'dopri5').
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            use_mixup=False,  # Whether to use mixup for the behavioral cloning loss. (prevent overfitting).
            mixup_alpha=2.0,  # mixup beta distribution parameter.
            mixup_bandwidth=2.75,  # mixup distance bandwith.
            actor_freq=2,  # Actor update frequency.
            alpha=10.0,  # AWR coefficient (need to be tuned for each environment).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            use_reward_func=False,  # Whether to use the ground truth reward function.
            use_target_reward=False,  # Whether to use the target reward network.
            reward_type='state',  # Reward type. ('state', 'state_action')
            encoder_actor_loss_grad=False,  # Whether to backpropagate gradients from the actor loss into the encoder.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            reward_env_info=ml_collections.config_dict.placeholder(dict),  # Environment information for computing the ground truth reward.
            # Dataset hyperparameters.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.625,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.375,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            num_value_goals=1,  # ?
            num_actor_goals=1,  # ?
            value_geom_start=1,  # ?
            actor_geom_start=1,  # ?
            relabel_reward=True,  # ?
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
