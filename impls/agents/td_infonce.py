from functools import partial
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor, GCBilinearValue, Value
)

class TDInfoNCEAgent(flax.struct.PyTreeNode):
    """Temporal Difference InfoNCE (TD InfoNCE) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    TD InfoNCE with DDPG+BC only fits a Q function, while TD InfoNCE with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        if self.config['reward_type'] == 'state_action':
            actions = batch['actions']
        else:
            actions = None
        rewards = batch['rewards']

        reward_preds = self.network.select('reward')(
            observations, actions=actions,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def contrastive_loss(self, batch, grad_params, td_loss_type='sarsa', rng=None):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]
        actions = batch['actions']

        # next observation logits
        next_v, next_phi, next_psi = self.network.select('critic')(
            batch['observations'],
            batch['next_observations'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(next_phi.shape) == 2:  # Non-ensemble.
            next_phi = next_phi[None, ...]
            next_psi = next_psi[None, ...]
        next_logits = jnp.einsum('eik,ejk->ije', next_phi, next_psi) / jnp.sqrt(next_phi.shape[-1])

        random_v, random_phi, random_psi = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(random_phi.shape) == 2:  # Non-ensemble.
            random_phi = random_phi[None, ...]
            random_psi = random_psi[None, ...]
        random_logits = jnp.einsum('eik,ejk->ije', random_phi, random_psi) / jnp.sqrt(random_phi.shape[-1])

        I = jnp.eye(batch_size)
        next_logits = jax.vmap(
            lambda logits1, logits2: I * logits1 + (1 - I) * logits2,
            in_axes=-1,
            out_axes=-1,
        )(next_logits, random_logits)
        v = jax.vmap(
            lambda v1, v2: I * v1 + (1 - I) * v2,
            in_axes=0,
            out_axes=0,
        )(next_v, random_v)

        loss1 = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(next_logits)

        # random goal logits
        # _, random_phi, random_psi = self.network.select(module_name)(
        #     batch['observations'],
        #     batch['value_goals'],
        #     actions=actions,
        #     info=True,
        #     params=grad_params,
        # )
        # if len(random_phi.shape) == 2:  # Non-ensemble.
        #     random_phi = random_phi[None, ...]
        #     random_psi = random_psi[None, ...]
        # random_logits = jnp.einsum('eik,ejk->ije', random_phi, random_psi) / jnp.sqrt(random_phi.shape[-1])

        if td_loss_type == 'q_learning':
            next_dist = self.network.select('actor')(batch['next_observations'], params=grad_params)
            if self.config['const_std']:
                next_actions = jnp.clip(next_dist.mode(), -1, 1)
            else:
                next_actions = jnp.clip(next_dist.sample(seed=rng), -1, 1)
        elif td_loss_type == 'sarsa':
            next_actions = batch['next_actions']
        else:
            raise NotImplementedError

        # importance sampling logits
        _, w_phi, w_psi = self.network.select('target_critic')(
            batch['next_observations'],
            batch['value_goals'],
            actions=next_actions,
            info=True,
        )
        if len(w_phi.shape) == 2:  # Non-ensemble.
            w_phi = w_phi[None, ...]
            w_psi = w_psi[None, ...]
        w_logits = jnp.einsum('eik,ejk->ije', w_phi, w_psi) / jnp.sqrt(w_phi.shape[-1])
        w_logits = jnp.min(w_logits, axis=-1)
        w = jax.nn.softmax(w_logits, axis=-1)
        w = jax.lax.stop_gradient(w)

        # Note that we remove the multiplier N for w to balance
        # one term of loss1 with N terms of loss2 in each row.
        loss2 = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits, labels=w),
            in_axes=-1,
            out_axes=-1,
        )(random_logits)

        contrastive_loss = (1 - self.config['discount']) * loss1 + self.config['discount'] * loss2
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(next_logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def behavioral_cloning_loss(self, batch, grad_params):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observations']
        actions = batch['actions']

        dist = self.network.select('actor')(observations, params=grad_params)
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        return bc_loss, {
            'bc_loss': bc_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Maximize log Q if actor_log_q.
        # def value_transform(x):
        #     return jnp.log(jnp.maximum(x, 1e-6))
        def value_transform(x):
            return x

        # DDPG+BC loss.
        # batch_size = batch['observations'].shape[0]

        dist = self.network.select('actor')(
            batch['observations'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        logits1, logits2 = value_transform(
            self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
        )
        logits = jnp.minimum(logits1, logits2)
        log_ratios = jnp.diag(logits) - jax.nn.logsumexp(logits, axis=-1) + jnp.log(logits.shape[-1])

        if self.config['reward_type'] == 'state_action':
            future_dist = self.network.select('actor')(
                batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                future_actions = jnp.clip(future_dist.mode(), -1, 1)
            else:
                future_actions = jnp.clip(future_dist.sample(seed=rng), -1, 1)
        else:
            future_actions = None
        reward_preds = self.network.select('reward')(batch['actor_goals'], future_actions)
        q = jnp.exp(log_ratios) * reward_preds

        # _, phi, psi = self.network.select('critic')(
        #     batch['observations'], batch['actor_goals'], q_actions, info=True)
        # if len(phi.shape) == 2:  # Non-ensemble.
        #     phi = phi[None, ...]
        #     psi = psi[None, ...]
        # q = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # q = jnp.min(q, axis=-1)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        # q = q / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        # I = jnp.eye(batch_size)
        # q_loss = optax.softmax_cross_entropy(logits=logits, labels=I)
        # q_loss = q_loss.mean()
        # log_prob = dist.log_prob(batch['actions'])

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])

        # MLE BC loss
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        # MSE BC loss
        # bc_loss = self.config['alpha'] * ((batch['actions'] - q_actions) ** 2).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}

        bc_loss, bc_info = self.behavioral_cloning_loss(batch, grad_params)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        critic_loss, critic_info = self.contrastive_loss(
            batch, grad_params, td_loss_type='sarsa')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss + bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.contrastive_loss(
            batch, grad_params, td_loss_type='q_learning', rng=critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            rng, actor_rng = jax.random.split(rng)
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = reward_loss + critic_loss + actor_loss
        return loss, info

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update=True, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

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
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['reward'] = encoder_module()
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define the reward, the value, and the actor networks.
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            encoder=encoders.get('reward'),
        )

        critic_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            reward=(reward_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, )),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='td_infonce',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            actor_freq=2,  # Actor update frequency.
            tau=0.005,  # Target network update rate.
            reward_type='state',  # Reward type. ('state', 'state_action')
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            relabel_reward=False,  # Whether to relabel the reward.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            value_geom_start=0,  # Whether the support of the geometric sampling is [0, inf) or [1, inf)
            num_value_goals=1,  # Number of value goals to sample
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            actor_geom_start=1,  # Whether the support the geometric sampling is [0, inf) or [1, inf)
            num_actor_goals=1,  # Number of actor goals to sample
        )
    )
    return config
