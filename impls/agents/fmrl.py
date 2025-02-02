from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCFMBilinearVectorField, GCFMBilinearValue, GCValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class FMRLAgent(flax.struct.PyTreeNode):
    """Flow Matching RL (FMRL) agent.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        rewards = batch['rewards']

        reward_preds = self.network.select('reward')(
            observations,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the contrastive value loss for the Q or V function."""
        rng, time_rng, noise_rng = jax.random.split(rng, 3)

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']
        
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        noises = jax.random.normal(noise_rng, shape=goals.shape)
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select('critic_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        cfm_loss = jnp.square(vf_pred - path_sample.dx_t[None]).mean()

        # distillation loss
        if self.config['distill_type'] == 'fwd_int':
            rng, g_rng = jax.random.split(rng)
            shortcut_noises = jax.random.normal(g_rng, shape=goals.shape, dtype=goals.dtype)
            sampled_goals = self.compute_fwd_flow_samples(
                shortcut_noises, observations, actions=actions)

            shortcut_goal_preds = shortcut_noises[None, None, :] + self.network.select('critic')(
                shortcut_noises, observations, actions=actions, params=grad_params)
            distill_loss = jnp.square(shortcut_goal_preds - sampled_goals[None]).mean()
        else:
            distill_loss = 0.0
        critic_loss = cfm_loss + distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'cond_flow_matching_loss': cfm_loss,
            'distillation_loss': distill_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss."""

        # DDPG+BC loss.
        assert not self.config['discrete']

        observations = batch['observations']
        rewards = batch['rewards']

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            rng, q_action_rng = jax.random.split(rng)
            q_actions = jnp.clip(dist.sample(seed=q_action_rng), -1, 1)

        rng, g_rng = jax.random.split(rng)
        noises = jax.random.normal(g_rng, shape=observations.shape, dtype=observations.dtype)
        if self.config['distill_type'] == 'fwd_int':
            sampled_goals = noises[None, None, :] + self.network.select('critic')(
                noises, observations, actions=q_actions)
            if self.config['q_agg'] == 'min':
                sampled_goals = sampled_goals.min(axis=0)
            else:
                sampled_goals = sampled_goals.mean(axis=0)
        else:
            sampled_goals = self.compute_fwd_flow_samples(noises, observations, actions=q_actions)

        goal_rewards = self.network.select('reward')(sampled_goals)
        q = (1 - self.config['discount']) * rewards + self.config['discount'] * goal_rewards.mean(axis=-1)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()

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

    def compute_fwd_flow_samples(self, noises, observations, actions):
        noisy_goals = jnp.repeat(noises[None], noises.shape[0], axis=0)
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_goals, ) = carry

            noisy_goals = jax.vmap(jnp.diag, -1, -1)(noisy_goals)

            # Time for this iteration
            times = jnp.full(noisy_goals.shape[:-1], i * step_size)

            vf = self.network.select('critic_vf')(
                noisy_goals, times, observations, actions=actions)

            # Update goals and divergence integral. We need to consider Q ensemble here.
            # if self.config['q_agg'] == 'min':
            #     new_noisy_goals = jnp.min(noisy_goals[None] + vf * step_size, axis=0)
            # else:
            #     new_noisy_goals = jnp.mean(noisy_goals[None] + vf * step_size, axis=0)
            new_noisy_goals = noisy_goals[None] + vf * step_size

            # Return updated carry and scan output
            return (new_noisy_goals, ), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, ), _ = jax.lax.scan(
            body_fn, (noisy_goals, ), jnp.arange(num_flow_steps))

        return noisy_goals

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = reward_loss + critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        seed, actor_seed = jax.random.split(seed)
        actions = dist.sample(seed=actor_seed)
        if not self.config['discrete']:
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
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        rng, time_rng = jax.random.split(rng)
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0], ))

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['critic_vf'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['reward'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            # critic_def = GCDiscreteBilinearCritic(
            #     hidden_dims=config['value_hidden_dims'],
            #     latent_dim=config['latent_dim'],
            #     layer_norm=config['layer_norm'],
            #     ensemble=True,
            #     value_exp=True,
            #     state_encoder=encoders.get('critic_state'),
            #     action_dim=action_dim,
            # )
            raise NotImplementedError
        else:
            if config['distill_type'] == 'fwd_int':
                output_dim = ex_goals.shape[-1]
            else:
                output_dim = 1
            critic_def = GCFMBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                output_dim=output_dim,
                layer_norm=config['critic_layer_norm'],
                num_ensembles=2,
                state_encoder=encoders.get('critic'),
            )
            critic_vf_def = GCFMBilinearVectorField(
                vector_dim=ex_goals.shape[-1],
                latent_dim=config['latent_dim'],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['critic_layer_norm'],
                num_ensembles=1,
                state_encoder=encoders.get('critic_vf'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=not config['const_std'],
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        reward_def = GCValue(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('reward'),
        )

        network_info = dict(
            critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, )),
            reward=(reward_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='fmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            critic_layer_norm=False,  # Whether to use layer normalization for the critic.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            q_agg='mean',  # Aggregation method for target Q values.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            num_flow_steps=10,  # Number of steps for solving ODEs using the Euler method.
            distill_type='none',  # Distillation type ('none', 'log_prob', 'rev_int').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            relabel_reward=False,  # Whether to relabel the reward.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric samling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
        )
    )
    return config
