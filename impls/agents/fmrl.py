from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCFMVelocityField
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class FMRLAgent(flax.struct.PyTreeNode):
    """Flow Matching RL (FMRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    FMRL with DDPG+BC only fits a Q function, while FMRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    def flow_matching_loss(self, batch, grad_params, rng, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        rng, guidance_rng, time_rng, noise_rng = jax.random.split(rng, 4)
        use_guidance = (jax.random.uniform(guidance_rng) >= self.config['uncond_prob'])
        
        observations = jax.lax.select(
            use_guidance,
            batch['observations'], 
            jnp.full_like(batch['observations'], jnp.nan)
        )
        if module_name == 'critic':
            actions = jax.lax.select(
                use_guidance, 
                batch['actions'], 
                jnp.full_like(batch['actions'], jnp.nan)
            )
        else:
            actions = None
        goals = batch['value_goals']
        
        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        vf_pred = self.network.select(module_name)(
            goals,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        
        gaussian_noise = jax.random.normal(noise_rng, shape=goals.shape)
        path_sample = self.cond_prob_path(x_0=gaussian_noise, x_1=goals, t=times)
        cfm_loss = jnp.mean((vf_pred - path_sample.dx_t) ** 2)

        return cfm_loss, {
            'cond_flow_matching_loss': cfm_loss,
            # 'v_mean': v.mean(),
            # 'v_max': v.max(),
            # 'v_min': v.min(),
            # 'binary_accuracy': jnp.mean((logits > 0) == I),
            # 'categorical_accuracy': jnp.mean(correct),
            # 'logits_pos': logits_pos,
            # 'logits_neg': logits_neg,
            # 'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Maximize log Q if actor_log_q is True (which is default).
        if self.config['actor_log_q']:

            def value_transform(x):
                return jnp.log(jnp.maximum(x, 1e-6))
        else:

            def value_transform(x):
                return x

        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = value_transform(self.network.select('value')(batch['observations'], batch['actor_goals']))
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            )
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            )
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
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
        elif self.config['actor_loss'] == 'sfbc':
            # BC loss.
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -log_prob.mean()

            actor_loss = bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.flow_matching_loss(batch, grad_params, critic_rng, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, value_rng = jax.random.split(rng)
        value_loss, value_info = self.flow_matching_loss(batch, grad_params, value_rng, 'value')
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
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
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
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
        
        rng, time_rng = jax.random.split(rng, 2)
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0], ))

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['value_state'] = encoder_module()
            encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            # critic_def = GCDiscreteBilinearCritic(
            #     hidden_dims=config['value_hidden_dims'],
            #     latent_dim=config['latent_dim'],
            #     layer_norm=config['layer_norm'],
            #     ensemble=True,
            #     value_exp=True,
            #     state_encoder=encoders.get('critic_state'),
            #     goal_encoder=encoders.get('critic_goal'),
            #     action_dim=action_dim,
            # )
            raise NotImplementedError
        else:
            critic_def = GCFMVelocityField(
                output_dim=ex_goals.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                # latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

            value_def = GCFMVelocityField(
                output_dim=ex_goals.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                # latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
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
                state_dependent_std=config['state_dependent_std'],
                const_std=False,
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            critic=(critic_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            value=(value_def, (ex_goals, ex_times, ex_observations)),
            actor=(actor_def, (ex_observations, ex_goals)),
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
        ode_solver = ODESolver(cond_prob_path)

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='fmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            # latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CosineScheduler',  # Scheduler class name.
            uncond_prob=0.2,  # Probability of training the marginal velocity field vs the guided velocity field.
            actor_loss='ddpgbc',  # Actor loss type ('ddpgbc' or 'awr' or 'sfbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            state_dependent_std=True,  # Whether to use state-dependent standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
