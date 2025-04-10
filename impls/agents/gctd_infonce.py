import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic


class GCTDInfoNCEAgent(flax.struct.PyTreeNode):
    """SARSA Temporal Difference InfoNCE (TD InfoNCE) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    TD InfoNCE with DDPG+BC only fits a Q function, while TD InfoNCE with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, module_name='critic', rng=None):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            actions = batch['actions']
            next_actions = batch['next_actions']
        else:
            actions = None
            next_actions = None

        # next observation logits
        next_v, next_phi, next_psi = self.network.select(module_name)(
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

        random_v, random_phi, random_psi = self.network.select(module_name)(
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

        # importance sampling logits
        _, w_phi, w_psi = self.network.select('target_' + module_name)(
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
            v, q1, q2 = jnp.diag(v), jnp.diag(q1), jnp.diag(q2)
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
            batch_size = batch['observations'].shape[0]

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            # q1, q2 = value_transform(
            #     self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            # )
            # q = jnp.minimum(q1, q2)

            _, phi, psi = self.network.select('critic')(
                batch['observations'], batch['actor_goals'], q_actions, info=True)
            if len(phi.shape) == 2:  # Non-ensemble.
                phi = phi[None, ...]
                psi = psi[None, ...]
            q = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
            q = jnp.min(q, axis=-1)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            # q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            # q = q / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            I = jnp.eye(batch_size)
            q_loss = optax.softmax_cross_entropy(logits=q, labels=I)
            q_loss = q_loss.mean()
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
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, 'critic', critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, value_rng = jax.random.split(rng)
        value_loss, value_info = self.critic_loss(batch, grad_params, 'value', value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
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
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'value')

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
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                network_type=config['network_type'],
                num_residual_blocks=config['value_num_residual_blocks'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                network_type=config['network_type'],
                num_residual_blocks=config['value_num_residual_blocks'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=True,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
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
                network_type=config['network_type'],
                num_residual_blocks=config['actor_num_residual_blocks'],
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        network_info.update(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config['use_lr_scheduler']:
            schedule = optax.cosine_decay_schedule(config['lr'], config['lr_decay_steps'], alpha=config['lr_alpha'])
            network_tx = optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
        else:
            network_tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='td_infonce',  # Agent name.
            normalize_observation=False,  # Whether to normalize observation s.t. each coordinate is centered with unit variance.
            network_type='mlp',  # Network type of the actor and critic ('mlp' or 'simba')
            use_lr_scheduler=False,  # Whether to use learning rate scheduler.
            lr=3e-4,  # Learning rate.
            lr_decay_steps=1_000_000,  # The number of steps to decay the learning rate.
            lr_alpha=0.01,  # Minimum learning rate multiplier.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            actor_num_residual_blocks=1,  # Actor network number of residual blocks when using SimBa architecture.
            value_num_residual_blocks=2,  # Critic network number of residual blocks when using SimBa architecture.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=1.0,  # Probability of using a random state as the value goal.
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
