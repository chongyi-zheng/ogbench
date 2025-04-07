from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor, GCBilinearValue, GCValue,
    GCDiscreteActor, GCDiscreteBilinearCritic, GCDiscreteCritic
)


class GCCRLInfoNCEAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent with InfoNCE loss.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            actions = batch['actions']
        else:
            actions = None

        if self.config['critic_arch'] == 'bilinear':
            v, phi, psi = self.network.select(module_name)(
                batch['observations'],
                batch['value_goals'],
                actions=actions,
                info=True,
                params=grad_params,
            )
            if len(phi.shape) == 2:  # Non-ensemble.
                phi = phi[None, ...]
                psi = psi[None, ...]
            logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])

            # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
            I = jnp.eye(batch_size)

            if self.config['contrastive_loss'] == 'forward_infonce':
                def loss_fn(_logits):  # pylint: disable=invalid-name
                    return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                            + self.config['logsumexp_penalty_coeff'] * jax.nn.logsumexp(_logits, axis=1) ** 2)

                contrastive_loss = jax.vmap(loss_fn, in_axes=-1, out_axes=-1)(logits)
            elif self.config['contrastive_loss'] == 'symmetric_infonce':
                contrastive_loss = jax.vmap(
                    lambda _logits: optax.softmax_cross_entropy(logits=_logits, labels=I),
                    in_axes=-1,
                    out_axes=-1,
                )(logits) + jax.vmap(
                    lambda _logits: optax.softmax_cross_entropy(logits=_logits, labels=I),
                    in_axes=-1,
                    out_axes=-1,
                )(logits.transpose([1, 0, 2]))
            else:
                raise NotImplementedError
        elif self.config['critic_arch'] == 'mlp':
            # if module_name == 'critic':
            #     actions = actions[:, None].repeat(batch_size, axis=1)
            # v = self.network.select(module_name)(
            #     batch['observations'][:, None].repeat(batch_size, axis=1),
            #     batch['value_goals'][None, :].repeat(batch_size, axis=0),
            #     actions=actions,
            #     params=grad_params,
            # )
            # if len(v.shape) == 2:  # Non-ensemble.
            #     v = v[None, ...]
            # logits = v.transpose([1, 2, 0])
            pos_v = self.network.select(module_name)(
                batch['observations'],
                batch['value_goals'],
                actions=actions,
                params=grad_params,
            )
            neg_v = self.network.select(module_name)(
                batch['observations'],
                jnp.roll(batch['value_goals'], -1, axis=0),
                actions=actions,
                params=grad_params,
            )
            if len(pos_v.shape) == 1:  # Non-ensemble.
                pos_v = pos_v[None, ...]
                neg_v = neg_v[None, ...]
            pos_v = pos_v.transpose([1, 0])
            neg_v = neg_v.transpose([1, 0])
            v = pos_v
            logits = jnp.stack([pos_v, neg_v], axis=1)

            # logits.shape is (B, 2, e) with one term for positive pair and one terms for negative pairs in each row.
            I = jnp.stack([jnp.ones(batch_size), jnp.zeros(batch_size)], axis=1)

            contrastive_loss = jax.vmap(
                lambda _logits: optax.sigmoid_binary_cross_entropy(_logits, I),
                in_axes=-1, out_axes=-1
            )(logits)

        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1)
        binary_correct = (jnp.diag(logits) > jnp.diag(jnp.roll(logits, 1, axis=-1)))
        categorial_correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean(binary_correct),
            'categorical_accuracy': jnp.mean(categorial_correct),
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
            if self.config['critic_arch'] == 'bilinear':
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

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            )
            if self.config['critic_arch'] == 'bilinear':
                q1, q2 = jnp.diag(q1), jnp.diag(q2)
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
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Always train a separate V network
        value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
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

        # Critic architecture
        assert config['critic_arch'] in ['bilinear', 'mlp']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

            if config['critic_arch'] == 'bilinear':
                encoders['critic_state'] = encoder_module()
                encoders['critic_goal'] = encoder_module()
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()
            elif config['critic_arch'] == 'mlp':
                encoders['value'] = GCEncoder(concat_encoder=encoder_module())
                encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        if config['discrete']:
            if config['critic_arch'] == 'bilinear':
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
            elif config['critic_arch'] == 'mlp':
                critic_def = GCDiscreteCritic(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    gc_encoder=encoders.get('critic'),
                    action_dim=action_dim,
                )
        else:
            if config['critic_arch'] == 'bilinear':
                critic_def = GCBilinearValue(
                    hidden_dims=config['value_hidden_dims'],
                    latent_dim=config['latent_dim'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    value_exp=True,
                    state_encoder=encoders.get('critic_state'),
                    goal_encoder=encoders.get('critic_goal'),
                )
            elif config['critic_arch'] == 'mlp':
                critic_def = GCValue(
                    hidden_dims=config['value_hidden_dims'],
                    network_type=config['network_type'],
                    num_residual_blocks=config['value_num_residual_blocks'],
                    layer_norm=config['layer_norm'],
                    ensemble=True,
                    gc_encoder=encoders.get('critic'),
                )

        # Always train a separte V network.
        # AWR requires a separate V network to compute advantages (Q - V).
        if config['critic_arch'] == 'bilinear':
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=True,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
            )
        elif config['critic_arch'] == 'mlp':
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                network_type=config['network_type'],
                num_residual_blocks=config['value_num_residual_blocks'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
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
            actor=(actor_def, (ex_observations, ex_goals)),
            value=(value_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gccrl_infonce',  # Agent name.
            normalize_observation=False,  # Whether to normalize observation s.t. each coordinate is centered with unit variance.
            network_type='mlp',  # Network type of the actor and critic ('mlp' or 'simba')
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            actor_num_residual_blocks=1,  # Actor network number of residual blocks when using SimBa architecture.
            value_num_residual_blocks=2,  # Critic network number of residual blocks when using SimBa architecture.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            critic_arch="bilinear",  # Contrastive critic architecture ('bilinear' or 'mlp')
            contrastive_loss='symmetric_infonce',  # Contrastive loss type ('forward_infonce', 'symmetric_infonce').
            logsumexp_penalty_coeff=0.01,  # Coefficient for the logsumexp regularization in forward InfoNCE loss.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
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
