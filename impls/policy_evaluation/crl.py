from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic


class CRLEstimator(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) estimator."""

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
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1)
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + value_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the estimator and return a new estimator with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def evaluate_estimation(
            self,
            batch,
            seed=None,
    ):
        observations = batch['observations']
        actions = batch['actions']
        random_observations = jnp.roll(observations, 1, axis=0)
        random_actions = jnp.roll(actions, 1, axis=0)

        # we compute the accuracy of predicting goals from the same trajectory.
        assert (self.config['value_p_trajgoal'] == 1.0) or (self.config['actor_p_trajgoal'] == 1.0)
        if self.config['value_p_trajgoal'] == 1.0:
            goals = batch['value_goals']
        else:
            goals = batch['actor_goals']

        # seed, q_seed, q_random_seed = jax.random.split(seed, 3)
        # q = self.compute_log_likelihood(
        #     goals, observations,
        #     q_seed, actions=actions,
        # )
        # q_random = self.compute_log_likelihood(
        #     goals, random_observations,
        #     q_random_seed, actions=random_actions,
        # )
        _, q_phi, q_psi = self.network.select('critic')(
            observations, goals, actions=actions, info=True)
        if len(q_phi.shape) == 2:  # Non-ensemble.
            q_phi = q_phi[None, ...]
            q_psi = q_psi[None, ...]
        q = jnp.einsum('eik,ejk->ije', q_phi, q_psi) / jnp.sqrt(q_phi.shape[-1])
        q = jnp.diag(jnp.min(q, axis=-1))

        _, rand_q_phi, rand_q_psi = self.network.select('critic')(
            random_observations, goals, actions=random_actions, info=True)
        if len(rand_q_phi.shape) == 2:  # Non-ensemble.
            rand_q_phi = rand_q_phi[None, ...]
            rand_q_psi = rand_q_psi[None, ...]
        q_random = jnp.einsum('eik,ejk->ije', rand_q_phi, rand_q_psi) / jnp.sqrt(rand_q_phi.shape[-1])
        q_random = jnp.diag(jnp.min(q_random, axis=-1))

        # seed, v_seed, v_random_seed = jax.random.split(seed, 3)
        # v = self.compute_log_likelihood(
        #     goals, observations, v_seed)
        # v_random = self.compute_log_likelihood(
        #     goals, random_observations, v_random_seed)
        _, v_phi, v_psi = self.network.select('value')(
            observations, goals, info=True)
        if len(v_phi.shape) == 2:  # Non-ensemble.
            v_phi = v_phi[None, ...]
            v_psi = v_psi[None, ...]
        v = jnp.einsum('eik,ejk->ije', v_phi, v_psi) / jnp.sqrt(v_phi.shape[-1])
        v = jnp.diag(jnp.min(v, axis=-1))

        _, rand_v_phi, rand_v_psi = self.network.select('value')(
            random_observations, goals, info=True)
        if len(rand_v_phi.shape) == 2:  # Non-ensemble.
            rand_v_phi = rand_v_phi[None, ...]
            rand_v_psi = rand_v_psi[None, ...]
        v_random = jnp.einsum('eik,ejk->ije', rand_v_phi, rand_v_psi) / jnp.sqrt(rand_v_phi.shape[-1])
        v_random = jnp.diag(jnp.min(v_random, axis=-1))

        # log p(g | s, a) - log p(g) > log p(g | s_rand, a_rand) - log p(g)
        binary_q_acc = jnp.mean(q > q_random)
        # log p(g | s) - log p(g) > log p(g | s_rand) - log p (g)
        binary_v_acc = jnp.mean(v > v_random)

        return {
            'binary_q_acc': binary_q_acc,
            'binary_v_acc': binary_v_acc,
        }

    @jax.jit
    def compute_values(
        self,
        observations,
        goals=None,
        seed=None,
    ):
        _, v_phi, v_psi = self.network.select('value')(
            observations, goals, info=True)
        if len(v_phi.shape) == 2:  # Non-ensemble.
            v_phi = v_phi[None, ...]
            v_psi = v_psi[None, ...]
        v = jnp.einsum('eik,ejk->ije', v_phi, v_psi) / jnp.sqrt(v_phi.shape[-1])
        v = jnp.diag(jnp.min(v, axis=-1))

        return v

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new estimator.

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
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

        value_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            value_exp=True,
            state_encoder=encoders.get('value_state'),
            goal_encoder=encoders.get('value_goal'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
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
            # Estimator hyperparameters.
            estimator_name='crl',  # estimator name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
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