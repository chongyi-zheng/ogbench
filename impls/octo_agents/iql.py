import copy
from functools import partial
from typing import Any

import flax
# import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
# import optax

from octo.utils.spec import ModuleSpec
# from octo.data.utils.data_utils import NormalizationType
# from octo.model.components.action_heads import DiffusionActionHead
# from octo.model.octo_model import OctoModel
# from octo.model.octo_module import OctoModule, OctoTransformer

from octo.utils.train_utils import (
    create_optimizer,
)
from octo_utils.networks import (
    LowdimActionTokenizer,
    OctoTransformer,
    ContinuousGaussianActorHead,
    ValueHead,
)

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class IQLAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, rng):
        """Compute the IQL value loss."""
        observations = batch['observation']
        tasks = batch['task']
        timestep_pad_masks = batch['observation']['timestep_pad_mask']
        actions = batch['action']
        action_pad_masks = batch['action_pad_mask']

        rng, transformer_dropout_rng, target_transformer_dropout_rng = jax.random.split(rng, 3)
        transformer_outputs = self.network.select('octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            train=True,
            params=grad_params,
            rngs={'dropout': transformer_dropout_rng},
        )
        target_transformer_outputs = self.network.select('target_octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            actions=actions, action_pad_mask=action_pad_masks,
            train=True,
            rngs={'dropout': target_transformer_dropout_rng},
        )

        rng, q_dropout_rng, v_dropout_rng = jax.random.split(rng, 3)
        q1, q2 = self.network.select('target_critic_head')(
            target_transformer_outputs,
            train=True,
            rngs={'dropout': q_dropout_rng},
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('value_head')(
            transformer_outputs,
            train=True,
            rngs={'dropout': v_dropout_rng},
            params=grad_params,
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the IQL critic loss."""
        observations = batch['observation']
        tasks = batch['task']
        timestep_pad_masks = batch['observation']['timestep_pad_mask']
        actions = batch['action']
        action_pad_masks = batch['action_pad_mask']
        next_observations = batch['next_observation']
        next_timestep_pad_masks = batch['next_observation']['timestep_pad_mask']
        rewards = batch['reward']
        masks = batch['mask']

        rng, transformer_dropout_rng, next_transformer_dropout_rng = jax.random.split(rng, 3)
        transformer_outputs = self.network.select('octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            actions=actions, action_pad_masks=action_pad_masks,
            train=True,
            params=grad_params,
            key={'dropout': transformer_dropout_rng},
        )
        next_transformer_outputs = self.network.select('octo_transformer')(
            observations=next_observations, tasks=tasks,
            timestep_pad_mask=next_timestep_pad_masks,
            train=True,
            key={'dropout': next_transformer_dropout_rng}
        )

        rng, next_v_dropout_rng, q_dropout_rng = jax.random.split(rng, 3)
        next_v = self.network.select('value_head')(
            next_transformer_outputs,
            train=True,
            rngs={'dropout': next_v_dropout_rng},
        )
        q = rewards + self.config['discount'] * masks * next_v

        q1, q2 = self.network.select('critic_head')(
            transformer_outputs,
            train=True,
            rngs={'dropout': q_dropout_rng},
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def behavioral_cloning_loss(self, batch, grad_params, rng):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observation']
        tasks = batch['task']
        timestep_pad_masks = batch['observation']['timestep_pad_mask']
        actions = batch['action']

        rng, transformer_dropout_rng, actor_dropout_rng = jax.random.split(rng, 3)
        transformer_outputs = self.network.select('octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            train=True,
            rngs={'dropout': transformer_dropout_rng},
            params=grad_params,
        )
        dist = self.network.select('actor_head')(
            transformer_outputs,
            train=True,
            rngs={'dropout': actor_dropout_rng},
            params=grad_params,
        )

        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        return bc_loss, {
            'bc_loss': bc_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - actions) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR)."""
        observations = batch['observation']
        tasks = batch['task']
        timestep_pad_masks = batch['observation']['timestep_pad_mask']
        actions = batch['action']
        action_pad_masks = batch['action_pad_mask']

        # AWR loss.
        rng, transformer_dropout_rng = jax.random.split(rng)
        transformer_outputs = self.network.select('octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            actions=actions, action_pad_mask=action_pad_masks,
            train=True,
            rngs={'dropout': transformer_dropout_rng},
            params=grad_params,
        )

        rng, v_dropout_rng, q_dropout_rng = jax.random.split(rng, 3)
        v = self.network.select('value_head')(
            transformer_outputs, 
            train=True,
            rngs={'dropout': v_dropout_rng},
        )
        q1, q2 = self.network.select('critic_head')(
            transformer_outputs,
            train=True,
            rngs={'dropout': q_dropout_rng},
        )
        q = jnp.minimum(q1, q2)
        adv = q - v
        adv = jax.lax.stop_gradient(adv)

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        rng, actor_dropout_rng = jax.random.split(rng)
        dist = self.network.select('actor_head')(
            transformer_outputs, 
            train=True,
            rngs={'dropout': actor_dropout_rng},
            params=grad_params,
        )
        log_prob = dist.log_prob(actions)

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

        bc_loss, bc_info = self.behavioral_cloning_loss(batch, grad_params, rng)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        loss = bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, value_rng, critic_rng = jax.random.split(rng, 3)
        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
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

        loss = value_loss + critic_loss + actor_loss
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

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'octo_transformer')
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update=True, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'octo_transformer')
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        tasks,
        timestep_pad_masks=None,
        seed=None,
        temperature=1.0,
        train=False,
    ):
        """Sample actions from the actor."""
        if timestep_pad_masks is None:
            timestep_pad_masks = observations["timestep_pad_mask"]
        if len(timestep_pad_masks.shape) == 1:
            observations = jax.tree_map(lambda x: x[None], observations)
            timestep_pad_masks = timestep_pad_masks[None]

        transformer_outputs = self.network.select('octo_transformer')(
            observations=observations, tasks=tasks,
            timestep_pad_mask=timestep_pad_masks,
            train=train,
        )
        dist = self.network.select('actor_head')(
            transformer_outputs,
            temperature=temperature,
            train=train,
        )

        # only get the last timestep in the window
        actions = dist.sample(seed=seed)[:, -1].squeeze()

        return actions, transformer_outputs, dist

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
        octo_config,
        verbose=True,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        octo_model_config = octo_config['model']
        octo_model_config['action_tokenizers'] = {
            'action': ModuleSpec.create(
                LowdimActionTokenizer,
                action_horizon=4,
                action_dim=example_batch['action'].shape[-1],
                n_bins=256,
                bin_type="normal",
                low=-5.0,
                high=5.0,
            )
        }

        octo_model_config['readouts'] = {'action': 1, 'value': 1, 'critic': 1}
        octo_model_config['heads']['action'] = ModuleSpec.create(
            ContinuousGaussianActorHead,
            readout_key='readout_action',
            use_map=True,
            action_horizon=4,
            action_dim=example_batch['action'].shape[-1],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        octo_model_config['heads']['value'] = ModuleSpec.create(
            ValueHead,
            readout_key='readout_value',
            use_map=True,
            num_ensembles=1,
        )
        octo_model_config['heads']['critic'] = ModuleSpec.create(
            ValueHead,
            readout_key='readout_critic',
            use_map=True,
            num_ensembles=2,
        )

        observation_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['observation_tokenizers'].items()
        }
        task_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['task_tokenizers'].items()
        }
        action_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['action_tokenizers'].items()
        }

        actor_head_def = ContinuousGaussianActorHead(
            readout_key='readout_action',
            use_map=True,
            action_horizon=4,
            action_dim=example_batch['action'].shape[-1],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        critic_head_def = ValueHead(
            readout_key='readout_value',
            use_map=True,
            num_ensembles=2,
        )
        value_head_def = ValueHead(
            readout_key='readout_critic',
            use_map=True,
            num_ensembles=1,
        )

        octo_transformer_def = OctoTransformer(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            action_tokenizers=action_tokenizer_defs,
            readouts=octo_model_config['readouts'],
            token_embedding_size=octo_model_config['token_embedding_size'],
            max_horizon=octo_model_config['max_horizon'],
            repeat_task_tokens=octo_model_config['repeat_task_tokens'],
            transformer_kwargs=octo_model_config['transformer_kwargs'],
            use_correct_attention=octo_model_config['use_correct_attention'],
        )

        if verbose:
            print(
                octo_transformer_def.tabulate(
                    rng,
                    observations=example_batch['observation'], tasks=example_batch['task'],
                    timestep_pad_mask=example_batch['observation']['timestep_pad_mask'],
                    actions=example_batch['action'], action_pad_mask=example_batch['action_pad_mask'],
                    train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details
        example_transformer_outputs, _ = octo_transformer_def.init_with_output(
            rng,
            observations=example_batch['observation'], tasks=example_batch['task'],
            timestep_pad_mask=example_batch['observation']['timestep_pad_mask'],
            actions=example_batch['action'], action_pad_mask=example_batch['action_pad_mask'],
            train=False
        )

        network_info = dict(
            octo_transformer=(octo_transformer_def, (
                example_batch['observation'], example_batch['task'],
                example_batch['observation']['timestep_pad_mask'],
                example_batch['action'], example_batch['action_pad_mask']
            )),
            target_octo_transformer=(copy.deepcopy(octo_transformer_def), (
                example_batch['observation'], example_batch['task'],
                example_batch['observation']['timestep_pad_mask'],
                example_batch['action'], example_batch['action_pad_mask']
            )),
            actor_head=(actor_head_def, (example_transformer_outputs, )),
            critic_head=(critic_head_def, (example_transformer_outputs, )),
            target_critic_head=(copy.deepcopy(critic_head_def), (example_transformer_outputs, )),
            value_head=(value_head_def, (example_transformer_outputs, )),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx, lr_callable, param_norm_callable = create_optimizer(
            network_params,
            **octo_config.optimizer.to_dict(),
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_octo_transformer'] = params['modules_octo_transformer']
        params['modules_target_critic_head'] = params['modules_critic_head']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            actor_freq=2,  # Actor update frequency.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
