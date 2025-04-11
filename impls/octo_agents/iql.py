import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.octo_model import OctoModel
# from octo.model.octo_module import OctoModule, OctoTransformer

from octo_utils.networks import (
    OctoTransformer,
    LowdimActionTokenizer
)

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, Value


# class ValueHead(nn.Module):
#     """Predicts value / critic using a MLP.
#
#     Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
#     actions are then predicted using a diffusion process conditioned on this embedding. The diffusion model
#     architecture is an MLP with residual connections (see `octo.model.components.diffusion`).
#
#     You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
#     attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
#     stream.
#     """
#
#     readout_key: str
#     use_map: bool = False
#     action_horizon: int = 1
#     action_dim: int = 7
#     max_action: float = 5.0
#     loss_type: str = "mse"
#
#     # diffusion-specific config with sane defaults
#     time_dim: int = 32
#     num_blocks: int = 3
#     dropout_rate: float = 0.0
#     hidden_dim: int = 256
#     use_layer_norm: bool = True
#     diffusion_steps: int = 20
#     n_diffusion_samples: int = 1
#
#     def setup(self):
#         if self.use_map:
#             self.map_head = MAPHead()
#
#         # create the diffusion model (score network)
#         self.diffusion_model = create_diffusion_model(
#             self.action_dim * self.action_horizon,
#             time_dim=self.time_dim,
#             num_blocks=self.num_blocks,
#             dropout_rate=self.dropout_rate,
#             hidden_dim=self.hidden_dim,
#             use_layer_norm=self.use_layer_norm,
#         )
#
#         # create beta schedule
#         self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
#         self.alphas = 1 - self.betas
#         self.alpha_hats = jnp.cumprod(self.alphas)
#
#     def __call__(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         time: Optional[ArrayLike] = None,
#         noisy_actions: Optional[ArrayLike] = None,
#         train: bool = True,
#     ) -> jax.Array:
#         """Performs a single forward pass through the diffusion model."""
#         token_group = transformer_outputs[self.readout_key]
#         assert token_group.tokens.ndim == 4, (
#             f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
#             f"but got shape {token_group.tokens.shape}"
#         )
#         if self.use_map:  # Multi-head attention pooling
#             embeddings = self.map_head(token_group, train=train)[:, :, 0]
#         else:  # mean pooling
#             embeddings = token_group.tokens.mean(axis=-2)
#         # Now, embeddings is (batch_size, window_size, embedding_size)
#
#         # time and noisy_actions are None during initialization, so we replace them with a dummy array
#         if (time is None or noisy_actions is None) and not self.is_initializing():
#             raise ValueError(
#                 "Must provide time and noisy_actions when calling diffusion action head"
#             )
#         elif self.is_initializing():
#             time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
#             noisy_actions = jnp.zeros(
#                 (*embeddings.shape[:2], self.action_dim * self.action_horizon),
#                 dtype=jnp.float32,
#             )
#         pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
#         return pred_eps
#
#     def loss(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         actions: ArrayLike,
#         timestep_pad_mask: ArrayLike,
#         action_pad_mask: ArrayLike,
#         train: bool = True,
#     ) -> Tuple[Array, Dict[str, Array]]:
#         """Computes the loss for the diffusion objective.
#
#         Args:
#             transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
#                 embedding_size)
#             actions: shape (batch_size, window_size, action_horizon, action_dim)
#             timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
#             action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
#
#         Returns:
#             loss: float
#             metrics: dict
#         """
#         batch_size, window_size = timestep_pad_mask.shape
#
#         # fold action_dim and action_horizon into one dimension
#         actions_flat = rearrange(actions, "b w h a -> b w (h a)")
#         actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)
#
#         # piggy-back on the dropout rng chain for diffusion rng
#         rng = self.make_rng("dropout")
#         time_key, noise_key = jax.random.split(rng)
#         time = jax.random.randint(
#             time_key,
#             (self.n_diffusion_samples, batch_size, window_size, 1),
#             0,
#             self.diffusion_steps,
#         )
#         noise = jax.random.normal(
#             noise_key, (self.n_diffusion_samples,) + actions_flat.shape
#         )
#
#         scale = jnp.sqrt(self.alpha_hats[time])
#         std = jnp.sqrt(1 - self.alpha_hats[time])
#         noisy_actions = scale * actions_flat[None] + std * noise
#
#         pred_eps = self(
#             transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
#         )
#
#         # combine the timestep pad mask with the action pad mask
#         mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
#         # flatten the mask to match the flat actions
#         mask = rearrange(mask, "b w h a -> b w (h a)")
#         # add a dimension to the mask for n_diffusion_samples
#         mask = mask[None]
#
#         loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
#         # Sum over action dimension instead of averaging
#         loss = loss * self.action_dim
#         metrics["loss"] = metrics["loss"] * self.action_dim
#         metrics["mse"] = metrics["mse"] * self.action_dim
#         return loss, metrics
#
#     def predict_action(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         rng: PRNGKey,
#         train: bool = True,
#         embodiment_action_dim: Optional[int] = None,
#         *args,
#         sample_shape: tuple = (),
#         **kwargs,
#     ) -> jax.Array:
#         """Convenience methods for predicting actions for the final timestep in the window."""
#         if embodiment_action_dim is None:
#             logging.warning(
#                 "embodiment_action_dim is highly recommended for diffusion action head"
#                 " if any action dimensions were masked during training"
#             )
#         batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
#         module, variables = self.unbind()
#
#         action_mask = jnp.ones(
#             (
#                 *sample_shape,
#                 batch_size,
#                 window_size,
#                 self.action_horizon,
#                 self.action_dim,
#             ),
#             dtype=bool,
#         )
#         if embodiment_action_dim is not None:
#             action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
#         flat_action_mask = rearrange(action_mask, "... p a -> ... (p a)")
#
#         def scan_fn(carry, time):
#             current_x, rng = carry
#             input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))
#
#             eps_pred = module.apply(
#                 variables, transformer_outputs, input_time, current_x, train=train
#             )
#
#             alpha_1 = 1 / jnp.sqrt(self.alphas[time])
#             alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
#             current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
#
#             rng, key = jax.random.split(rng)
#             z = jax.random.normal(key, shape=current_x.shape)
#             current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)
#
#             current_x = jnp.clip(current_x, -self.max_action, self.max_action)
#
#             # set non-eval actions to the noise that would have been seen during training
#             current_x = jnp.where(
#                 flat_action_mask, current_x, jnp.sqrt(1 - self.alpha_hats[time]) * z
#             )
#
#             return (current_x, rng), ()
#
#         rng, key = jax.random.split(rng)
#         noise = jax.random.normal(
#             key,
#             (
#                 *sample_shape,
#                 batch_size,
#                 window_size,
#                 self.action_horizon * self.action_dim,
#             ),
#         )
#
#         (actions_flat, _), () = jax.lax.scan(
#             scan_fn,
#             (noise, rng),
#             jnp.arange(self.diffusion_steps - 1, -1, -1),
#         )
#
#         actions = rearrange(
#             actions_flat,
#             "... (h a) -> ... h a",
#             h=self.action_horizon,
#             a=self.action_dim,
#         )
#         # only get the last timestep in the window
#         return actions[..., -1, :, :]


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

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
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
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'])
            q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
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
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], actions=q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
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
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}

        bc_loss, bc_info = self.behavioral_cloning_loss(batch, grad_params)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        loss = bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
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

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        text_processor,
        dataset_statistics,
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
        rng, init_rng, octo_init_rng = jax.random.split(rng, 3)

        octo_model_config = octo_config['model']

        octo_model_config['action_tokenizers'] = ModuleSpec.create(
            LowdimActionTokenizer,
            n_bins=256,
            bin_type="normal",
            low=-5.0,
            high=5.0,
        )
        octo_model_config['readouts'] = {'action': 1, 'v': 1, 'q': 1}
        octo_model_config['heads']['action'] = ModuleSpec.create(
            DiffusionActionHead,
            readout_key='readout_action',
            use_map=False,
            action_horizon=4,
            action_dim=example_batch['action'].shape[-1],
            n_diffusion_samples=1,
            dropout_rate=0.0,
        )
        octo_model_config['heads']['v'] = ModuleSpec.create(
            DiffusionActionHead,
            readout_key='readout_action',
            use_map=False,
            action_horizon=4,
            action_dim=example_batch['action'].shape[-1],
            n_diffusion_samples=1,
            dropout_rate=0.0,
        )
        octo_model_config['heads']['q'] = ModuleSpec.create(
            DiffusionActionHead,
            readout_key='readout_action',
            use_map=False,
            action_horizon=4,
            action_dim=example_batch['action'].shape[-1],
            n_diffusion_samples=1,
            dropout_rate=0.0,
        )

        # model = OctoModel.from_config(
        #     octo_config.to_dict(),
        #     example_batch,
        #     text_processor,
        #     verbose=True,
        #     rng=init_rng,
        #     dataset_statistics=dataset_statistics,
        # )

        # module = OctoModule.create(**config["model"])
        # @classmethod
        # def create(
        #     cls,
        #     observation_tokenizers: Dict[str, ModuleSpec],
        #     task_tokenizers: Dict[str, ModuleSpec],
        #     heads: Dict[str, ModuleSpec],
        #     readouts: Dict[str, int],
        #     transformer_kwargs: Dict,
        #     token_embedding_size: int,
        #     max_horizon: int,
        #     repeat_task_tokens: bool = False,
        #     use_correct_attention: bool = False,
        # ) -> "OctoModule":
        #     """
        #     Canonical way to create an OctoModule from configuration.
        #
        #     Args:
        #         observation_tokenizers: dict of {tokenizer_name: tokenizer_spec} (see tokenizers.py)
        #         task_tokenizers: dict of {tokenizer_name: tokenizer_spec} (see tokenizers.py)
        #         heads: dict of {head_name: head_spec} (see heads.py)
        #         readouts: dict of {readout_name (str): n_tokens_for_readout (int)}
        #         token_embedding_size (int): The latent dimension of the token embeddings
        #         max_horizon (int): Sets the size of positional embeddings, and provides an upper limit on the
        #             maximum horizon of the model
        #         repeat_task_tokens (bool): If true, repeats the task tokens at each observation timestep.
        #         transformer_kwargs: additional kwargs to forward to the transformer, which include:
        #             num_layers (int): number of layers
        #             mlp_dim (int): hidden dimension of the MLPs
        #             num_heads (int): Number of heads in nn.MultiHeadDotProductAttention
        #             dropout_rate (float): dropout rate.
        #             attention_dropout_rate (float): dropout rate in self attention.
        #     """
        #
        #     observation_tokenizer_defs = {
        #         k: ModuleSpec.instantiate(spec)()
        #         for k, spec in observation_tokenizers.items()
        #     }
        #     task_tokenizer_defs = {
        #         k: ModuleSpec.instantiate(spec)() for k, spec in task_tokenizers.items()
        #     }
        #
        #     head_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in heads.items()}
        #
        #     model_def = OctoTransformer(
        #         observation_tokenizers=observation_tokenizer_defs,
        #         task_tokenizers=task_tokenizer_defs,
        #         readouts=readouts,
        #         token_embedding_size=token_embedding_size,
        #         max_horizon=max_horizon,
        #         repeat_task_tokens=repeat_task_tokens,
        #         transformer_kwargs=transformer_kwargs,
        #         use_correct_attention=use_correct_attention,
        #     )
        #
        #     return cls(
        #         octo_transformer=model_def,
        #         heads=head_defs,
        #     )
        rng = rng if rng is not None else jax.random.PRNGKey(0)

        observation_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['observation_tokenizers'].items()
        }
        task_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['task_tokenizers'].items()
        }
        action_tokenizer_def = (
            ModuleSpec.instantiate(octo_model_config['action_tokenizers'])()
        )

        head_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in octo_model_config['heads'].items()
        }

        octo_transformer_def = OctoTransformer(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            action_tokenizers=action_tokenizer_def,
            readouts=octo_model_config['readouts'],
            token_embedding_size=octo_model_config['token_embedding_size'],
            max_horizon=octo_model_config['max_horizon'],
            repeat_task_tokens=octo_model_config['repeat_task_tokens'],
            transformer_kwargs=octo_model_config['transformer_kwargs'],
            use_correct_attention=octo_model_config['use_correct_attention'],
        )

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )

        if verbose:
            print(
                module.tabulate(rng, *init_args, train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details

        @jax.jit
        def _init(rng):
            return module.init(rng, *init_args, train=False)

        network_params = _init(rng)["params"]

        # return cls(
        #     module=module,
        #     params=params,
        #     text_processor=text_processor,
        #     example_batch=example_batch,
        #     config=config,
        #     dataset_statistics=dataset_statistics,
        # )

        # network = TrainState.create(network_def, network_params, tx=network_tx)

        # action_dim = ex_actions.shape[-1]
        #
        # # Define encoders.
        # encoders = dict()
        # if config['encoder'] is not None:
        #     encoder_module = encoder_modules[config['encoder']]
        #     encoders['value'] = encoder_module()
        #     encoders['critic'] = encoder_module()
        #     encoders['actor'] = encoder_module()
        #
        # # Define networks.
        # value_def = Value(
        #     hidden_dims=config['value_hidden_dims'],
        #     layer_norm=config['layer_norm'],
        #     num_ensembles=1,
        #     encoder=encoders.get('value'),
        # )
        # critic_def = Value(
        #     hidden_dims=config['value_hidden_dims'],
        #     layer_norm=config['layer_norm'],
        #     num_ensembles=2,
        #     encoder=encoders.get('critic'),
        # )
        # actor_def = GCActor(
        #     hidden_dims=config['actor_hidden_dims'],
        #     action_dim=action_dim,
        #     state_dependent_std=False,
        #     layer_norm=config['actor_layer_norm'],
        #     const_std=config['const_std'],
        #     gc_encoder=encoders.get('actor'),
        # )
        #
        # network_info = dict(
        #     value=(value_def, (ex_observations,)),
        #     critic=(critic_def, (ex_observations, ex_actions)),
        #     target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
        #     actor=(actor_def, (ex_observations,)),
        # )
        # networks = {k: v[0] for k, v in network_info.items()}
        # network_args = {k: v[1] for k, v in network_info.items()}
        #
        # network_def = ModuleDict(networks)
        # network_tx = optax.adam(learning_rate=config['lr'])
        # network_params = network_def.init(init_rng, **network_args)['params']
        # network = TrainState.create(network_def, network_params, tx=network_tx)
        #
        # params = network_params
        # params['modules_target_critic'] = params['modules_critic']

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
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
