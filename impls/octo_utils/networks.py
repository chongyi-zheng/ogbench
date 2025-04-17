import logging
from typing import Dict, Optional

import flax.linen as nn
import distrax
import jax
import jax.numpy as jnp
from einops import rearrange

from octo.model.components.base import TokenGroup
from octo.model.components.transformer import MAPHead
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from octo.utils.typing import Data, Sequence

from utils.networks import (
    default_init,
    ensemblize,
    TransformedWithMode
)


class LowdimActionTokenizer(BinTokenizer):
    """
    Tokenizer for actions. Optionally discretizes into bins per dimension (see BinTokenizer).

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """

    # obs_keys: Sequence[str] = tuple()
    discretize: bool = False
    action_horizon: int = 1
    action_dim: int = 7

    def __call__(self, actions, *unused_args, **unused_kwargs):
        tokenizer_inputs = rearrange(
            actions, "b w h a -> b w (h a)", h=self.action_horizon, a=self.action_dim
        )
        if self.discretize:
            tokenized_inputs = super().__call__(tokenizer_inputs)
            tokens = jax.nn.one_hot(tokenized_inputs, self.n_bins)
        else:
            tokens = tokenizer_inputs[..., None]
        mask = jnp.ones(tokens.shape[:-1], dtype=jnp.bool)
        return TokenGroup(tokens, mask)


class OctoTransformer(nn.Module):
    """
    The following transformer code is adapted from:
        https://github.com/octo-models/octo/blob/main/octo/model/octo_module.py

    This module forms the base of the Octo architecture.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    The task is tokenized using a set of *task tokenizers* (for example, a tokenizer that processes the
    language instruction into tokens, or one that processes the goal images into tokens).

    The observation at each timestep is tokenized using a set of *observation tokenizers*
    (for example, a tokenizer that processes the primary image into tokens, or one that processes
    the wrist image into tokens).

    We introduce additional tokens ("readouts") that "read out" the information in the transformer for
    downstream action or value prediction. For example, we may have an "action" readout that provides
    embeddings that are useful for predicting actions, and a "value" readout with embeddings that are useful
    for predicting values.

    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or
    previous timesteps.  The easiest way to understand how the model works is to run:

    ```
        >>> model(observations, tasks, timestep_pad_mask, verbose=True)
    ```

    Generally, the model runs the transformer on something like the following sequence:

    [
        <task language tokens>,
        <t=0 "image_primary" tokens>, <t=0 "image_wrist" tokens>, <t=0 readout_action tokens>, ...
        <t=1 "image_primary" tokens>, <t=1 "image_wrist" tokens>, <t=1 readout_action tokens>, ...
        <t=2 "image_primary" tokens>, <t=2 "image_wrist" tokens>, <t=2 readout_action tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous
    timesteps. So, "image_wrist" can attend to "image_primary" and vice versa.

    Readouts provide a mechanism for "reading out" the information in the transformer. They are designed to
    only *read* from the sequence before it, without the ability to influence (i.e. write) the computation for
    any of the non-readout tokens. By design, different readouts (e.g. "action" vs "value") are completely
    independent of each other, meaning they can be run separately without affecting each other.

    Args:
        observations_tokenizers (Dict[str, nn.Module]): Dictionary of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Dict[str, nn.Module]): Dictionary of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        readouts (Dict[str, int]): Dictionary of {readout_name: n_tokens_for_readout}.
        transformer_kwargs (Dict): Dictionary of kwargs to forward to the Transformer.
        token_embedding_size (int): Dimension of the token embeddings
        max_horizon (int): The maximum number of timesteps that the transformer can be run with. Note that while the
            transformer can be run with any horizon <= max_horizon, the model will only generate sane outputs for
            horizon lengths smaller or equal to the pre-training horizon.
        repeat_task_tokens: If true, repeats the task tokens at each observation timesetep.

    """

    observation_tokenizers: Dict[str, nn.Module]
    task_tokenizers: Dict[str, nn.Module]
    action_tokenizers: Dict[str, nn.Module]
    readouts: Dict[str, int]
    transformer_kwargs: Dict
    token_embedding_size: int
    max_horizon: int
    repeat_task_tokens: bool
    use_correct_attention: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Data,
        tasks: Data,
        timestep_pad_mask: jax.Array,
        actions: Data = None,
        action_pad_mask: Data = None,
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
    ) -> Dict[str, TokenGroup]:
        """
        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            timestep_pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            readouts: A list of readouts to compute. If None, defaults to all readouts. Must be a subset of the readouts specified in the model config.
            train: Whether model is being trained.
            verbose: If True, prints out the transformer structure.

        Returns:
            transformer_outputs: A dictionary {token_group_name: token_group},
                which contain the transformer embeddings for all observation tokens, task tokens, and readout tokens.
                The special keys "task" and "obs" contain the concatenated embeddings for all task tokens and observation tokens, respectively.

        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        #
        # Check that all inputs are valid
        #

        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readouts must be specified in the model config"

        batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, observations)
        ), "observations must have the same horizon"

        #
        # Attention rules for the transformer
        #

        # Tasks attend to all other tasks, but not to observations or readouts
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}

        # Observations attend to all tasks and all other observations tokens causally,
        # e.g. at same timestep or before, but do not attend to readouts

        observation_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }
        
        # Actions attend to all tasks and all other actions tokens causally,
        # e.g. at same timestep or before, but do not attend to readouts
        action_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "action_*": AttentionRule.CAUSAL,
        }

        #
        # Create inputs for the transformer
        #

        all_prefix_groups = []
        all_timestep_groups = []

        #
        # First, add the task tokens
        #

        for name, tok in self.task_tokenizers.items():
            group_name = f"task_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping task tokenizer: {group_name}")
                continue

            task_tokens = nn.Dense(
                self.token_embedding_size, name=f"{group_name}_projection"
            )(tokenizer_output.tokens)
            # task_tokens shape is (batch, n_tokens, token_embedding_size)

            # Add positional embedding
            task_tokens += self._create_positional_embedding(group_name, task_tokens)

            all_prefix_groups.append(
                PrefixGroup(
                    tokens=task_tokens,
                    mask=tokenizer_output.mask,
                    name=group_name,
                    attention_rules=task_attention_rules,
                )
            )

        #
        # Next, add the observation tokens
        #

        for name, tok in self.observation_tokenizers.items():
            group_name = f"obs_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping observation tokenizer: {group_name}")
                continue

            obs_tokens = nn.Dense(
                self.token_embedding_size, name=f"{group_name}_projection"
            )(tokenizer_output.tokens)
            # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # Add positional embedding
            obs_tokens += self._create_positional_embedding(group_name, obs_tokens)

            # Update mask to account for which timesteps are padding
            obs_pad_mask = jnp.logical_and(
                timestep_pad_mask[:, :, None], tokenizer_output.mask
            )

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=obs_tokens,
                    mask=obs_pad_mask,
                    name=group_name,
                    attention_rules=observation_attention_rules,
                )
            )
        if self.repeat_task_tokens:
            logging.info(
                "repeating task tokens at each observation timestep to perform cross-modal attention"
            )
            # get task tokens
            for tasks in all_prefix_groups:
                # lang (batch, n_tokens, token_embedding_size)
                task_tokens = tasks.tokens[:, jnp.newaxis, :, :]
                ws = all_timestep_groups[0].tokens.shape[1]
                task_tokens = jnp.tile(task_tokens, [1, ws, 1, 1])
                task_pad_mask = tasks.mask[:, jnp.newaxis, :]
                task_pad_mask = jnp.tile(task_pad_mask, [1, ws, 1])
                group_name = f"obs_{tasks.name}"
                all_timestep_groups.append(
                    TimestepGroup(
                        tokens=task_tokens,
                        mask=task_pad_mask,
                        name=group_name,
                        attention_rules=observation_attention_rules,
                    )
                )

        #
        # Then, add the action tokens
        #

        for name, tok in self.action_tokenizers.items():
            group_name = f"action_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            # TODO (chongyiz): add tasks input to the tokenizer
            if actions is None:
                actions = jnp.zeros((batch_size, horizon, tok.action_horizon, tok.action_dim),
                                    dtype=jax.tree_util.tree_leaves(observations)[0].dtype)
                action_pad_mask = jnp.ones_like(actions, dtype=timestep_pad_mask.dtype)
            tokenizer_output: TokenGroup = tok(actions, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping action tokenizer: {group_name}")
                continue

            action_tokens = nn.Dense(
                self.token_embedding_size, name=f"{group_name}_projection"
            )(tokenizer_output.tokens)
            # action_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # Add positional embedding
            action_tokens += self._create_positional_embedding(
                group_name, action_tokens)

            # Update mask to account for which timesteps are padding
            if len(action_pad_mask.shape) == 4:
                action_pad_mask = rearrange(
                    action_pad_mask, "b w h a -> b w (h a)", h=tok.action_horizon, a=tok.action_dim
                )
            aggregated_action_pad_mask = (
                action_pad_mask & timestep_pad_mask[:, :, None] & tokenizer_output.mask
            )

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=action_tokens,
                    mask=aggregated_action_pad_mask,
                    name=group_name,
                    attention_rules=action_attention_rules,
                )
            )
        if self.repeat_task_tokens:
            logging.info(
                "repeating task tokens at each action timestep to perform cross-modal attention"
            )
            # get task tokens
            for tasks in all_prefix_groups:
                # lang (batch, n_tokens, token_embedding_size)
                task_tokens = tasks.tokens[:, jnp.newaxis, :, :]
                ws = all_timestep_groups[0].tokens.shape[1]
                task_tokens = jnp.tile(task_tokens, [1, ws, 1, 1])
                task_pad_mask = tasks.mask[:, jnp.newaxis, :]
                task_pad_mask = jnp.tile(task_pad_mask, [1, ws, 1])
                group_name = f"action_{tasks.name}"
                all_timestep_groups.append(
                    TimestepGroup(
                        tokens=task_tokens,
                        mask=task_pad_mask,
                        name=group_name,
                        attention_rules=action_attention_rules,
                    )
                )

        #
        # Finally, add the readout tokens
        #

        for readout_name in readouts:
            group_name = f"readout_{readout_name}"
            # Readouts do not correspond to any inputs, just positional embeddings
            n_tokens_for_readout = self.readouts[readout_name]
            readout_tokens = jnp.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size)
            )

            # Add positional embedding
            readout_tokens += self._create_positional_embedding(
                group_name, readout_tokens
            )
            readout_mask = jnp.ones((batch_size, horizon, n_tokens_for_readout))

            if readout_name in ['action', 'reward', 'value']:
                readout_attention_rules = {
                    "task_*": AttentionRule.CAUSAL,
                    "obs_*": AttentionRule.CAUSAL,
                    group_name: AttentionRule.CAUSAL,
                }  # Attend to tasks, all previous observations, and *only it's own readout*
            else:
                readout_attention_rules = {
                    "task_*": AttentionRule.CAUSAL,
                    "obs_*": AttentionRule.CAUSAL,
                    "action_*": AttentionRule.CAUSAL,
                    group_name: AttentionRule.CAUSAL,
                }  # Attend to tasks, all previous observations, all previous actions, and *only it's own readout*

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=readout_tokens,
                    mask=readout_mask,
                    name=group_name,
                    attention_rules=readout_attention_rules,
                )
            )

        # Run the transformer!
        assert (
            self.transformer_kwargs.get("add_position_embedding", False) is False
        ), "Already added positional embeddings to the tokens"

        prefix_outputs, timestep_outputs = BlockTransformer(
            self.transformer_kwargs, use_correct_attention=self.use_correct_attention
        )(
            all_prefix_groups,
            all_timestep_groups,
            train=train,
            verbose=verbose,
        )
        outputs = {}
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in prefix_outputs
            }
        )
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
            }
        )

        if len(prefix_outputs) > 0:
            outputs["task"] = TokenGroup.concatenate(
                [TokenGroup(group.tokens, group.mask) for group in prefix_outputs]
            )

        outputs["obs"] = TokenGroup.concatenate(
            [
                TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
                if group.name.startswith("obs_")
            ],
            axis=-2,
        )
        outputs["action"] = TokenGroup.concatenate(
            [
                TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
                if group.name.startswith("action_")
            ],
            axis=-2,
        )

        return outputs

    def _create_positional_embedding(self, name: str, tokens: jax.Array):
        if tokens.ndim == 3:  # for prefixes
            shape = (1, *tokens.shape[-2:])
        elif (
            tokens.ndim == 4
        ):  # for timesteps, create embedding for max_horizon, then truncate
            shape = (1, self.max_horizon, *tokens.shape[-2:])
        else:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}")

        embedding = self.param(
            f"{name}_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            shape,
        )
        if tokens.ndim == 4:
            # Use only the timesteps we receive as input
            embedding = embedding[:, : tokens.shape[1]]
        return jnp.broadcast_to(embedding, tokens.shape)


class ContinuousGaussianActorHead(nn.Module):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action].

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    use_map: bool = False
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()
        self.mean_net = nn.Dense(self.action_horizon * self.action_dim,
                                 kernel_init=default_init(self.final_fc_init_scale))

        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_horizon * self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    'log_stds', nn.initializers.zeros,
                    (self.action_horizon * self.action_dim,)
                )

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        temperature: float = 1.0,
        train: bool = True
    ) -> distrax.Distribution:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # mean = self.mean_proj(embeddings)
        # mean = rearrange(
        #     mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        # )
        # mean = jnp.tanh(mean / self.max_action) * self.max_action

        mean_preds = self.mean_net(embeddings)
        means = rearrange(
            mean_preds, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        )
        if self.state_dependent_std:
            log_stds = self.log_std_net(embeddings)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(mean_preds)
            else:
                log_stds = self.log_stds
        log_stds = rearrange(
            log_stds, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            # chain of bijectors are applied in the reversed order.
            distribution = TransformedWithMode(
                distribution,
                distrax.Chain([distrax.Block(distrax.ScalarAffine(0.0, self.max_action), ndims=1),
                               distrax.Block(distrax.Tanh(), ndims=1),
                               distrax.Block(distrax.ScalarAffine(0.0, 1.0 / self.max_action), ndims=1)]),
            )

        return distribution

    # def predict_action(
    #     self,
    #     transformer_outputs: Dict[str, TokenGroup],
    #     train: bool = True,
    #     *args,
    #     sample_shape: tuple = (),
    #     **kwargs,
    # ) -> jax.Array:
    #     """Convenience methods for predicting actions for the final timestep in the window."""
    #     # only get the last timestep in the window
    #     mean = self(transformer_outputs, train=train)[:, -1]
    #     return jnp.broadcast_to(mean, sample_shape + mean.shape)


class ValueHead(nn.Module):
    """Predicts values / critics.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    use_map: bool = False
    num_ensembles: int = 1

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        if self.num_ensembles > 1:
            dense_layer_cls = ensemblize(nn.Dense, self.num_ensembles)
        else:
            dense_layer_cls = nn.Dense
        self.value_net = dense_layer_cls(1)

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True
    ) -> jax.Array:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # value is (ensemble_size, batch_size, window_size), the ensemble_size is optional.
        v = self.value_net(embeddings).squeeze(-1)

        return v