from typing import Any, Optional, Sequence

import math
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class SimBaMLP(nn.Module):
    """Multi-layer perceptron with residual blocks from https://arxiv.org/pdf/2410.09754.

        Attributes:
            num_residual_blocks: Number of residual blocks.
            hidden_dims: Hidden layer dimensions. (Only hidden_dims[0] and hidden_dims[-1] are used.)
            layer_norm: Whether to apply layer normalization.
        """

    num_residual_blocks: int
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dims[0], kernel_init=nn.initializers.orthogonal(1))(x)
        for _ in range(self.num_residual_blocks):
            res = x
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.Dense(self.hidden_dims[0] * 4, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dims[0], kernel_init=nn.initializers.he_normal())(x)
            x = res + x

        if self.layer_norm:
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.hidden_dims[-1], kernel_init=nn.initializers.orthogonal(1))(x)
        if self.activate_final:
            x = self.activations(x)

        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def __getattr__(self, name):
        return getattr(self.distribution, name)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    network_type: str = 'mlp'
    layer_norm: bool = True
    num_residual_blocks: int = 1
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.network_type == 'mlp':
            mlp_class = MLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, 1),
                          activate_final=False, layer_norm=self.layer_norm)
        elif self.network_type == 'simba':
            mlp_class = SimBaMLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, 1), num_residual_blocks=self.num_residual_blocks,
                          activate_final=False, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(**kwargs)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        network_type: Type of MLP network. ('mlp' or 'simba')
        num_residual_blocks: Number of residual blocks.
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    network_type: str = 'mlp'
    num_residual_blocks: int = 1
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            self.actor_net = MLP(
                self.hidden_dims,
                activate_final=True,
                layer_norm=self.layer_norm
            )

        elif self.network_type == 'simba':
            self.actor_net = SimBaMLP(
                self.num_residual_blocks,
                self.hidden_dims,
                layer_norm=self.layer_norm
            )

        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        network_type: Type of MLP network. ('mlp' or 'simba')
        num_residual_blocks: Number of residual blocks.
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    network_type: str = 'mlp'
    num_residual_blocks: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    gc_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
        elif self.network_type == 'simba':
            network_module = SimBaMLP

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module,  self.num_ensembles)

        if self.network_type == 'mlp':
            value_net = network_module(
                (*self.hidden_dims, 1),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        elif self.network_type == 'simba':
            value_net = network_module(
                self.num_residual_blocks,
                (*self.hidden_dims, 1),
                layer_norm=self.layer_norm
            )

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    network_type: str = 'mlp'
    num_residual_blocks: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self) -> None:
        if self.network_type == 'mlp':
            network_module = MLP
        elif self.network_type == 'simba':
            network_module = SimBaMLP

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module,  self.num_ensembles)

        if self.network_type == 'mlp':
            self.phi = network_module(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False, layer_norm=self.layer_norm
            )
            self.psi = network_module(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False, layer_norm=self.layer_norm
            )
        elif self.network_type == 'simba':
            self.phi = network_module(
                self.num_residual_blocks,
                (*self.hidden_dims, self.latent_dim),
                layer_norm=self.layer_norm
            )
            self.psi = network_module(
                self.num_residual_blocks,
                (*self.hidden_dims, self.latent_dim),
                layer_norm=self.layer_norm
            )

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        # v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if len(phi.shape) == 2:  # Non-ensemble.
            v = jnp.einsum('ik,jk->ij', phi, psi) / jnp.sqrt(self.latent_dim)
        else:
            v = jnp.einsum('eik,ejk->eij', phi, psi) / jnp.sqrt(self.latent_dim)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class SinusoidalPosEmb(nn.Module):
    emb_dim: int

    def __call__(self, x):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        # emb = x[:, None] * emb[None, :]
        emb = x[..., None] * emb[None]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class GCFMVectorField(nn.Module):
    """Goal-conditioned flow matching vector field function.

    This module can be used for both value velocity field u(s, g) and critic velocity filed u(s, a, g) functions.

    Attributes:
        network_type: Type of MLP network. ('mlp')
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    vector_dim: int
    hidden_dims: Sequence[int]
    time_dim: int = 64
    network_type: str = 'mlp'
    layer_norm: bool = True
    num_residual_blocks: int = 1
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, self.vector_dim),
                          activate_final=False, layer_norm=self.layer_norm)
        else:
            network_module = SimBaMLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, self.vector_dim),
                          num_residual_blocks=self.num_residual_blocks,
                          activate_final=False, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module, self.num_ensembles)

        self.velocity_field_net = network_module(**kwargs)
        self.time_embedding = SinusoidalPosEmb(emb_dim=self.time_dim)

    def __call__(self, noisy_goals, times, observations=None, actions=None, commanded_goals=None):
        """Return the value/critic velocity field.

        Args:
            noisy_goals: Noisy goals.
            times: Times.
            observations: Observations.
            actions: Actions (Optional).
        """
        if self.goal_encoder is not None:
            noisy_goals = self.goal_encoder(noisy_goals)
        
        if self.state_encoder is not None:
            # This will be all nans if observations are all nan
            observations = self.state_encoder(observations)

        times = self.time_embedding(times)
        inputs = [noisy_goals, times]
        if observations is None:
            inputs.append(observations)
        if commanded_goals is not None:
            # conds = jnp.concatenate([conds, commanded_goals], axis=-1)
            inputs.append(commanded_goals)
        if actions is not None:
            # This will be all nans if both observations and actions are all nan
            # conds = jnp.concatenate([conds, actions], axis=-1)
            inputs.append(actions)
        # inputs = jnp.concatenate([noisy_goals, times, conds], axis=-1)
        inputs = jnp.concatenate(inputs, axis=-1)

        # vf = self.velocity_field_net(h)
        vf = self.velocity_field_net(inputs)

        return vf


class GCFMBilinearVectorField(nn.Module):
    vector_dim: int
    latent_dim: int
    hidden_dims: Sequence[int]
    time_dim: int = 64
    network_type: str = 'mlp'
    layer_norm: bool = True
    num_residual_blocks: int = 1
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
        else:
            raise NotImplementedError

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module, self.num_ensembles)

        if self.network_type == 'mlp':
            self.phi = network_module(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
            self.psi = network_module(
                (*self.hidden_dims, self.latent_dim * self.vector_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            raise NotImplementedError

        self.time_embedding = SinusoidalPosEmb(emb_dim=self.time_dim)

    def sa_reprs(self, observations, actions=None, commanded_goals=None):
        if self.state_encoder is not None:
            # This will be all nans if observations are all nan
            observations = self.state_encoder(observations)

        conds = observations
        if commanded_goals is not None:
            conds = jnp.concatenate([conds, commanded_goals], axis=-1)
        if actions is not None:
            conds = jnp.concatenate([conds, actions], axis=-1)

        phi = self.phi(conds)

        return phi

    def ngt_reprs(self, noisy_goals, times):
        if self.goal_encoder is not None:
            noisy_goals = self.goal_encoder(noisy_goals)

        times = self.time_embedding(times)
        psi = self.psi(jnp.concatenate([noisy_goals, times], axis=-1))

        return psi

    def __call__(self, noisy_goals, times, observations, actions=None, commanded_goals=None):
        """Return the value/critic velocity field.

        Args:
            noisy_goals: Noisy goals.
            times: Times.
            observations: Observations.
            actions: Actions (Optional).
        """
        # if self.goal_encoder is not None:
        #     noisy_goals = self.goal_encoder(noisy_goals)
        #
        # if self.state_encoder is not None:
        #     # This will be all nans if observations are all nan
        #     observations = self.state_encoder(observations)
        #
        # conds = observations
        # if commanded_goals is not None:
        #     conds = jnp.concatenate([conds, commanded_goals], axis=-1)
        # if actions is not None:
        #     conds = jnp.concatenate([conds, actions], axis=-1)
        # times = self.time_embedding(times)
        #
        # phi = self.phi(conds)
        # psi = self.psi(jnp.concatenate([noisy_goals, times], axis=-1))

        phi = self.sa_reprs(observations, actions, commanded_goals)
        psi = self.ngt_reprs(noisy_goals, times)

        if self.num_ensembles > 1:
            psi = psi.reshape([self.num_ensembles, -1, self.latent_dim, self.vector_dim])
            vf = jnp.einsum('eik,ejkl->eijl', phi, psi) / jnp.sqrt(self.latent_dim)
        else:
            psi = psi.reshape([-1, self.latent_dim, self.vector_dim])
            vf = jnp.einsum('ik,jkl->ijl', phi, psi) / jnp.sqrt(self.latent_dim)

        return vf


class GCActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
    """

    vector_dim: int
    hidden_dims: Sequence[int]
    network_type: str = 'mlp'
    layer_norm: bool = True
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
        else:
            raise NotImplementedError

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module, self.num_ensembles)

        if self.network_type == 'mlp':
            velocity_field_net = network_module(
                (*self.hidden_dims, self.vector_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            raise NotImplementedError

        self.velocity_field_net = velocity_field_net

    @nn.compact
    def __call__(self, noisy_actions, times, observations, goals=None):
        """Return the vectors at the given states, actions, and times.

        Args:
            noisy_actions: Actions.
            observations: Observations.
            times: Times (optional).
            goals: Goals (optional).
        """
        if self.goal_encoder is not None and goals is not None:
            goals = self.goal_encoder(goals)

        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        conds = observations
        if goals is not None:
            conds = jnp.concatenate([conds, goals], axis=-1)

        if len(times.shape) == 1:
            times = jnp.expand_dims(times, axis=-1)

        inputs = jnp.concatenate([noisy_actions, times, conds], axis=-1)

        vf = self.velocity_field_net(inputs)

        return vf


class GCFMValue(nn.Module):
    """Goal-conditioned flow matching value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.
    We typically use this module to distill the ODE solutions.

    Attributes:
        network_type: Type of MLP network. ('mlp' or 'simba')
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of value function ensembles.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    output_dim: int = 1
    network_type: str = 'mlp'
    layer_norm: bool = True
    activate_final: bool = False
    num_residual_blocks: int = 1
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, self.output_dim),
                          activate_final=self.activate_final, layer_norm=self.layer_norm)
        else:
            network_module = SimBaMLP
            kwargs = dict(hidden_dims=(*self.hidden_dims, self.output_dim),
                          num_residual_blocks=self.num_residual_blocks,
                          activate_final=self.activate_final, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module, self.num_ensembles)

        # if self.network_type == 'mlp':
        #     value_net = network_module(
        #         (*self.hidden_dims, self.output_dim),
        #         activate_final=self.activate_final,
        #         layer_norm=self.layer_norm
        #     )
        # else:
        #     raise NotImplementedError

        self.value_net = network_module(**kwargs)

    def __call__(self, goals, observations, actions=None, commanded_goals=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        conds = observations
        if commanded_goals is not None:
            conds = jnp.concatenate([conds, commanded_goals], axis=-1)
        if actions is not None:
            conds = jnp.concatenate([conds, actions], axis=-1)
        inputs = jnp.concatenate([goals, conds], axis=-1)

        if self.output_dim == 1:
            output = self.value_net(inputs).squeeze(-1)
        else:
            output = self.value_net(inputs)

        return output


class GCFMBilinearValue(nn.Module):
    """Bilinear goal-conditioned flow matching value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.
    We typically use this module to distill the ODE solutions.

    Attributes:
        network_type: Type of MLP network. ('mlp' or 'simba')
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    output_dim: int = 1
    network_type: str = 'mlp'
    layer_norm: bool = True
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        if self.network_type == 'mlp':
            network_module = MLP
        else:
            raise NotImplementedError

        if self.num_ensembles > 1:
            network_module = ensemblize(network_module, self.num_ensembles)

        if self.network_type == 'mlp':
            self.phi = network_module(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
            self.psi = network_module(
                (*self.hidden_dims, self.latent_dim * self.output_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            raise NotImplementedError

    def __call__(self, goals, observations, actions=None, commanded_goals=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """

        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        conds = observations
        if commanded_goals is not None:
            conds = jnp.concatenate([conds, commanded_goals], axis=-1)
        if actions is not None:
            # This will be all nans if both observations and actions are all nan
            conds = jnp.concatenate([conds, actions], axis=-1)
        # inputs = jnp.concatenate([goals, conds], axis=-1)
        # inputs = jax.lax.select(
        #     jnp.logical_not(jnp.all(jnp.isnan(conds))),
        #     jnp.concatenate([inputs, conds], axis=-1),
        #     inputs
        # )
        # phi = self.phi(goals).reshape(
        #     [self.num_ensembles, -1, self.output_dim, self.latent_dim])
        # psi = self.psi(conds)
        phi = self.phi(conds)
        psi = self.psi(goals)
        if self.num_ensembles > 1:
            psi = psi.reshape([self.num_ensembles, -1, self.latent_dim, self.output_dim])
            output = jnp.einsum('eik,ejkl->eijl', phi, psi) / jnp.sqrt(self.latent_dim)
        else:
            psi = psi.reshape([-1, self.latent_dim, self.output_dim])
            output = jnp.einsum('ik,jkl->ijl', phi, psi) / jnp.sqrt(self.latent_dim)

        return output
