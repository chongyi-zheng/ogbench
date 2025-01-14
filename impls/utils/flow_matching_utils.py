import dataclasses
from typing import Optional, Union, Tuple, Sequence

import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (jnp.ndarray): :math:`\alpha_t`, shape (...).
        sigma_t (jnp.ndarray): :math:`\sigma_t`, shape (...).
        d_alpha_t (jnp.ndarray): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (jnp.ndarray): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: jnp.ndarray = dataclasses.field(metadata={"help": "alpha_t"})
    sigma_t: jnp.ndarray = dataclasses.field(metadata={"help": "sigma_t"})
    d_alpha_t: jnp.ndarray = dataclasses.field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: jnp.ndarray = dataclasses.field(metadata={"help": "Derivative of sigma_t."})


@dataclasses.dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (jnp.ndarray): the target sample :math:`X_1`.
        x_0 (jnp.ndarray): the source sample :math:`X_0`.
        t (jnp.ndarray): the time sample :math:`t`.
        x_t (jnp.ndarray): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (jnp.ndarray): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).

    """
    x_1: jnp.ndarray = dataclasses.field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: jnp.ndarray = dataclasses.field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: jnp.ndarray = dataclasses.field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: jnp.ndarray = dataclasses.field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: jnp.ndarray = dataclasses.field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


class CondOTScheduler(flax.struct.PyTreeNode):
    """CondOT Scheduler. (Linear Path)"""

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=jnp.ones_like(t),
            d_sigma_t=-jnp.ones_like(t),
        )


class CosineScheduler(flax.struct.PyTreeNode):
    """Cosine Scheduler."""

    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=jnp.sin(jnp.pi / 2 * t),
            sigma_t=jnp.cos(jnp.pi / 2 * t),
            d_alpha_t=jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t),
            d_sigma_t=-jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t),
        )


class AffineCondProbPath(flax.struct.PyTreeNode):
    r"""The ``AffineCondProbPath`` class represents a specific type of probability path where the transformation between distributions is affine.
    An affine transformation can be represented as:

    .. math::

        X_t = \alpha_t X_1 + \sigma_t X_0,

    where :math:`X_t` is the transformed data point at time `t`. :math:`X_0` and :math:`X_1` are the source and target data points, respectively. :math:`\alpha_t` and :math:`\sigma_t` are the parameters of the affine transformation at time `t`.

    The scheduler is responsible for providing the time-dependent parameters :math:`\alpha_t` and :math:`\sigma_t`, as well as their derivatives, which define the affine transformation at any given time `t`.

    Args:
        scheduler (Scheduler): An instance of a scheduler that provides the parameters :math:`\alpha_t`, :math:`\sigma_t`, and their derivatives over time.

    """
    scheduler: flax.struct.PyTreeNode

    def __call__(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> PathSample:
        r"""Sample from the affine probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
        | return :math:`X_0, X_1, X_t = \alpha_t X_1 + \sigma_t X_0`, and the conditional velocity at :math:`X_t, \dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0`.

        Args:
            x_0 (jnp.ndarray): source data point, shape (batch_size, ...).
            x_1 (jnp.ndarray): target data point, shape (batch_size, ...).
            t (jnp.ndarray): times in [0,1], shape (batch_size, ).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        assert (
                len(t.shape) == 1
        ), f"The time vector t must have shape [batch_size, ]. Got {t.shape}."
        assert (
                t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"
        assert (
                x_0.shape == x_1.shape
        ), f"Target data point x_1 and source data point x_) must have same shape. Got {x_1.shape} for x_1 and {x_0.shape} for x_0"

        scheduler_output = self.scheduler(t)

        alpha_t = jnp.expand_dims(scheduler_output.alpha_t, np.arange(1, len(x_1.shape)))
        sigma_t = jnp.expand_dims(scheduler_output.sigma_t, np.arange(1, len(x_1.shape)))
        d_alpha_t = jnp.expand_dims(scheduler_output.d_alpha_t, np.arange(1, len(x_1.shape)))
        d_sigma_t = jnp.expand_dims(scheduler_output.d_sigma_t, np.arange(1, len(x_1.shape)))

        # construct xt ~ p_t(x|x1).
        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)


# class ODESolver(flax.struct.PyTreeNode):
#     velocity_field: nn.Module

#     def sample(
#         self,
#         x_init: jnp.ndarray,
#         step_size: Optional[float],
#         method: str = "euler",
#         atol: float = 1e-5,
#         rtol: float = 1e-5,
#         time_grid: jnp.ndarray = jnp.asarray([0.0, 1.0]),
#         return_intermediates: bool = False,
#         enable_grad: bool = False,
#         **model_extras,
#     ) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
#         pass

#     def compute_likelihood(
#         self,
#         x_1: jnp.ndarray,
#         step_size: Optional[float],
#         method: str = "euler",
#         atol: float = 1e-5,
#         rtol: float = 1e-5,
#         time_grid: jnp.ndarray = jnp.asarray([1.0, 0.0]),
#         return_intermediates: bool = False,
#         exact_divergence: bool = False,
#         enable_grad: bool = False,
#         **model_extras,
#     ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[Sequence[jnp.ndarray], jnp.ndarray]]:
#         # TODO (chongyi)
#         pass



cond_prob_path_class = dict(
    AffineCondProbPath=AffineCondProbPath
)

scheduler_class = dict(
    CondOTScheduler=CondOTScheduler,
    CosineScheduler=CosineScheduler,
)
