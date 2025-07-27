import functools
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp



# class ResNetBlock(nn.Module):
#     """ResNet block."""
#
#     filters: int
#     conv: Any
#     norm: Any
#     act: Callable
#     strides: Tuple[int, int] = (1, 1)
#
#     @nn.compact
#     def __call__(self, x):
#         residual = x
#         y = self.conv(self.filters, (3, 3), self.strides)(x)
#         y = self.norm()(y)
#         y = self.act(y)
#         y = self.conv(self.filters, (3, 3))(y)
#         y = self.norm()(y)
#
#         if residual.shape != y.shape:
#             residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
#                 residual
#             )
#             residual = self.norm(name="norm_proj")(residual)
#
#         return self.act(residual + y)


# def upsample_nn(x, factor: int = 2):
#     # Nearest-neighbor 2x up-sample without importing jax.image
#     x = jnp.repeat(x, factor, axis=-3)
#     x = jnp.repeat(x, factor, axis=-2)
#     return x


def upsample_fn(x, factor=2):
    n, h, w, c = x.shape
    x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
    return x


class ResNetUpBlock(nn.Module):
    """Residual block with optional 2x up-sampling on the main and skip paths."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    upsample: bool = False

    @nn.compact
    def __call__(self, x):
        residual = x
        y = x

        # spatial dims x2 and (usually) channels decrease
        if self.upsample:
            y = upsample_fn(x, 2)
            residual = upsample_fn(residual, 2)

        y = self.conv(self.filters, (3, 3))(y)
        # y = self.conv(self.filters, (3, 3), self.strides, padding="SAME")(y)
        y = self.norm()(y)
        y = self.act(y)

        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class GroupNorm(nn.GroupNorm):
    def __call__(self, x, **kwargs):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetDecoder(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]  # shallow -> deep
    block_cls: Any
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "swish"
    conv: Any = nn.Conv
    norm: str = "group"
    final_channels: int = 9
    final_act: str = "tanh"

    @nn.compact
    def __call__(self, latents: jnp.ndarray, train=True):
        x = latents  # latents are the spatial codes from VQ layer: (B, H / 32, W / 32, latent_dim)

        conv = functools.partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "group":
            norm = functools.partial(GroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = functools.partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        # Project codebook embeddings to the deepest encoder width (e.g., 512 for ResNet-34).
        deepest_filters = self.num_filters * (2 ** (len(self.stage_sizes) - 1))
        x = conv(deepest_filters, (3, 3), name="conv_init")(x)
        x = norm(name="norm_init")(x)
        x = act(x)

        # Here we up-sample at the first block of each reversed stage except the deepest.
        # Traverse stages in reverse (deep -> shallow).
        # for i in reversed(range(len(self.stage_sizes))):
        for i, block_size in reversed(list(enumerate(self.stage_sizes))):
            # target_filters = self.num_filters * (2 ** i)
            for j in range(block_size):
            # for j in range(self.stage_sizes[i]):
                upsample = (j == 0 and i != len(self.stage_sizes) - 1)  # up-sample at stage boundary
                x = self.block_cls(
                    filters=self.num_filters * 2 ** i,
                    conv=conv,
                    norm=norm,
                    act=act,
                    upsample=upsample,
                )(x)

        # Invert the encoder stem: max-pool (×2) and initial conv stride-2 (×2) -> two more 2× upsamples
        for k in range(2):
            x = upsample_fn(x, 2)
            x = conv(self.num_filters, (3, 3))(x)
            x = norm(name=f"norm_final_{k}")(x)
            x = act(x)

        # Final 7x7 conv to get image channels; mirror encoder's 7x7 "conv_init".
        x = conv(
            self.final_channels, (7, 7), (1, 1),
            padding=[(3, 3), (3, 3)], name="conv_final"
        )(x)

        if self.final_act == "tanh":
            x = jnp.tanh(x)  # in [-1, 1]
        elif self.final_act == "sigmoid":
            x = nn.sigmoid(x)  # in [0, 1]
        elif self.final_act == "none":
            pass
        else:
            raise ValueError(f"Unknown final_act: {self.final_act}")

        # x = conv(
        #     self.num_filters, (7, 7), (2, 2),
        #     padding=[(3, 3), (3, 3)], name="conv_init"
        # )(x)
        #
        # x = norm(name="norm_init")(x)
        # x = act(x)
        # x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        # for i, block_size in enumerate(self.stage_sizes):
        #     for j in range(block_size):
        #         stride = (2, 2) if i > 0 and j == 0 else (1, 1)
        #         x = self.block_cls(
        #             self.num_filters * 2 ** i,
        #             strides=stride,
        #             conv=conv,
        #             norm=norm,
        #             act=act,
        #         )(x)
        #
        # x = jnp.mean(x, axis=(-3, -2))

        return x



decoder_modules = {
    'resnet_34': functools.partial(
        ResNetDecoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetUpBlock,
    ),
}
