import flax.linen as nn
import jax.numpy as jnp

import jax


def squared_euclidean_distance(a: jnp.ndarray,
                               b: jnp.ndarray,
                               b2: jnp.ndarray = None) -> jnp.ndarray:
    """Computes the pairwise squared Euclidean distance.

    Args:
        a: float32: (n, d): An array of points.
        b: float32: (m, d): An array of points.
        b2: float32: (d, m): b square transpose.

    Returns:
        d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
        a[i] and b[j].
    """
    if b2 is None:
        b2 = jnp.sum(b.T ** 2, axis=0, keepdims=True)
    a2 = jnp.sum(a ** 2, axis=1, keepdims=True)
    ab = jnp.matmul(a, b.T)
    d = a2 - 2 * ab + b2
    return d


class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""
    # train: bool
    latent_dim: int = 512
    codebook_size: int = 1024
    # entropy_loss_ratio: float = 0.1
    # entropy_loss_type: str = 'softmax'
    # entropy_temperature: float = 0.01
    # commitment_cost: float = 0.25

    def setup(self):
        self.codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform"),
            (self.codebook_size, self.latent_dim)
        )

    @nn.compact
    def __call__(self, x):
        # codebook_size = self.codebook_size
        # emb_dim = x.shape[-1]
        # codebook = self.param(
        #     "codebook",
        #     jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform"),
        #     (self.codebook_size, emb_dim))
        codebook = jnp.asarray(self.codebook)  # (codebook_size, emb_dim)
        distances = jnp.reshape(
            squared_euclidean_distance(jnp.reshape(x, (-1, self.latent_dim)), codebook),
            x.shape[:-1] + (self.codebook_size, )
        )  # [x, codebook_size] similarity matrix.
        encoding_indices = jnp.argmin(distances, axis=-1)
        encoding_onehot = jax.nn.one_hot(encoding_indices, self.codebook_size)
        # quantized = self.quantize(encoding_onehot)
        quantized = jnp.dot(encoding_onehot, codebook)
        # result_dict = dict()
        # if self.train:
        #     e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2) * self.commitment_cost
        #     q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
        #     entropy_loss = 0.0
        #     if self.entropy_loss_ratio != 0:
        #         entropy_loss = entropy_loss_fn(
        #             -distances,
        #             loss_type=self.entropy_loss_type,
        #             temperature=self.entropy_temperature,
        #         ) * self.entropy_loss_ratio
        #     e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
        #     q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
        #     entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
        #     loss = e_latent_loss + q_latent_loss + entropy_loss
        #     result_dict = dict(
        #         quantizer_loss=loss,
        #         e_latent_loss=e_latent_loss,
        #         q_latent_loss=q_latent_loss,
        #         entropy_loss=entropy_loss)
        #     quantized = x + jax.lax.stop_gradient(quantized - x)
        #
        # result_dict.update({
        #     "z_ids": encoding_indices,
        # })
        # return quantized, result_dict
        return quantized, distances

    # def quantize(self, encoding_onehot: jnp.ndarray) -> jnp.ndarray:
    #     codebook = jnp.asarray(self.variables["params"]["codebook"])
    #     return jnp.dot(encoding_onehot, codebook)

    # def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    #     codebook = self.variables["params"]["codebook"]
    #     return jnp.take(codebook, ids, axis=0)
