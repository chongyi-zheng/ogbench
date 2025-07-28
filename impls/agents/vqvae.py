from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.decoders import decoder_modules
from utils.quantizer import VectorQuantizer
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class VQVAEAgent(flax.struct.PyTreeNode):
    """Vector Quantized-Variational AutoEncoder (VQ-VAE) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def entropy_loss_fn(affinity, loss_type="softmax", temperature=1.0):
        """Calculates the entropy loss. Affinity is the similarity/distance matrix."""
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        flat_affinity /= temperature
        probs = jax.nn.softmax(flat_affinity, axis=-1)
        log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
        if loss_type == "softmax":
            target_probs = probs
        elif loss_type == "argmax":
            codes = jnp.argmax(flat_affinity, axis=-1)
            onehots = jax.nn.one_hot(
                codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype)
            onehots = probs - jax.lax.stop_gradient(probs - onehots)
            target_probs = onehots
        else:
            raise ValueError("Entropy loss {} not supported".format(loss_type))
        avg_probs = jnp.mean(target_probs, axis=0)
        avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
        sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
        loss = sample_entropy - avg_entropy
        return loss

    def vqvae_loss(self, batch, grad_params):
        """Compute the VQVAE loss."""
        images = batch['observations'].astype(jnp.float32) / 127.5 - 1.0  # put inputs in [-1, 1]
        latents = self.network.select('encoder')(batch['observations'], flatten=False, params=grad_params)  # the encoder will put inputs in [-1, 1] internally
        quantized_latents, distances = self.network.select('quantizer')(latents, params=grad_params)
        reconstructed_images = self.network.select('decoder')(
            latents + jax.lax.stop_gradient(quantized_latents - latents),
            params=grad_params
        )
        recon_loss = jnp.mean((reconstructed_images - images) ** 2)

        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized_latents) - latents) ** 2)
        q_latent_loss = jnp.mean((quantized_latents - jax.lax.stop_gradient(latents)) ** 2)
        entropy_loss = self.entropy_loss_fn(
            -distances,
            loss_type=self.config['entropy_loss_type'],
            temperature=self.config['entropy_temperature'],
        )
        e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
        q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
        entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
        quantizer_loss = self.config['commitment_cost'] * e_latent_loss + q_latent_loss + self.config['entropy_loss_ratio'] * entropy_loss
        loss = recon_loss + self.config['quantizer_loss_ratio'] * quantizer_loss

        info = {
            'loss': loss,
            'recon_loss': recon_loss,
            'e_latent_loss': e_latent_loss,
            'q_latent_loss': q_latent_loss,
            'entropy_loss': entropy_loss,
        }

        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        # rng = rng if rng is not None else self.rng

        vqvae_loss, vqvae_info = self.vqvae_loss(batch, grad_params)
        for k, v in vqvae_info.items():
            info[f'vqvae/{k}'] = v

        loss = vqvae_loss
        return loss, info

    # def target_update(self, network, module_name):
    #     """Update the target network."""
    #     new_target_params = jax.tree_util.tree_map(
    #         lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
    #         self.network.params[f'modules_{module_name}'],
    #         self.network.params[f'modules_target_{module_name}'],
    #     )
    #     network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # self.target_update(new_network, 'quantizer')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flatten',))
    def encode(self, images, flatten=False):
        latents = self.network.select('encoder')(images, flatten=False)
        quantized_latents, _ = self.network.select('quantizer')(latents)
        if flatten:
            quantized_latents = quantized_latents.reshape(quantized_latents.shape[0], -1)
        return quantized_latents

    @jax.jit
    def decode(self, quantized_latents):
        reconstructed_images = self.network.select('decoder')(quantized_latents)
        return reconstructed_images

    @jax.jit
    def reconstruct(self, images):
        quantized_latents = self.encode(images)
        reconstructed_images = self.decode(quantized_latents)
        return reconstructed_images

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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # (chongyi): make this 512 configable
        ex_latents = jnp.ones((ex_actions.shape[0], 4, 4, 512), dtype=ex_actions.dtype)
        # ex_latent_idxs = jnp.ones((ex_actions.shape[0],), dtype=jnp.int32)

        # Define encoder.
        assert config['encoder'] is not None
        assert config['decoder'] is not None

        # Define encoder, decoder, and quantizer networks.
        encoder_def = encoder_modules[config['encoder']]()
        decoder_def = decoder_modules[config['decoder']]()
        quantizer_def = VectorQuantizer(
            latent_dim=512,  # make this 512 configable
            codebook_size=config['codebook_size'],
            # entropy_loss_ratio=config['entropy_loss_ratio'],
            # entropy_loss_type=config['entropy_loss_type'],
            # entropy_temperature=config['entropy_temperature'],
            # commitment_cost=config['commitment_cost'],
        )

        network_info = dict(
            encoder=(encoder_def, (ex_observations)),
            quantizer=(quantizer_def, (ex_latents)),
            # target_quantizer=(copy.deepcopy(quantizer_def), (ex_latents)),
            decoder=(decoder_def, (ex_latents)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # params = network.params
        # params['modules_target_quantizer'] = params['modules_quantizer']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='vqvae',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            codebook_size=1024,  # Codebook size.
            quantizer_loss_ratio=1.0,  # Quantizer loss weight.
            entropy_loss_ratio=0.1,  # Entropy loss weight.
            entropy_loss_type='softmax',  # Entropy loss type.
            entropy_temperature=0.01,  # Entropy temperature.
            commitment_cost=0.25,  # Commitment cost.
            actor_freq=1,  # Unused.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name ('resnet_34', etc.).
            decoder=ml_collections.config_dict.placeholder(str),  # Visual decoder name ('resnet_34', etc.).
        )
    )
    return config
