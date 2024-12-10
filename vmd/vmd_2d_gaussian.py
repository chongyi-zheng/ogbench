import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn




class ScoreNetwork(nn.Module):
    hidden_units: int = 512
    init_gamma_0: float = -13.3  # initial gamma_0
    init_gamma_1: float = 5.  # initial gamma_1

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_units)
        self.dense2 = nn.Dense(self.hidden_units)
        self.dense3 = nn.Dense(2)
        self.ff = Base2FourierFeatures()

    def __call__(self, z, gamma_t):
        # Normalize gamma_t
        lb = self.init_gamma_0
        ub = self.init_gamma_1
        gamma_t_norm = ((gamma_t - lb) / (ub - lb)) * 2 - 1  # ---> [-1,+1] (not exactly in this range)

        # Concatenate normalized gamma_t as extra feature
        h = jnp.concatenate([z, gamma_t_norm[:, None]], axis=1)

        # append Fourier features
        h_ff = self.ff(h)
        h = jnp.concatenate([h, h_ff], axis=1)

        # Three dense layers
        h = nn.swish(self.dense1(h))
        h = nn.swish(self.dense2(h))
        h = self.dense3(h)

        return h


class Base2FourierFeatures(nn.Module):
    # Create Base 2 Fourier features
    @nn.compact
    def __call__(self, inputs):
        freqs = jnp.asarray(range(8), dtype=inputs.dtype)  # [0, 1, ..., 7]
        # w = 2. ** freqs * 2 * jnp.pi
        # w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
        # h = jnp.repeat(inputs, len(freqs), axis=-1)
        # h *= w
        # h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)

        # 8 Fourier features for each input dimension
        w = 2.0 ** freqs * 2 * jnp.pi
        w = w[None, None, :]
        h = inputs[..., None]
        h *= w
        h = h.reshape([h.shape[0], -1])
        h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)

        return h


def constant_init(value, dtype='float32'):
    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


# Simple scalar noise schedule, i.e. gamma(t) in the paper:
# gamma(t) = abs(w) * t + b
class NoiseSchedule(nn.Module):
    init_gamma_0: float = -13.3  # initial gamma_0
    init_gamma_1: float = 5.  # initial gamma_1

    def setup(self):
        init_bias = self.init_gamma_0
        init_scale = self.init_gamma_1 - self.init_gamma_0
        self.w = self.param('w', constant_init(init_scale), (1,))
        self.b = self.param('b', constant_init(init_bias), (1,))

    def __call__(self, t):
        return abs(self.w) * t + self.b

class VDM(nn.Module):

    def setup(self):
        self.score_net = ScoreNetwork()
        self.noise_schedule = NoiseSchedule()

    def __call__(self, z, t):
        gamma_t = self.noise_schedule(t)
        return self.score_net(z, gamma_t)

    def score(self, z, t):
        return self.score_net(z, t)

    def gamma(self, t):
        return self.noise_schedule(t)


def main():
    """Define hyper-parameters"""
    # Data hyper-parameters
    num_training_data = 1024
    num_eval_data = 512

    # Optimization hyper-parameters
    learning_rate = 3e-3
    num_train_steps = 20000  # nr of training steps

    rng = jax.random.PRNGKey(seed=0)
    np.random.seed(0)

    """Create datasets"""
    # # Make 8-bit swirl dataset
    # theta = np.sqrt(np.random.rand(N)) * 3 * np.pi  # np.linspace(0,2*pi,100)
    # r_a = 2 * theta + np.pi
    # x = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    # # We use 8 bits, to make this a bit similar to image data, which has 8-bit
    # # color channels.
    # x = 4 * (x + .25 * np.random.randn(N, 2) + 30)
    # x = x.astype('uint8')

    mean = np.ones(2)
    std = 2
    training_data = mean + np.random.normal(size=[num_training_data, 2]) * std
    eval_data = mean + np.random.normal(size=[num_eval_data, 2]) * std

    # plot dataset
    fig = plt.figure()
    plt.scatter(training_data[:, 0], training_data[:, 1], alpha=0.5)
    plt.title("Training data")
    fig.savefig("./2d_gaussian_training_data.png")

    # Initialize model
    model = VDM()
    rng, rng1, rng2 = jax.random.split(rng, 3)
    init_inputs = [jnp.ones((1, 2)), jnp.zeros((1,))]
    params = model.init({"params": rng1, "sample": rng2}, *init_inputs)

    # initialize optimizer
    optimizer = optax.adamw(learning_rate)
    optim_state = optimizer.init(params)

    def encode(x):
        f = (x - training_data.mean(axis=0)) / training_data.std(axis=0)

        return f

    def decode(f, z_0, alpha_0, sigma_0):
        x = f * training_data.std(axis=0) + training_data.std(axis=0)

        logits = z_0 / sigma_0 - alpha_0 / sigma_0 * f
        log_probs = -0.5 * jnp.sum(logits ** 2, axis=-1)

        return x, log_probs

    # def data_decode(z_0_rescaled, gamma_0, vocab_size=256):
    #     # Logits are exact if there are no dependencies between dimensions of x
    #     x_vals = jnp.arange(0, vocab_size)[:, None]
    #     x_vals = jnp.repeat(x_vals, z_0_rescaled.shape[-1], 1)
    #     x_vals = data_encode(x_vals).transpose([1, 0])[None, :, :]
    #     inv_stdev = jnp.exp(-0.5 * gamma_0[..., None])
    #     logits = -0.5 * jnp.square((z_0_rescaled[..., None] - x_vals) * inv_stdev)
    #
    #     logprobs = jax.nn.log_softmax(logits)
    #     return logprobs

    # def data_logprob(x, z_0_rescaled, gamma_0, vocab_size=256):
    #     x = x.round().astype('int32')
    #     x_onehot = jax.nn.one_hot(x, vocab_size)
    #     logprobs = data_decode(z_0_rescaled, gamma_0)
    #     logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2))
    #     return logprob

    # Define training step
    def loss_fn(params, x, rng, T_train=0):
        # gamma = lambda t: model.apply(params, t, method=VDM.gamma)
        # gamma_0, gamma_1 = gamma(0.), gamma(1.)

        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        gamma_1 = model.apply(params, 1.0, method=VDM.gamma)
        sigma_0, sigma_1 = jnp.sqrt(jax.nn.sigmoid(gamma_0)), jnp.sqrt(jax.nn.sigmoid(gamma_1))
        alpha_0, alpha_1 = jnp.sqrt(jax.nn.sigmoid(-gamma_0)), jnp.sqrt(jax.nn.sigmoid(-gamma_1))
        batch_size = x.shape[0]

        # encode
        f = encode(x)

        # 1. RECONSTRUCTION LOSS
        rng, rng1 = jax.random.split(rng)
        eps_0 = jax.random.normal(rng1, shape=f.shape)
        z_0 = alpha_0 * f + sigma_0 * eps_0
        _, log_probs = decode(f, z_0, alpha_0, sigma_0)
        loss_recon = -log_probs

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        loss_klz = 0.5 * jnp.sum((alpha_1 ** 2) * (f ** 2) + sigma_1 ** 2 - 2 * jnp.log(sigma_1) - 1., axis=1)

        # 3. DIFFUSION LOSS
        # sample time steps
        rng, rng1 = jax.random.split(rng)
        t = jax.random.uniform(rng1, shape=(batch_size,))

        # discretize time steps if we're working with discrete time
        if T_train > 0:
            t = jnp.ceil(t * T_train) / T_train

        # sample z_t
        gamma_t = model.apply(params, t, method=VDM.gamma)
        sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))
        alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_t))

        rng, rng1 = jax.random.split(rng)
        eps = jax.random.normal(rng1, shape=f.shape)
        z_t = alpha_t[:, None] * f + sigma_t[:, None] * eps

        # compute predicted noise
        eps_hat = model.apply(params, z_t, gamma_t, method=VDM.score)
        # compute MSE of predicted noise
        loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=1)

        if T_train == 0:
            # loss for infinite depth T, i.e. continuous time
            gamma = lambda t: model.apply(params, t, method=VDM.gamma)

            _, g_t_grad = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
            loss_diff = 0.5 * g_t_grad * loss_diff_mse
        else:
            # loss for finite depth T, i.e. discrete time
            s = t - (1. / T_train)
            gamma_s = model.apply(params, s, method=VDM.gamma)
            loss_diff = 0.5 * T_train * jnp.expm1(gamma_t - gamma_s) * loss_diff_mse


        # Compute loss in terms of bits per dimension
        # rescale_to_bpd = 1. / (np.prod(x.shape[1:]) * np.log(2.))
        # bpd_latent = jnp.mean(loss_klz) * rescale_to_bpd
        # bpd_recon = jnp.mean(loss_recon) * rescale_to_bpd
        # bpd_diff = jnp.mean(loss_diff) * rescale_to_bpd
        # bpd = bpd_recon + bpd_latent + bpd_diff
        # loss = bpd
        loss_recon = jnp.mean(loss_recon)
        loss_latent = jnp.mean(loss_klz)
        loss_diff = jnp.mean(loss_diff)
        loss = loss_recon + loss_latent + loss_diff

        metrics = {
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
            "loss_diff": loss_diff,
        }

        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(rng, optim_state, params, x):
        rng, rng1 = jax.random.split(rng)
        (loss, metrics), grads = grad_fn(params, x, rng1)
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return rng, optim_state, params, loss, metrics

    # training loop (should take ~20 mins)
    losses = []
    for i in tqdm.trange(num_train_steps):
        rng, optim_state, params, loss, _metrics = train_step(rng, optim_state, params, training_data)
        losses.append(loss)

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./2d_gaussian_losses.png")
    print("Finish training...")

    # Plot the learned endpoints of the noise schedule
    print('gamma_0', model.apply(params, 0., method=VDM.gamma))
    print('gamma_1', model.apply(params, 1., method=VDM.gamma))

    # Generate samples
    def data_generate_x(z_0, gamma_0, rng):
        var_0 = nn.sigmoid(gamma_0)
        z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
        logits = data_decode(z_0_rescaled, gamma_0)
        samples = jax.random.categorical(rng, logits)
        return samples

    # t_end is integer between 0 and T_sample
    def sample_fn(rng, params, N_sample, T_sample):
        # sample z_0 from the diffusion model
        rng, rng1 = jax.random.split(rng)
        z = [jax.random.normal(rng1, (N_sample, 2))]
        x_pred = []

        for i in tqdm.trange(T_sample):
            rng, rng1 = jax.random.split(rng)
            _z, _x_pred = sample_step(i, T_sample, z[-1], rng1)
            z.append(_z)
            x_pred.append(_x_pred)

        gamma_0 = model.apply(params, 0., method=VDM.gamma)
        x_sample = data_generate_x(z[-1], gamma_0, rng)

        return z, x_pred, x_sample

    def sample_step(i, T_sample, z_t, rng):

        eps = jax.random.normal(rng, z_t.shape)

        t = (T_sample - i) / T_sample
        s = (T_sample - i - 1) / T_sample

        gamma_s = model.apply(params, s, method=Model.gamma)
        gamma_t = model.apply(params, t, method=Model.gamma)
        gamma_s *= jnp.ones((z_t.shape[0],), gamma_t.dtype)
        gamma_t *= jnp.ones((z_t.shape[0],), gamma_t.dtype)

        eps_hat = model.apply(params, z_t, gamma_t, method=Model.score)
        a = nn.sigmoid(-gamma_s)[:, None]
        b = nn.sigmoid(-gamma_t)[:, None]
        c = - jnp.expm1(gamma_s - gamma_t)[:, None]
        sigma_t = jnp.sqrt(nn.sigmoid(gamma_t))[:, None]

        # equations (32) and (33) in the paper.
        z_s = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + \
              jnp.sqrt((1. - a) * c) * eps

        alpha_t = jnp.sqrt(1 - sigma_t)
        x_pred = (z_t - sigma_t * eps_hat) / alpha_t

        return z_s, x_pred

    rng, rng1 = jax.random.split(rng)
    z, x_pred, _ = sample_fn(rng1, params, N_sample=1024, T_sample=200)

    # Create square plot:
    def plot(data, color='blue'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        plt.scatter(data[:, 0], data[:, 1], alpha=0.1, c=color)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

    plot(z[199])

    def sample_q_t(rng, t, x):
        f = data_encode(x)
        gamma_t = model.apply(params, t, method=Model.gamma)
        var_t = nn.sigmoid(gamma_t)[:, None]
        eps = jax.random.normal(rng, shape=x.shape)
        z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
        return z_t

    # loss_diff, loss_klz, loss_recon = vdm.apply(unreplicate(pstore.params), ims, 0 * lbs, rngs={"sample": rng})
    # losses = jax.tree_map(lambda x: np.mean(x) / (onp.prod(ims.shape[1:]) * np.log(2)),
    #                       {"loss_diff": loss_diff, "loss_klz": loss_klz, "loss_recon": loss_recon})
    # print(losses, "\n", sum(losses.values()))


if __name__ == "__main__":
    main()
