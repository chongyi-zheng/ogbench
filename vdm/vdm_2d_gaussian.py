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
    learning_rate = 3e-4
    num_train_steps = 50_000  # nr of training steps

    rng = jax.random.PRNGKey(seed=np.random.randint(0, 2 ** 32))
    np.random.seed(np.random.randint(0, 2 ** 32))

    """Create datasets"""
    def sample_data(num_data):
        # 2d swirl data
        # theta = np.sqrt(np.random.rand(num_data)) * 3 * np.pi
        # r_a = 2 * theta + np.pi
        # data = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T

        # gaussian data
        mean = np.ones(2) * 2
        std = 2
        data = mean + np.random.normal(size=[num_data, 2]) * std

        return data

    training_data = sample_data(num_training_data)
    eval_data = sample_data(num_eval_data)

    # plot dataset
    axis_lim = 10
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    plt.scatter(training_data[:, 0], training_data[:, 1], alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-axis_lim, 0, axis_lim])
    ax.set_yticks([-axis_lim, 0, axis_lim])
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_title("Training data")
    fig.savefig("./gaussian_training_data.png")

    """Training model"""
    model = VDM()
    rng, rng1, rng2 = jax.random.split(rng, 3)
    init_inputs = [jnp.ones((1, 2)), jnp.zeros((1,))]
    params = model.init({"params": rng1, "sample": rng2}, *init_inputs)

    # initialize optimizer
    optimizer = optax.adamw(learning_rate)
    # optimizer = optax.chain(
    #     optax.scale_by_schedule(optax.cosine_decay_schedule(1, num_train_steps, 1e-4)),
    #     optax.adamw(learning_rate),
    # )
    optim_state = optimizer.init(params)

    def normalize(x):
        f = (x - training_data.mean(axis=0)) / training_data.std(axis=0)

        return f

    def unnormalize(f):
        x = f * training_data.std(axis=0) + training_data.mean(axis=0)

        return x

    def encode(params, x, rng):
        f = normalize(x)

        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        sigma_0 = jnp.sqrt(jax.nn.sigmoid(gamma_0))
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        eps_0 = jax.random.normal(rng, shape=f.shape)
        z_0 = alpha_0 * f + sigma_0 * eps_0

        return z_0

    def decode(params, z_0, rng):
        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        sigma_0 = jnp.sqrt(jax.nn.sigmoid(gamma_0))
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        f = z_0 / alpha_0
        eps_0 = jax.random.normal(rng, shape=f.shape)
        sampled_f = z_0 / alpha_0 + sigma_0 / alpha_0 * eps_0

        x = unnormalize(f)
        sampled_x = unnormalize(sampled_f)

        return x, sampled_x

    def decode_log_probs(params, z_0, x):
        f = normalize(x)

        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        sigma_0 = jnp.sqrt(jax.nn.sigmoid(gamma_0))
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        logits = z_0 / sigma_0 - alpha_0 / sigma_0 * f
        log_probs = -0.5 * jnp.sum(logits ** 2, axis=-1)

        return log_probs

    # Define training step
    def loss_fn(params, x, rng, T_train=0):
        # gamma = lambda t: model.apply(params, t, method=VDM.gamma)
        # gamma_0, gamma_1 = gamma(0.), gamma(1.)

        # gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        # gamma_1 = model.apply(params, 1.0, method=VDM.gamma)
        # sigma_1 = jnp.sqrt(jax.nn.sigmoid(gamma_1))
        # alpha_1 = jnp.sqrt(jax.nn.sigmoid(-gamma_1))
        batch_size = x.shape[0]

        # 1. RECONSTRUCTION LOSS
        rng, rng1 = jax.random.split(rng)
        z_0 = encode(params, x, rng1)
        log_probs = decode_log_probs(params, z_0, x)
        loss_recon = -log_probs

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        gamma_1 = model.apply(params, 1.0, method=VDM.gamma)
        sigma_1 = jnp.sqrt(jax.nn.sigmoid(gamma_1))
        alpha_1 = jnp.sqrt(jax.nn.sigmoid(-gamma_1))

        f = normalize(x)

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
        eps_t = jax.random.normal(rng1, shape=f.shape)
        z_t = alpha_t[:, None] * f + sigma_t[:, None] * eps_t

        # compute predicted noise
        eps_t_hat = model.apply(params, z_t, gamma_t, method=VDM.score)
        # compute MSE of predicted noise
        loss_diff_mse = jnp.sum(jnp.square(eps_t - eps_t_hat), axis=1)

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

    # training loop
    losses = []
    for i in tqdm.trange(num_train_steps):
        training_data = sample_data(num_training_data)
        rng, optim_state, params, loss, _metrics = train_step(rng, optim_state, params, training_data)
        losses.append(loss)

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./vmd_loss.png")
    print("Finish training...")

    """Evaluation"""
    # Plot the learned endpoints of the noise schedule
    print('gamma_0', model.apply(params, 0., method=VDM.gamma))
    print('gamma_1', model.apply(params, 1., method=VDM.gamma))

    # Generate samples
    def diffusion_step(params, idx, x, rng, num_diffusion_steps):
        """compute q(z_t | x)"""
        batch_size = x.shape[0]
        f = normalize(x)

        t = idx / num_diffusion_steps
        gamma_t = model.apply(params, t, method=VDM.gamma)
        gamma_t = jnp.repeat(gamma_t, batch_size)
        alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_t))[:, None]
        sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))[:, None]

        eps_t = jax.random.normal(rng, shape=x.shape)
        z_t = alpha_t * f + sigma_t * eps_t

        return z_t

    def denoising_step(params, idx, z_t, rng, num_diffusion_steps):
        """compute p(z_s | z_t)"""
        batch_size = z_t.shape[0]

        t = (num_diffusion_steps - idx) / num_diffusion_steps
        s = (num_diffusion_steps - idx - 1) / num_diffusion_steps

        gamma_s = model.apply(params, s, method=VDM.gamma)
        gamma_t = model.apply(params, t, method=VDM.gamma)
        gamma_s = jnp.repeat(gamma_s, batch_size)
        gamma_t = jnp.repeat(gamma_t, batch_size)

        eps_t = jax.random.normal(rng, z_t.shape)
        eps_t_hat = model.apply(params, z_t, gamma_t, method=VDM.score)
        alpha_s, alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_s)), jnp.sqrt(jax.nn.sigmoid(-gamma_t))
        alpha_s, alpha_t = alpha_s[:, None], alpha_t[:, None]
        sigma_s, sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_s)), jnp.sqrt(jax.nn.sigmoid(gamma_t))
        sigma_s, sigma_t = sigma_s[:, None], sigma_t[:, None]
        expm1 = jnp.expm1(gamma_s - gamma_t)
        expm1 = expm1[:, None]

        # (chongyi): equations (32) and (33) in the paper.
        z_s = alpha_s / alpha_t * (z_t + sigma_t * expm1 * eps_t_hat) + \
            sigma_s * jnp.sqrt(-expm1) * eps_t

        pred_x = (z_t - sigma_t * eps_t_hat) / alpha_t

        return z_s, pred_x

    def sample_fn(params, x, rng, num_diffusion_steps=200):
        # sample z_0 from the diffusion model
        rng, rng1 = jax.random.split(rng)
        diffusion_zs = []
        denoising_zs = [jax.random.normal(rng1, (x.shape[0], 2))]
        pred_xs = []

        for idx in tqdm.trange(num_diffusion_steps):
            rng, rng1, rng2 = jax.random.split(rng, num=3)

            diffusion_z = diffusion_step(params, idx, x, rng1, num_diffusion_steps)
            denoising_z, pred_x = denoising_step(params, idx, denoising_zs[-1], rng2, num_diffusion_steps)

            diffusion_zs.append(diffusion_z)
            denoising_zs.append(denoising_z)
            pred_xs.append(pred_x)

        rng, rng1 = jax.random.split(rng)
        diffusion_z = diffusion_step(params, num_diffusion_steps, x, rng1, num_diffusion_steps)
        diffusion_zs.append(diffusion_z)
        _, sampled_x = decode(params, denoising_zs[-1], rng)

        diffusion_zs = jnp.asarray(diffusion_zs)
        denoising_zs = jnp.asarray(denoising_zs)
        pred_xs = jnp.asarray(pred_xs)

        return diffusion_zs, denoising_zs, pred_xs, sampled_x

    rng, rng1 = jax.random.split(rng)
    diffusion_zs, denoising_zs, _, sampled_x = sample_fn(params, eval_data, rng1)

    # Create square plot:
    def plot_generative_process(diffusion_zs, denoising_zs, x, sampled_x, num_timesteps=11, filename='./vmd_zs.png'):
        fig, axes = plt.subplots(2, num_timesteps + 1)
        fig.set_figheight(3 * 2)
        fig.set_figwidth(3 * (num_timesteps + 1))

        num_diffusion_steps = diffusion_zs.shape[0] - 1
        axis_lim = 10
        alpha = 0.5

        ax = axes[0, 0]
        ax.scatter(x[:, 0], x[:, 1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-axis_lim, 0, axis_lim])
        ax.set_yticks([-axis_lim, 0, axis_lim])
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_title("$x$")

        ax = axes[1, 0]
        ax.scatter(sampled_x[:, 0], sampled_x[:, 1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-axis_lim, 0, axis_lim])
        ax.set_yticks([-axis_lim, 0, axis_lim])
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_title(r"$\hat{x}$")

        axis_lim = 3
        for timestep_idx, timestep in enumerate(np.linspace(0, 1, 11)):
            t = int(timestep * num_diffusion_steps)

            ax = axes[0, timestep_idx + 1]
            ax.scatter(diffusion_zs[t, :, 0], diffusion_zs[t, :, 1], alpha=alpha)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-axis_lim, 0, axis_lim])
            ax.set_yticks([-axis_lim, 0, axis_lim])
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_title(r"$q(z_t \mid x), t = {:0.1f}$".format(timestep))

            ax = axes[1, timestep_idx + 1]
            ax.scatter(denoising_zs[num_diffusion_steps - t, :, 0],
                       denoising_zs[num_diffusion_steps - t, :, 1], alpha=alpha)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-axis_lim, 0, axis_lim])
            ax.set_yticks([-axis_lim, 0, axis_lim])
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_title(r"$p(z_s \mid z_t), t = {:0.1f}$".format(timestep))

        fig.savefig(filename)

    plot_generative_process(diffusion_zs, denoising_zs, eval_data, sampled_x)

    # Estimate log likelihood
    def compute_ground_truth_log_probs(x):
        # mean = training_data.mean(axis=0)
        # std = training_data.std(axis=0)

        mean = np.ones(2) * 2
        std = 2
        log_probs = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2 * jnp.log(std) + (x - mean) ** 2 / std ** 2, axis=-1)

        return log_probs

    def compute_elbo(params, x, rng, num_diffusion_steps=0):
        batch_size = x.shape[0]

        # 1. RECONSTRUCTION LOSS
        rng, rng1 = jax.random.split(rng)
        z_0 = encode(params, x, rng1)
        log_probs = decode_log_probs(params, z_0, x)
        loss_recon = -log_probs

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        gamma_1 = model.apply(params, 1.0, method=VDM.gamma)
        sigma_1 = jnp.sqrt(jax.nn.sigmoid(gamma_1))
        alpha_1 = jnp.sqrt(jax.nn.sigmoid(-gamma_1))

        f = normalize(x)

        loss_klz = 0.5 * jnp.sum((alpha_1 ** 2) * (f ** 2) + sigma_1 ** 2 - 2 * jnp.log(sigma_1) - 1., axis=1)

        # 3. DIFFUSION LOSS
        # sample time steps
        rng, rng1 = jax.random.split(rng)
        t = jax.random.uniform(rng1, shape=(batch_size,))

        # discretize time steps if we're working with discrete time
        if num_diffusion_steps > 0:
            t = jnp.ceil(t * num_diffusion_steps) / num_diffusion_steps

        # sample z_t
        gamma_t = model.apply(params, t, method=VDM.gamma)
        sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))
        alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_t))

        rng, rng1 = jax.random.split(rng)
        eps_t = jax.random.normal(rng1, shape=f.shape)
        z_t = alpha_t[:, None] * f + sigma_t[:, None] * eps_t

        # compute predicted noise
        eps_t_hat = model.apply(params, z_t, gamma_t, method=VDM.score)
        # compute MSE of predicted noise
        loss_diff_mse = jnp.sum(jnp.square(eps_t - eps_t_hat), axis=1)

        if num_diffusion_steps == 0:
            # loss for infinite depth T, i.e. continuous time
            gamma = lambda t: model.apply(params, t, method=VDM.gamma)

            _, g_t_grad = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
            loss_diff = 0.5 * g_t_grad * loss_diff_mse
        else:
            # loss for finite depth T, i.e. discrete time
            s = t - (1. / num_diffusion_steps)
            gamma_s = model.apply(params, s, method=VDM.gamma)
            loss_diff = 0.5 * num_diffusion_steps * jnp.expm1(gamma_t - gamma_s) * loss_diff_mse

        # loss_recon = jnp.mean(loss_recon)
        # loss_latent = jnp.mean(loss_klz)
        # loss_diff = jnp.mean(loss_diff)
        elbo = -(loss_recon + loss_klz + loss_diff)

        return elbo

    gt_log_probs = compute_ground_truth_log_probs(eval_data)
    rng, rng1 = jax.random.split(rng)
    elbo = compute_elbo(params, eval_data, rng)

    mean_abs_error = jnp.mean(jnp.abs(gt_log_probs - elbo))

    # from scipy.stats import spearmanr
    # spearmanr(jax.nn.softmax(gt_log_probs), jax.nn.softmax(elbo))

    print(mean_abs_error)


if __name__ == "__main__":
    main()
