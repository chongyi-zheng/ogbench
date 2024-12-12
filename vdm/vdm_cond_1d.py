import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from flax import linen as nn


def constant_init(value, dtype='float32'):
    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


class ScoreNetwork(nn.Module):
    hidden_units: int = 512
    init_gamma_0: float = -13.3  # initial gamma_0
    init_gamma_1: float = 5.  # initial gamma_1

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_units)
        self.dense2 = nn.Dense(self.hidden_units)
        self.dense3 = nn.Dense(2)
        self.ff = Base2FourierFeatures()

    def __call__(self, x, z, gamma_t):
        onehot_x = jax.nn.one_hot(x, num_classes=2, axis=-1)

        # Normalize gamma_t
        lb = self.init_gamma_0
        ub = self.init_gamma_1
        gamma_t_norm = ((gamma_t - lb) / (ub - lb)) * 2 - 1  # ---> [-1,+1] (not exactly in this range)

        # Concatenate normalized gamma_t as extra feature
        h = jnp.concatenate([onehot_x, z, gamma_t_norm[:, None]], axis=-1)

        # append Fourier features
        h_ff = self.ff(h)
        h = jnp.concatenate([h, h_ff], axis=-1)

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

class Encoder(nn.Module):
    hidden_units: int = 512

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_units)
        self.dense2 = nn.Dense(self.hidden_units)
        self.dense3 = nn.Dense(2)

    def __call__(self, x, y):
        onehot_x = jax.nn.one_hot(x, num_classes=2, axis=-1)
        onehot_y = jax.nn.one_hot(y, num_classes=2, axis=-1)
        h = jnp.concatenate([onehot_x, onehot_y], axis=-1)

        h = nn.swish(self.dense1(h))
        h = nn.swish(self.dense2(h))
        z = self.dense3(h)

        return z

class Decoder(nn.Module):
    hidden_units: int = 512

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_units)
        self.dense2 = nn.Dense(self.hidden_units)
        self.dense3 = nn.Dense(2)

    def __call__(self, x, z):
        onehot_x = jax.nn.one_hot(x, num_classes=2, axis=-1)
        h = jnp.concatenate([onehot_x, z], axis=-1)

        h = nn.swish(self.dense1(h))
        h = nn.swish(self.dense2(h))
        logits = self.dense3(h)

        dist = distrax.Categorical(logits=logits)

        return dist

class VDM(nn.Module):

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.score_net = ScoreNetwork()
        self.noise_schedule = NoiseSchedule()

    def __call__(self, x, z, t):
        gamma_t = self.noise_schedule(t)
        return self.score_net(x, z, gamma_t)

    def encode(self, x, y):
        return self.encoder(x, y)

    def decode(self, x, z):
        return self.decoder(x, z)

    def score(self, x, z, t):
        return self.score_net(x, z, t)

    def gamma(self, t):
        return self.noise_schedule(t)


def main():
    """Define hyper-parameters"""
    # Data hyper-parameters
    num_training_data = 100_000
    num_eval_data = 10_000
    batch_size = 10240

    # Optimization hyper-parameters
    learning_rate = 3e-4
    num_train_steps = 30_000

    rng = jax.random.PRNGKey(seed=np.random.randint(0, 2 ** 32))
    np.random.seed(np.random.randint(0, 2 ** 32))

    marginal_prob_x = np.array([0.4, 0.6])  # p(x)
    cond_prob = np.array([
        [0.2, 0.8],
        [0.7, 0.3]
    ])  # p(y | x)
    joint_prob = cond_prob * marginal_prob_x[:, None]  # p(x, y)
    marginal_prob_y = np.sum(joint_prob, axis=0)  # p(y)

    """Create datasets"""
    def create_dataset(num_data):
        dataset = dict()
        x = np.random.choice(2, size=(num_data, ), p=marginal_prob_x)
        neg_y = np.random.choice(2, size=(num_data, ), p=marginal_prob_y)
        prob = cond_prob[x]

        c = prob.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        pos_y = (u < c).argmax(axis=1)

        dataset['x'] = x
        dataset['pos_y'] = pos_y
        dataset['neg_y'] = neg_y

        return dataset

    training_data = create_dataset(num_training_data)
    eval_data = create_dataset(num_eval_data)

    emp_marginal_prob_x = np.array([
        np.sum(training_data['x'] == 0) / num_training_data,
        np.sum(training_data['x'] == 1) / num_training_data,
    ])

    emp_cond_prob = jnp.array([
        [np.sum(np.logical_and(training_data['x'] == 0, training_data['pos_y'] == 0)) / np.sum(training_data['x'] == 0),
         np.sum(np.logical_and(training_data['x'] == 0, training_data['pos_y'] == 1)) / np.sum(training_data['x'] == 0)],
        [np.sum(jnp.logical_and(training_data['x'] == 1, training_data['pos_y'] == 0)) / np.sum(training_data['x'] == 1),
         np.sum(jnp.logical_and(training_data['x'] == 1, training_data['pos_y'] == 1)) / np.sum(training_data['x'] == 1)]
    ])

    emp_marginal_prob_y = jnp.array([
        np.sum(eval_data['neg_y'] == 0) / eval_data['neg_y'].shape[0],
        np.sum(eval_data['neg_y'] == 1) / eval_data['neg_y'].shape[0]
    ])

    # plot dataset
    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(3 * 1)
    fig.set_figwidth(3 * 3)

    ax = axes[0]
    labels, counts = np.unique(training_data['x'], return_counts=True)
    ax.bar(labels, counts, align='center')
    ax.set_xticks([0, 1])
    ax.set_title(r'$p(x)$')

    ax = axes[1]
    mask = training_data['x'] == 0
    labels, counts = np.unique(training_data['pos_y'][mask], return_counts=True)
    ax.bar(labels, counts, align='center')
    ax.set_xticks([0, 1])
    ax.set_title(r'$p(y \mid x = 0)$')

    ax = axes[2]
    mask = training_data['x'] == 1
    labels, counts = np.unique(training_data['pos_y'][mask], return_counts=True)
    ax.bar(labels, counts, align='center')
    ax.set_xticks([0, 1])
    ax.set_title(r'$p(y \mid x = 1)$')

    fig.tight_layout()
    fig.savefig("./vmd_cond_1d_training_data.png")

    """Training model"""
    model = VDM()
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    # init_inputs = [jnp.ones((1,), dtype=jnp.int32), jnp.ones((1, 2)), jnp.zeros((1,))]
    dummy_x, dummy_y, dummy_z, dummy_t = jnp.ones((1,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32), jnp.ones((1, 2)), jnp.zeros((1,))
    params = model.init(rng1, dummy_x, dummy_z, dummy_t)
    encoder_params = model.init(rng2, dummy_x, dummy_y, method=model.encode)
    decoder_params = model.init(rng3, dummy_x, dummy_z, method=model.decode)
    params['params'].update(encoder_params['params'])
    params['params'].update(decoder_params['params'])
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print("Number of parameters: {}".format(num_params))  # 840200

    # initialize optimizer
    optimizer = optax.chain(
        optax.scale_by_schedule(optax.cosine_decay_schedule(1, num_train_steps, 1e-5)),
        optax.adamw(learning_rate),
    )
    optim_state = optimizer.init(params)

    @jax.jit
    def encode(params, x, y, rng):
        f = model.apply(params, x, y, method=VDM.encode)

        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        sigma_0 = jnp.sqrt(jax.nn.sigmoid(gamma_0))
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        eps_0 = jax.random.normal(rng, shape=f.shape)
        z_0 = alpha_0 * f + sigma_0 * eps_0

        return z_0

    @jax.jit
    def decode(params, x, z_0, rng):
        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        z_0_rescaled = z_0 / alpha_0
        dist = model.apply(params, x, z_0_rescaled, method=VDM.decode)
        x = dist.mode()
        sampled_x = dist.sample(seed=rng)

        return x, sampled_x

    @jax.jit
    def decode_log_probs(params, x, z_0, y):
        gamma_0 = model.apply(params, 0.0, method=VDM.gamma)
        alpha_0 = jnp.sqrt(jax.nn.sigmoid(-gamma_0))

        rescaled_z_0 = z_0 / alpha_0
        dist = model.apply(params, x, rescaled_z_0, method=VDM.decode)
        log_probs = dist.log_prob(y)

        return log_probs

    # Define training step
    def loss_fn(params, batch, rng, T_train=0):
        x = batch['x']
        y = batch['pos_y']
        batch_size = x.shape[0]

        # 1. RECONSTRUCTION LOSS
        rng, rng1 = jax.random.split(rng)
        z_0 = encode(params, x, y, rng1)
        log_probs = decode_log_probs(params, x, z_0, y)
        loss_recon = -log_probs

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        gamma_1 = model.apply(params, 1.0, method=VDM.gamma)
        sigma_1 = jnp.sqrt(jax.nn.sigmoid(gamma_1))
        alpha_1 = jnp.sqrt(jax.nn.sigmoid(-gamma_1))

        f = model.apply(params, x, y, method=VDM.encode)
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
        eps_t_hat = model.apply(params, x, z_t, gamma_t, method=VDM.score)
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

        loss = jnp.mean(loss_recon + loss_klz + loss_diff)

        info = {
            "loss_recon": loss_recon,
            "loss_latent": loss_klz,
            "loss_diff": loss_diff,
        }

        return loss, info

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(rng, optim_state, params, batch):
        rng, rng1 = jax.random.split(rng)
        (loss, metrics), grads = grad_fn(params, batch, rng1)
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return rng, optim_state, params, loss, metrics

    # training loop
    losses = []
    for _ in tqdm.trange(num_train_steps):
        idxs = np.random.randint(num_train_steps, size=batch_size)
        batch = jax.tree.map(lambda x: x[idxs], training_data)

        rng, optim_state, params, loss, _ = train_step(rng, optim_state, params, batch)
        losses.append(loss)

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./vdm_cond_1d_loss.png")
    print("Finish training...")

    """Evaluation"""
    # Plot the learned endpoints of the noise schedule
    print('gamma_0', model.apply(params, 0., method=VDM.gamma))
    print('gamma_1', model.apply(params, 1., method=VDM.gamma))

    # Generate samples
    @jax.jit
    def diffusion_step(params, idx, x, y, rng, num_diffusion_steps):
        """compute q(z_t | x)"""
        batch_size = x.shape[0]
        f = model.apply(params, x, y, method=VDM.encode)

        t = idx / num_diffusion_steps
        gamma_t = model.apply(params, t, method=VDM.gamma)
        gamma_t = jnp.repeat(gamma_t, batch_size)
        alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_t))[:, None]
        sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))[:, None]

        eps_t = jax.random.normal(rng, shape=f.shape)
        z_t = alpha_t * f + sigma_t * eps_t

        return z_t

    @jax.jit
    def denoising_step(params, idx, x, z_t, rng, num_diffusion_steps):
        """compute p(z_s | z_t)"""
        batch_size = z_t.shape[0]

        t = (num_diffusion_steps - idx) / num_diffusion_steps
        s = (num_diffusion_steps - idx - 1) / num_diffusion_steps

        gamma_s = model.apply(params, s, method=VDM.gamma)
        gamma_t = model.apply(params, t, method=VDM.gamma)
        gamma_s = jnp.repeat(gamma_s, batch_size)
        gamma_t = jnp.repeat(gamma_t, batch_size)

        eps_t = jax.random.normal(rng, z_t.shape)
        eps_t_hat = model.apply(params, x, z_t, gamma_t, method=VDM.score)
        alpha_s, alpha_t = jnp.sqrt(jax.nn.sigmoid(-gamma_s)), jnp.sqrt(jax.nn.sigmoid(-gamma_t))
        alpha_s, alpha_t = alpha_s[:, None], alpha_t[:, None]
        sigma_s, sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_s)), jnp.sqrt(jax.nn.sigmoid(gamma_t))
        sigma_s, sigma_t = sigma_s[:, None], sigma_t[:, None]
        expm1 = jnp.expm1(gamma_s - gamma_t)
        expm1 = expm1[:, None]

        # (chongyi): equations (32) and (33) in the paper.
        z_s = alpha_s / alpha_t * (z_t + sigma_t * expm1 * eps_t_hat) + \
            sigma_s * jnp.sqrt(-expm1) * eps_t

        return z_s

    def sample_fn(params, data, rng, num_diffusion_steps=200):
        x = data['x']
        y = data['pos_y']

        # sample z_0 from the diffusion model
        rng, rng1 = jax.random.split(rng)
        denoising_z = jax.random.normal(rng1, (x.shape[0], 2))
        diffusion_zs = []
        denoising_zs = [denoising_z]

        for idx in tqdm.trange(num_diffusion_steps):
            rng, rng1, rng2 = jax.random.split(rng, num=3)

            diffusion_z = diffusion_step(params, idx, x, y, rng1, num_diffusion_steps)
            denoising_z = denoising_step(params, idx, x, denoising_z, rng2, num_diffusion_steps)

            diffusion_zs.append(diffusion_z)
            denoising_zs.append(denoising_z)

        rng, rng1, rng2 = jax.random.split(rng, num=3)
        diffusion_z = diffusion_step(params, num_diffusion_steps, x, y, rng1, num_diffusion_steps)
        diffusion_zs.append(diffusion_z)
        _, sampled_y = decode(params, x, denoising_z, rng2)

        diffusion_zs = jnp.asarray(diffusion_zs)
        denoising_zs = jnp.asarray(denoising_zs)

        sampled_data = {
            'x': x,
            'y': np.asarray(sampled_y)
        }

        return diffusion_zs, denoising_zs, sampled_data

    rng, rng1 = jax.random.split(rng)
    diffusion_zs, denoising_zs, sampled_data = sample_fn(params, eval_data, rng1)

    # Create square plot:
    def plot_generative_process(diffusion_zs, denoising_zs, data, sampled_data, num_timesteps=11, filename='./vdm_cond_1d_zs.png'):
        x = data['x']
        y = data['pos_y']
        assert np.all(x == sampled_data['x'])
        sampled_y = sampled_data['y']

        fig, axes = plt.subplots(2, num_timesteps + 2)
        fig.set_figheight(3 * 2)
        fig.set_figwidth(3 * (num_timesteps + 2))

        num_diffusion_steps = diffusion_zs.shape[0] - 1

        ax = axes[0, 0]
        mask = x == 0
        labels, counts = np.unique(y[mask], return_counts=True)
        ax.bar(labels, counts, align='center')
        ax.set_xticks([0, 1])
        ax.set_title(r'$p(y \mid x = 0)$')

        ax = axes[0, 1]
        mask = x == 1
        labels, counts = np.unique(y[mask], return_counts=True)
        ax.bar(labels, counts, align='center')
        ax.set_xticks([0, 1])
        ax.set_title(r'$p(y \mid x = 1)$')

        ax = axes[1, 0]
        mask = x == 0
        labels, counts = np.unique(sampled_y[mask], return_counts=True)
        ax.bar(labels, counts, align='center')
        ax.set_xticks([0, 1])
        ax.set_title(r'$\hat{p}(y \mid x = 0)$')

        ax = axes[1, 1]
        mask = x == 1
        labels, counts = np.unique(sampled_y[mask], return_counts=True)
        ax.bar(labels, counts, align='center')
        ax.set_xticks([0, 1])
        ax.set_title(r'$p(y \mid x = 0)$')

        axis_lim = 3
        alpha = 0.5
        for timestep_idx, timestep in enumerate(np.linspace(0, 1, 11)):
            t = int(timestep * num_diffusion_steps)

            ax = axes[0, timestep_idx + 2]
            # ax.hist(diffusion_zs[t].squeeze(axis=-1))
            # ax.set_xlim(-axis_lim, axis_lim)
            ax.scatter(diffusion_zs[t, :, 0], diffusion_zs[t, :, 1], alpha=alpha)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-axis_lim, 0, axis_lim])
            ax.set_yticks([-axis_lim, 0, axis_lim])
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_title(r"$q(z_t \mid x, y), t = {:0.1f}$".format(timestep))

            ax = axes[1, timestep_idx + 2]
            # ax.hist(denoising_zs[num_diffusion_steps - t].squeeze(axis=-1))
            # ax.set_xlim(-axis_lim, axis_lim)
            ax.scatter(denoising_zs[num_diffusion_steps - t, :, 0], denoising_zs[num_diffusion_steps - t, :, 1], alpha=alpha)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-axis_lim, 0, axis_lim])
            ax.set_yticks([-axis_lim, 0, axis_lim])
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_title(r"$p(z_s \mid x, z_t), t = {:0.1f}$".format(timestep))

        fig.tight_layout()
        fig.savefig(filename)

    plot_generative_process(diffusion_zs, denoising_zs, eval_data, sampled_data)

    # Estimate log likelihood
    rng, rng1 = jax.random.split(rng)
    losses = loss_fn(params, eval_data, rng)[1]
    elbo = -(losses['loss_recon'] + losses['loss_latent'] + losses['loss_diff'])

    mask_00 = jnp.logical_and(eval_data['x'] == 0, eval_data['pos_y'] == 0)
    mask_01 = jnp.logical_and(eval_data['x'] == 0, eval_data['pos_y'] == 1)
    mask_10 = jnp.logical_and(eval_data['x'] == 1, eval_data['pos_y'] == 0)
    mask_11 = jnp.logical_and(eval_data['x'] == 1, eval_data['pos_y'] == 1)
    elbo = jnp.array([
        [jnp.sum(elbo[mask_00]) / jnp.sum(mask_00), jnp.sum(elbo[mask_01]) / jnp.sum(mask_01)],
        [jnp.sum(elbo[mask_10]) / jnp.sum(mask_10), jnp.sum(elbo[mask_11]) / jnp.sum(mask_11)]
    ])

    mean_abs_error = jnp.mean(jnp.abs(emp_cond_prob - jax.nn.softmax(elbo, axis=-1)))

    # mean_abs_err = 0.003398485481739044
    print("mean_abs_err = {}".format(mean_abs_error))


if __name__ == "__main__":
    main()
