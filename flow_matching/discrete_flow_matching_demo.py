import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from sklearn.datasets import make_moons


class DiscreteFlow(nn.Module):
    dim: int = 2
    hidden_units: int = 128
    vocab_size: int = 128

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.hidden_units)
        self.net = nn.Sequential([
            nn.Dense(self.hidden_units),
            nn.elu,
            nn.Dense(self.hidden_units),
            nn.elu,
            nn.Dense(self.hidden_units),
            nn.elu,
            nn.Dense(self.dim * self.vocab_size)
        ])

    def __call__(self, x_t, t):
        embed = jax.lax.collapse(self.embed(x_t), 1)
        h = jnp.concatenate([t[:, None], embed], axis=-1)
        out = self.net(h).reshape(list(x_t.shape) + [self.vocab_size])

        return out


def main():
    # training
    batch_size = 256
    vocab_size = 128
    learning_rate = 0.001
    num_training_steps = 10000

    key = jax.random.PRNGKey(seed=0)
    np.random.seed(0)

    model = DiscreteFlow(vocab_size=vocab_size)
    key, init_key = jax.random.split(key)
    dummy_x_t, dummy_t = jnp.ones((1, 2), dtype=jnp.int32), jnp.zeros((1,))
    params = model.init(key, dummy_x_t, dummy_t)

    optimizer = optax.adamw(learning_rate)
    optim_state = optimizer.init(params)

    def loss_fn(params, key, x_t, t, x_1):
        logits = model.apply(params, x_t, t)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=jax.lax.collapse(logits, 0, 2),
            labels=jax.lax.collapse(x_1, 0, 2)
        )
        loss = jnp.mean(loss)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, optim_state, key, x_t, t, x_1):
        key, loss_key = jax.random.split(key)
        loss, grads = grad_fn(params, loss_key, x_t, t, x_1)
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return params, optim_state, key, loss

    losses = []
    for _ in tqdm.trange(num_training_steps):
        key, noise_key, time_key, interp_key = jax.random.split(key, num=4)

        x_1 = jnp.asarray(make_moons(batch_size, noise=0.05)[0])
        x_1 = jnp.round(jnp.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).astype(jnp.int32)

        x_0 = jax.random.randint(noise_key, shape=(batch_size, 2), minval=0, maxval=vocab_size)

        t = jax.random.uniform(time_key, shape=(batch_size, ))
        x_t = jnp.where(jax.random.uniform(interp_key, shape=(batch_size, 2)) < t[:, None], x_1, x_0)

        params, optim_state, key, loss = train_step(params, optim_state, key, x_t, t, x_1)
        losses.append(loss)

    # sampling
    key, interp_key, cate_key = jax.random.split(key, num=3)
    x_t = jax.random.randint(interp_key, shape=(200, 2), minval=0, maxval=vocab_size)
    t = 0.0
    results = [(x_t, t)]
    while t < 1.0 - 1e-3:
        p1 = jax.nn.softmax(model.apply(params, x_t, jnp.ones(200) * t), axis=-1)
        h = min(0.1, 1.0 - t)
        one_hot_x_t = jax.nn.one_hot(x_t, vocab_size).astype(jnp.float32)
        u = (p1 - one_hot_x_t) / (1.0 - t)
        x_t = jax.random.categorical(cate_key, logits=jnp.log(one_hot_x_t + h * u))
        t += h
        results.append((x_t, t))

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./discrete_flow_matching_loss.png")
    print("Finish training...")

    fig, axes = plt.subplots(1, len(results), figsize=(15, 2), sharex=True, sharey=True)

    for (x_t, t), ax in zip(results, axes):
        ax.scatter(x_t[:, 0], x_t[:, 1], s=10)
        ax.set_title(f't={t:.1f}')

    plt.tight_layout()
    fig.savefig("./discrete_flow_matching_sampling.png")


if __name__ == "__main__":
    main()
