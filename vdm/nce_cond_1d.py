import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn


class NCE(nn.Module):
    hidden_units: int = 512

    def setup(self):
        # lookup table
        # self.critic = nn.Dense(1, use_bias=False)
        self.critic = nn.Sequential([
            nn.Dense(self.hidden_units),
            nn.swish,
            nn.Dense(self.hidden_units),
            nn.swish,
            nn.Dense(self.hidden_units),
            nn.swish,
            nn.Dense(self.hidden_units),
            nn.swish,
            nn.Dense(128),
            nn.swish,
            nn.Dense(1)
        ])

    def __call__(self, x, y):
        onehot_x = jax.nn.one_hot(x, num_classes=2)
        onehot_y = jax.nn.one_hot(y, num_classes=2)
        logits = self.critic(jnp.concatenate([onehot_x, onehot_y], axis=-1))
        logits = logits.squeeze(axis=-1)

        return logits


def main():
    """Define hyper-parameters"""
    # Data hyper-parameters
    num_training_data = 100_000
    num_eval_data = 10_000
    batch_size = 10240

    # Optimization hyper-parameters
    learning_rate = 3e-4
    num_train_steps = 30_000  # nr of training steps

    rng = jax.random.PRNGKey(seed=np.random.randint(0, 2 ** 32))
    np.random.seed(np.random.randint(0, 2 ** 32))

    marginal_prob_x = np.array([0.4, 0.6])  # p(x)
    cond_prob = np.array([
        [0.2, 0.8],
        [0.7, 0.3]
    ])  # p(y | x)
    joint_prob = cond_prob * marginal_prob_x[:, None]  # p(x, y)
    # marginal_prob_y = np.sum(joint_prob, axis=0)  # p(y)
    marginal_prob_y = np.array([0.95, 0.05])

    """Create datasets"""
    def create_dataset(num_data):
        dataset = dict()
        x = np.random.choice(2, size=(num_data,), p=marginal_prob_x)
        neg_y = np.random.choice(2, size=(num_data,), p=marginal_prob_y)
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
         np.sum(np.logical_and(training_data['x'] == 0, training_data['pos_y'] == 1)) / np.sum(
             training_data['x'] == 0)],
        [np.sum(jnp.logical_and(training_data['x'] == 1, training_data['pos_y'] == 0)) / np.sum(
            training_data['x'] == 1),
         np.sum(jnp.logical_and(training_data['x'] == 1, training_data['pos_y'] == 1)) / np.sum(
             training_data['x'] == 1)]
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
    fig.savefig("./nce_cond_1d_training_data.png")

    """Training model"""
    model = NCE()
    rng, rng1 = jax.random.split(rng, 2)
    dummy_x, dummy_y = jnp.ones((1,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)
    params = model.init(rng1, dummy_x, dummy_y)
    num_params = sum(x.size for x in jax.tree.leaves(params))  # 856321
    print("Number of parameters: {}".format(num_params))

    # initialize optimizer
    optimizer = optax.chain(
        optax.scale_by_schedule(optax.cosine_decay_schedule(1, num_train_steps, 1e-5)),
        optax.adamw(learning_rate),
    )
    optim_state = optimizer.init(params)

    # Define training step
    def loss_fn(params, batch):
        x = batch['x']
        pos_y = batch['pos_y']
        neg_y = batch['neg_y']

        # binary NCE loss
        pos_logits = model.apply(params, x, pos_y)
        neg_logits = model.apply(params, x, neg_y)

        loss_nce = optax.sigmoid_binary_cross_entropy(logits=pos_logits, labels=jnp.ones_like(pos_logits)) \
                   + optax.sigmoid_binary_cross_entropy(logits=neg_logits, labels=jnp.zeros_like(neg_logits))

        loss_nce = loss_nce.mean()
        loss = loss_nce

        metrics = {
            "loss_nce": loss_nce,
        }

        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(rng, optim_state, params, batch):
        rng, rng1 = jax.random.split(rng)
        (loss, metrics), grads = grad_fn(params, batch)
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return rng, optim_state, params, loss, metrics

    # training loop
    losses = []
    for i in tqdm.trange(num_train_steps):
        idxs = np.random.randint(num_train_steps, size=batch_size)
        batch = jax.tree.map(lambda x: x[idxs], training_data)
        # batch_pos_x = training_data[0][idxs]
        # batch_neg_x = training_data[1][idxs]

        rng, optim_state, params, loss, _ = train_step(rng, optim_state, params, batch)
        losses.append(loss)

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./nce_cond_1d_loss.png")
    print("Finish training...")

    """Evaluation"""
    # Estimate log likelihood
    def compute_ratio_preditions(params, data):
        x = data['x']
        y = data['pos_y']

        logits = model.apply(params, x, y)
        ratios = jnp.exp(logits)

        return ratios

    eval_xys = dict(x=jnp.asarray([0, 0, 1, 1]), pos_y=jnp.asarray([0, 1, 0, 1]))
    ratio_preds = compute_ratio_preditions(params, eval_xys)
    ratio_preds = ratio_preds.reshape([2, 2])
    prob_preds = ratio_preds * emp_marginal_prob_y[None, :]

    mean_abs_error = jnp.mean(jnp.abs(emp_cond_prob - prob_preds))

    # marginal_prob_y = [0.95, 0.05], mean_abs_err = 0.016496554017066956
    print("mean_abs_err = {}".format(mean_abs_error))


if __name__ == "__main__":
    main()
