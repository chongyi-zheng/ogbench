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
            nn.Dense(32),
            nn.swish,
            nn.Dense(1)
        ])

    def __call__(self, x):
        onehot_x = jax.nn.one_hot(x, num_classes=2)
        logits = self.critic(onehot_x)
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

    marginal_prob_x = np.array([0.2, 0.8])  # p(x)
    noise_prob_x = np.array([0.9, 0.1])

    """Create datasets"""
    def sample_data(num_data):
        # 1d data
        pos_data = np.random.choice(2, size=(num_data, ), p=marginal_prob_x)
        neg_data = np.random.choice(2, size=(num_data, ), p=noise_prob_x)

        return pos_data, neg_data

    training_data = sample_data(num_training_data)
    eval_data = sample_data(num_eval_data)

    emp_marginal_prob_x = np.array([
        np.sum(training_data[0] == 0) / num_training_data,
        np.sum(training_data[0] == 1) / num_training_data,
    ])
    emp_noise_prob_x = np.array([
        np.sum(eval_data[1] == 0) / num_eval_data,
        np.sum(eval_data[1] == 1) / num_eval_data,
    ])

    # plot dataset
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    labels, counts = np.unique(training_data, return_counts=True)
    plt.bar(labels, counts, align='center')
    ax.set_xticks([0, 1])
    ax.set_title("Training data")
    fig.savefig("./nce_1d_training_data.png")

    """Training model"""
    model = NCE()
    rng, rng1, rng2 = jax.random.split(rng, 3)
    init_inputs = [jnp.ones((1,))]
    params = model.init({"params": rng1, "sample": rng2}, *init_inputs)
    num_params = sum(x.size for x in jax.tree.leaves(params))  # 280641
    print("Number of parameters: {}".format(num_params))

    # initialize optimizer
    # optimizer = optax.adamw(learning_rate)
    optimizer = optax.chain(
        optax.scale_by_schedule(optax.cosine_decay_schedule(1, num_train_steps, 1e-5)),
        optax.adamw(learning_rate),
    )
    optim_state = optimizer.init(params)

    # Define training step
    def normalize(x):
        f = (x - training_data.mean(axis=0)) / training_data.std(axis=0)
        f = f[:, None]

        return f

    def loss_fn(params, pos_x, neg_x):
        # batch_size = pox_x.shape[0]
        # pos_x = normalize(pos_x)
        # neg_x = normalize(neg_x)

        # binary NCE loss
        pos_logits = model.apply(params, pos_x)
        neg_logits = model.apply(params, neg_x)

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
    def train_step(rng, optim_state, params, pos_x, neg_x):
        rng, rng1 = jax.random.split(rng)
        (loss, metrics), grads = grad_fn(params, pos_x, neg_x)
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return rng, optim_state, params, loss, metrics

    # training loop
    losses = []
    neg_xs = []
    for i in tqdm.trange(num_train_steps):
        idxs = np.random.randint(num_train_steps, size=batch_size)
        batch_pos_x = training_data[0][idxs]
        batch_neg_x = training_data[1][idxs]

        rng, optim_state, params, loss, _metrics = train_step(rng, optim_state, params, batch_pos_x, batch_neg_x)
        losses.append(loss)

    fig = plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    fig.savefig("./nce_1d_loss.png")
    print("Finish training...")

    """Evaluation"""
    # Estimate log likelihood
    def compute_ground_truth_log_probs(x):
        log_probs = jnp.log(emp_marginal_prob_x[x])

        return log_probs

    def compute_ratio_preditions(params, x):
        logits = model.apply(params, x)
        ratios = jnp.exp(logits)

        return ratios

    eval_x = jnp.arange(2)
    # gt_log_probs = compute_ground_truth_log_probs(eval_x)
    ratio_preds = compute_ratio_preditions(params, eval_x)
    prob_preds = ratio_preds * emp_noise_prob_x
    # prob_preds /= np.sum(prob_preds, keepdims=True)

    mean_abs_error = jnp.mean(jnp.abs(emp_marginal_prob_x - prob_preds))

    print(mean_abs_error)


if __name__ == "__main__":
    main()
