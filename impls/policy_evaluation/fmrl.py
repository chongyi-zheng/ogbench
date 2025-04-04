from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCFMVectorField, GCFMValue
from utils.flow_matching_utils import cond_prob_path_class, scheduler_class


class FMRLEstimator(flax.struct.PyTreeNode):
    """Flow Matching RL (FMRL) estimator."""

    rng: Any
    network: Any
    cond_prob_path: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        rng, guidance_rng, time_rng, path_rng, likelihood_rng = jax.random.split(rng, 5)
        use_guidance = (jax.random.uniform(guidance_rng) >= self.config['uncond_prob'])

        observations = jax.lax.select(
            use_guidance,
            batch['observations'],
            jnp.zeros_like(batch['observations'])
        )
        if module_name == 'critic':
            actions = jax.lax.select(
                use_guidance,
                batch['actions'],
                jnp.zeros_like(batch['actions'])
            )
        else:
            actions = None
        goals = batch['value_goals']

        times = jax.random.uniform(time_rng, shape=(batch_size, ))
        noises = jax.random.normal(path_rng, shape=goals.shape)
        path_sample = self.cond_prob_path(x_0=noises, x_1=goals, t=times)
        vf_pred = self.network.select(module_name + '_vf')(
            path_sample.x_t,
            times,
            observations,
            actions=actions,
            params=grad_params,
        )
        cfm_loss = jnp.pow(vf_pred - path_sample.dx_t[None], 2).mean()

        # use a fixed noise to estimate divergence for each ODE solving step.
        # likelihood_noises = jax.random.normal(likelihood_rng, shape=goals.shape)
        # likelihood_noises = jax.random.rademacher(
        #     likelihood_rng, shape=goals.shape, dtype=goals.dtype)
        q = self.compute_log_likelihood(
            goals, observations, likelihood_rng, actions=actions)

        if self.config['distill_likelihood']:
            q_pred = self.network.select(module_name)(
                goals, observations, actions=actions, params=grad_params)
            q_pred = q_pred.min(axis=0)

            distill_loss = jnp.mean((q - q_pred) ** 2)
        else:
            distill_loss = 0.0
        critic_loss = cfm_loss + self.config['distill_coeff'] * distill_loss

        return critic_loss, {
            'cond_flow_matching_loss': cfm_loss,
            'distillation_loss': distill_loss,
            'v_mean': q.mean(),
            'v_max': q.max(),
            'v_min': q.min(),
        }

    def compute_log_likelihood(self, goals, observations, rng, actions=None):
        if actions is not None:
            module_name = 'critic'
            num_ensembles = self.config['num_ensembles_q']
        else:
            module_name = 'value'
            num_ensembles = 1

        noisy_goals = goals
        div_int = jnp.zeros(goals.shape[:-1])
        num_flow_steps = self.config['num_flow_steps']
        step_size = 1.0 / num_flow_steps

        # Define the body function to be scanned
        def body_fn(carry, i):
            """
            carry: (noisy_goals, div_int, rng)
            i: current step index
            """
            noisy_goals, div_int, rng = carry

            # Time for this iteration
            times = 1.0 - jnp.full(noisy_goals.shape[:-1], i * step_size)

            if self.config['exact_divergence']:
                def compute_exact_div(noisy_goals, times, observations, actions):
                    def vf_func(noisy_goal, time, observation, action):
                        noisy_goal = jnp.expand_dims(noisy_goal, 0)
                        time = jnp.expand_dims(time, 0)
                        observation = jnp.expand_dims(observation, 0)
                        if action is not None:
                            action = jnp.expand_dims(action, 0)
                        vf = self.network.select(module_name + '_vf')(
                            noisy_goal, time, observation, action).squeeze(1)

                        return vf.reshape(-1)

                    def div_func(noisy_goal, time, observation, action):
                        jac = jax.jacrev(vf_func)(noisy_goal, time, observation, action)
                        jac = jac.reshape([num_ensembles, noisy_goal.shape[-1], noisy_goal.shape[-1]])

                        return jnp.trace(jac, axis1=-2, axis2=-1)

                    vf = self.network.select(module_name + '_vf')(
                        noisy_goals, times, observations, actions)

                    if actions is not None:
                        div = jax.vmap(div_func, in_axes=(0, 0, 0, 0), out_axes=1)(
                            noisy_goals, times, observations, actions)
                    else:
                        div = jax.vmap(div_func, in_axes=(0, 0, 0, None), out_axes=1)(
                            noisy_goals, times, observations, actions)

                    return vf, div

                vf, div = compute_exact_div(noisy_goals, times, observations, actions)
            else:
                def compute_hutchinson_div(noisy_goals, times, observations, actions, rng):
                    # Define vf_func for jvp
                    def vf_func(goals):
                        vf = self.network.select(module_name + '_vf')(
                            goals,
                            times,
                            observations,
                            actions=actions,
                        )

                        return vf.reshape([-1, *vf.shape[2:]])

                    # Split RNG and sample noise
                    z = jax.random.normal(rng, shape=noisy_goals.shape, dtype=noisy_goals.dtype)

                    # Forward (vf) and linearization (jac_vf_dot_z)
                    vf, jac_vf_dot_z = jax.jvp(vf_func, (noisy_goals,), (z, ))
                    vf = vf.reshape([num_ensembles, -1, *vf.shape[1:]])
                    jac_vf_dot_z = jac_vf_dot_z.reshape([num_ensembles, -1, *jac_vf_dot_z.shape[1:]])

                    # Hutchinson's trace estimator
                    # shape assumptions: jac_vf_dot_z, z both (B, D) => div is shape (B,)
                    div = jnp.einsum("eij,ij->ei", jac_vf_dot_z, z)

                    return vf, div

                rng, div_rng = jax.random.split(rng)
                vf, div = compute_hutchinson_div(noisy_goals, times, observations, actions, div_rng)

            # Update goals and divergence integral. We need to consider Q ensemble here.
            new_noisy_goals = jnp.min(noisy_goals[None] - vf * step_size, axis=0)
            new_div_int = jnp.min(div_int[None] - div * step_size, axis=0)

            # Return updated carry and scan output
            return (new_noisy_goals, new_div_int, rng), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals, div_int, rng), _ = jax.lax.scan(
            body_fn, (noisy_goals, div_int, rng), jnp.arange(num_flow_steps))

        # Finally, compute log_prob using the final noisy_goals and div_int
        gaussian_log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noisy_goals ** 2, axis=-1)
        log_prob = gaussian_log_prob + div_int  # log p_1(g | s, a)

        return log_prob

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, value_rng = jax.random.split(rng)
        value_loss, value_info = self.critic_loss(batch, grad_params, value_rng, 'value')
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + value_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the estimator and return a new estimator with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def evaluate_estimation(
        self,
        batch,
        seed=None,
    ):
        observations = batch['observations']
        actions = batch['actions']
        random_observations = jnp.roll(observations, 1, axis=0)
        random_actions = jnp.roll(actions, 1, axis=0)

        # we compute the accuracy of predicting goals from the same trajectory.
        assert (self.config['value_p_trajgoal'] == 1.0) or (self.config['actor_p_trajgoal'] == 1.0)
        if self.config['value_p_trajgoal'] == 1.0:
            goals = batch['value_goals']
        else:
            goals = batch['actor_goals']

        seed, q_seed, q_random_seed = jax.random.split(seed, 3)
        q = self.compute_log_likelihood(
            goals, observations,
            q_seed, actions=actions,
        )
        q_random = self.compute_log_likelihood(
            goals, random_observations,
            q_random_seed, actions=random_actions,
        )

        seed, v_seed, v_random_seed = jax.random.split(seed, 3)
        v = self.compute_log_likelihood(
            goals, observations, v_seed)
        v_random = self.compute_log_likelihood(
            goals, random_observations, v_random_seed)

        # log p(g | s, a) > log p(g | s_rand, a_rand)
        binary_q_acc = jnp.mean(q > q_random)
        # log p(g | s) > log p(g | s_rand)
        binary_v_acc = jnp.mean(v > v_random)

        return {
            'binary_q_acc': binary_q_acc,
            'binary_v_acc': binary_v_acc,
        }

    @jax.jit
    def compute_values(
        self,
        observations,
        goals=None,
        seed=None,
    ):
        seed, v_seed = jax.random.split(seed)
        v = self.compute_log_likelihood(
            goals, observations, v_seed)

        return v


    @classmethod
    def create(
            cls,
            seed,
            ex_observations,
            ex_actions,
            config,
    ):
        """Create a new estimator.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        ex_goals = ex_observations
        ex_times = jax.random.uniform(time_rng, shape=(ex_observations.shape[0],))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['value_state'] = encoder_module()
            encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            # critic_def = GCDiscreteBilinearCritic(
            #     hidden_dims=config['value_hidden_dims'],
            #     latent_dim=config['latent_dim'],
            #     layer_norm=config['layer_norm'],
            #     ensemble=True,
            #     value_exp=True,
            #     state_encoder=encoders.get('critic_state'),
            #     goal_encoder=encoders.get('critic_goal'),
            #     action_dim=action_dim,
            # )
            raise NotImplementedError
        else:
            critic_vf_def = GCFMVectorField(
                vector_dim=ex_goals.shape[-1],
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_ensembles_q'],
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )
            critic_def = GCFMValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

        value_vf_def = GCFMVectorField(
            vector_dim=ex_goals.shape[-1],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            state_encoder=encoders.get('value_state'),
            goal_encoder=encoders.get('value_goal'),
        )
        value_def = GCFMValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            state_encoder=encoders.get('value_state'),
            goal_encoder=encoders.get('value_goal'),
        )

        network_info = dict(
            critic_vf=(critic_vf_def, (ex_goals, ex_times, ex_observations, ex_actions)),
            critic=(critic_def, (ex_goals, ex_observations, ex_actions)),
            value_vf=(value_vf_def, (ex_goals, ex_times, ex_observations)),
            value=(value_def, (ex_goals, ex_observations)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        cond_prob_path = cond_prob_path_class[config['prob_path_class']](
            scheduler=scheduler_class[config['scheduler_class']]()
        )

        return cls(rng, network=network, cond_prob_path=cond_prob_path, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Estimator hyperparameters.
            estimator_name='fmrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            num_ensembles_q=2,  # Number of ensemble for the critic.
            prob_path_class='AffineCondProbPath',  # Conditional probability path class name.
            scheduler_class='CondOTScheduler',  # Scheduler class name.
            uncond_prob=0.0,  # Probability of training the marginal velocity field vs the guided velocity field.
            num_flow_steps=20,  # Number of steps for solving ODEs using the Euler method.
            exact_divergence=False,  # Whether to compute the exact divergence or the Hutchinson's divergence estimator.
            distill_likelihood=False,  # Whether to distill the log-likelihood solutions.
            distill_coeff=1.0,  # Likelihood distillation loss coefficient.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric samling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
