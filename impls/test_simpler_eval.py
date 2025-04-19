from collections import defaultdict

import gymnasium
import jax
import numpy as np
from octo.model.octo_model import OctoModel
from tqdm import trange

from octo_utils.wrappers import SimplerOctoWrapper
from utils.env_utils import EpisodeMonitor
from utils.evaluation import add_to, flatten


def main():
    env_name = 'google_robot_pick_coke_can'
    window_size = 2
    num_eval_episodes = 5
    num_video_episodes = 1
    video_frame_skip = 3

    model_type = f"hf://rail-berkeley/octo-small"
    model = OctoModel.load_pretrained(model_type)

    env = simpler_env.make(env_name)
    env = SimplerOctoWrapper(
        env,
        example_batch=model.example_batch,
        # TODO (chongyiz): avoid hardcoding the dataset name.
        unnormalization_statistics=model.dataset_statistics['bridge']['action'],
        text_processor=model.text_processor,
        window_size=window_size,
        pred_action_horizon=4,
    )

    env = EpisodeMonitor(env)

    rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))
    trajs = []
    stats = defaultdict(list)
    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            rng, action_rng = jax.random.split(rng)
            observation = jax.tree.map(lambda obs: obs[None], observation)
            action = model.sample_actions(
                observation,
                env.task,
                rng=action_rng,
            )
            action = np.array(action[0])

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = observation['image_primary'][0, -1]
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    print()


if __name__ == '__main__':
    main()
