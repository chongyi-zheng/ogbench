from collections import defaultdict
from tqdm import trange

import jax
import numpy as np

import simpler_env
from octo_utils.wrappers import SimplerOctoWrapper
from utils.env_utils import EpisodeMonitor
from utils.evaluation import add_to, flatten


def main():
    env_name = 'google_robot_pick_coke_can'
    window_size = 2
    num_eval_episodes = 5
    num_video_episodes = 1
    video_frame_skip = 3

    env = simpler_env.make(env_name)
    env = SimplerOctoWrapper(env, window_size=window_size, pred_action_horizon=4)
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
            observation = jax.tree_map(lambda obs: obs[None], observation)
            action = env.model.sample_actions(
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
