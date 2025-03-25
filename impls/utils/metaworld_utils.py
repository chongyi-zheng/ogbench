import os
import numpy as np

from utils.datasets import Dataset
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


DEFAULT_DATASET_DIR = '~/.ogbench/metaworld_data'


def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32):
    """Load metaworld dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
        If `add_info` is True, the dictionary may also contain additional keys for observation information.
    """
    file = np.load(dataset_path)

    dataset = dict()
    for k in ['observations', 'actions', 'terminals', 'masks']:
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        else:
            dtype = np.float32
        dataset[k] = file[k][...].astype(dtype, copy=False)

    # Regular dataset: Generate `next_observations` by shifting `observations`.
    # Our goal is to have the following structure:
    #                       |<- traj 1 ->|  |<- traj 2 ->|  ...
    # ----------------------------------------------------------
    # 'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
    # 'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
    # 'rewards'          : [r0, r1, r2, r3, r0, r1, r2, r3, ...]
    # 'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
    # 'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]
    # 'masks'            : [ 0,  0,  1,  0,  0,  1,  0,  0, ...]
    # masks denotes whether the agent should get a Bellman backup from the next observation.
    # It is 0 only when the task is complete (and 1 otherwise).
    # In this case, the agent should set the target Q-value to 0,
    # instead of using the next observation's target Q-value.
    #
    # terminals simply denotes whether the dataset trajectory is over, regardless of task completion.

    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask]
    dataset['rewards'] = dataset['rewards'][ob_mask]
    dataset['masks'] = dataset['masks'][ob_mask]
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    return dataset


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    env_only=False,
    randomize_init_state=True,
):
    splits = dataset_name.split('-')
    env_name, dataset_name = splits[0], splits[1]

    """Make D4RL environment."""
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
    env = env_cls(render_mode='rgb_array')
    env._freeze_rand_vec = not randomize_init_state
    # set the rendering resolution
    env.width, env.height = 200, 200
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 200, 200

    if env_only:
        return env

    train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
    val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    ob_dtype = np.uint8 if 'visual' in env_name else np.float32
    action_dtype = np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
    )

    return env, train_dataset, val_dataset
