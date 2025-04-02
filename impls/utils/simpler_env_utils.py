import abc
import dataclasses
import os
from collections import defaultdict
from typing import Any, Union
from typing import Dict, Optional

import numpy as np
import reverb
import simpler_env
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import tree
from rlds import rlds_types
from rlds import transformations

DEFAULT_DATASET_DIR = '~/tensorflow_datasets/fractal20220817_data/0.1.0'


"""
The following dataset construction code is adapted from: 
    https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

def _features_to_tensor_spec(
        feature: tfds.features.FeatureConnector
) -> tf.TensorSpec:
    """Converts a tfds Feature into a TensorSpec."""

    def _get_feature_spec(nested_feature: tfds.features.FeatureConnector):
        if isinstance(nested_feature, tf.DType):
            return tf.TensorSpec(shape=(), dtype=nested_feature)
        else:
            return nested_feature.get_tensor_spec()

    # FeaturesDict can sometimes be a plain dictionary, so we use tf.nest to
    # make sure we deal with the nested structure.
    return tf.nest.map_structure(_get_feature_spec, feature)


def _encoded_feature(feature: Optional[tfds.features.FeatureConnector],
                     image_encoding: Optional[str],
                     tensor_encoding: Optional[tfds.features.Encoding]):
    """Adds encoding to Images and/or Tensors."""

    def _apply_encoding(feature: tfds.features.FeatureConnector,
                        image_encoding: Optional[str],
                        tensor_encoding: Optional[tfds.features.Encoding]):
        if image_encoding and isinstance(feature, tfds.features.Image):
            return tfds.features.Image(
                shape=feature.shape,
                dtype=feature.dtype,
                use_colormap=feature.use_colormap,
                encoding_format=image_encoding)
        if tensor_encoding and isinstance(
                feature, tfds.features.Tensor) and feature.dtype != tf.string:
            return tfds.features.Tensor(
                shape=feature.shape, dtype=feature.dtype, encoding=tensor_encoding)
        return feature

    if not feature:
        return None
    return tf.nest.map_structure(
        lambda x: _apply_encoding(x, image_encoding, tensor_encoding), feature)


@dataclasses.dataclass
class RLDSSpec(metaclass=abc.ABCMeta):
    """Specification of an RLDS Dataset.

    It is used to hold a spec that can be converted into a TFDS DatasetInfo or
    a `tf.data.Dataset` spec.
    """
    observation_info: Optional[tfds.features.FeatureConnector] = None
    action_info: Optional[tfds.features.FeatureConnector] = None
    reward_info: Optional[tfds.features.FeatureConnector] = None
    discount_info: Optional[tfds.features.FeatureConnector] = None
    step_metadata_info: Optional[tfds.features.FeaturesDict] = None
    episode_metadata_info: Optional[tfds.features.FeaturesDict] = None

    def step_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        step = {}
        if self.observation_info:
            step[rlds_types.OBSERVATION] = _features_to_tensor_spec(
                self.observation_info)
        if self.action_info:
            step[rlds_types.ACTION] = _features_to_tensor_spec(
                self.action_info)
        if self.discount_info:
            step[rlds_types.DISCOUNT] = _features_to_tensor_spec(
                self.discount_info)
        if self.reward_info:
            step[rlds_types.REWARD] = _features_to_tensor_spec(
                self.reward_info)
        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step[k] = _features_to_tensor_spec(v)

        step[rlds_types.IS_FIRST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_LAST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_TERMINAL] = tf.TensorSpec(shape=(), dtype=bool)
        return step

    def episode_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        episode = {}
        episode[rlds_types.STEPS] = tf.data.DatasetSpec(
            element_spec=self.step_tensor_spec())
        if self.episode_metadata_info:
            for k, v in self.episode_metadata_info.items():
                episode[k] = _features_to_tensor_spec(v)
        return episode

    def to_dataset_config(
            self,
            name: str,
            image_encoding: Optional[str] = None,
            tensor_encoding: Optional[tfds.features.Encoding] = None,
            citation: Optional[str] = None,
            homepage: Optional[str] = None,
            description: Optional[str] = None,
            overall_description: Optional[str] = None,
    ) -> tfds.rlds.rlds_base.DatasetConfig:
        """Obtains the DatasetConfig for TFDS from the Spec."""
        return tfds.rlds.rlds_base.DatasetConfig(
            name=name,
            description=description,
            overall_description=overall_description,
            homepage=homepage,
            citation=citation,
            observation_info=_encoded_feature(self.observation_info, image_encoding,
                                              tensor_encoding),
            action_info=_encoded_feature(self.action_info, image_encoding,
                                         tensor_encoding),
            reward_info=_encoded_feature(self.reward_info, image_encoding,
                                         tensor_encoding),
            discount_info=_encoded_feature(self.discount_info, image_encoding,
                                           tensor_encoding),
            step_metadata_info=_encoded_feature(self.step_metadata_info,
                                                image_encoding, tensor_encoding),
            episode_metadata_info=_encoded_feature(self.episode_metadata_info,
                                                   image_encoding, tensor_encoding))

    def to_features_dict(self):
        """Returns a TFDS FeaturesDict representing the dataset config."""
        step_config = {
            rlds_types.IS_FIRST: tf.bool,
            rlds_types.IS_LAST: tf.bool,
            rlds_types.IS_TERMINAL: tf.bool,
        }

        if self.observation_info:
            step_config[rlds_types.OBSERVATION] = self.observation_info
        if self.action_info:
            step_config[rlds_types.ACTION] = self.action_info
        if self.discount_info:
            step_config[rlds_types.DISCOUNT] = self.discount_info
        if self.reward_info:
            step_config[rlds_types.REWARD] = self.reward_info

        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step_config[k] = v

        if self.episode_metadata_info:
            return tfds.features.FeaturesDict({
                rlds_types.STEPS: tfds.features.Dataset(step_config),
                **self.episode_metadata_info,
            })
        else:
            return tfds.features.FeaturesDict({
                rlds_types.STEPS: tfds.features.Dataset(step_config),
            })


RLDS_SPEC = RLDSSpec
TENSOR_SPEC = Union[tf.TensorSpec, dict[str, tf.TensorSpec]]


@dataclasses.dataclass
class TrajectoryTransform(metaclass=abc.ABCMeta):
    """Specification the TrajectoryTransform applied to a dataset of episodes.

    A TrajectoryTransform is a set of rules transforming a dataset
    of RLDS episodes to a dataset of trajectories.
    This involves three distinct stages:
    - An optional `episode_to_steps_map_fn(episode)` is called at the episode
      level, and can be used to select or modify steps.
      - Augmentation: an `episode_key` could be propagated to `steps` for
        debugging.
      - Selection: Particular steps can be selected.
      - Stripping: Features can be removed from steps. Prefer using `step_map_fn`.
    - An optional `step_map_fn` is called at the flattened steps dataset for each
      step, and can be used to featurize a step, e.g. add/remove features, or
      augument images
    - A `pattern` leverages DM patterns to set a rule of slicing an episode to a
      dataset of overlapping trajectories.

    Importantly, each TrajectoryTransform must define a `expected_tensor_spec`
    which specifies a nested TensorSpec of the resulting dataset. This is what
    this TrajectoryTransform will produce, and can be used as an interface with
    a neural network.
    """
    episode_dataset_spec: RLDS_SPEC
    episode_to_steps_fn_dataset_spec: RLDS_SPEC
    steps_dataset_spec: Any
    pattern: reverb.structured_writer.Pattern
    episode_to_steps_map_fn: Any
    expected_tensor_spec: TENSOR_SPEC
    step_map_fn: Optional[Any] = None

    def get_for_cached_trajectory_transform(self):
        """Creates a copy of this traj transform to use with caching.

        The returned TrajectoryTransfrom copy will be initialized with the default
        version of the `episode_to_steps_map_fn`, because the effect of that
        function has already been materialized in the cached copy of the dataset.
        Returns:
          trajectory_transform: A copy of the TrajectoryTransform with overridden
            `episode_to_steps_map_fn`.
        """
        traj_copy = dataclasses.replace(self)
        traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
        traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
        return traj_copy

    def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
        """Applies this TrajectoryTransform to the dataset of episodes."""

        # Convert the dataset of episodes to the dataset of steps.
        steps_dataset = episodes_dataset.map(
            self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).flat_map(lambda x: x)

        return self._create_pattern_dataset(steps_dataset)

    def transform_steps_rlds_dataset(
            self, steps_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Applies this TrajectoryTransform to the dataset of episode steps."""

        return self._create_pattern_dataset(steps_dataset)

    def create_test_dataset(
            self,
    ) -> tf.data.Dataset:
        """Creates a test dataset of trajectories.

        It is guaranteed that the structure of this dataset will be the same as
        when flowing real data. Hence this is a useful construct for tests or
        initialization of JAX models.
        Returns:
          dataset: A test dataset made of zeros structurally identical to the
            target dataset of trajectories.
        """
        zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

        return tf.data.Dataset.from_tensors(zeros)

    def _create_pattern_dataset(
            self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Create PatternDataset from the `steps_dataset`."""
        config = create_structured_writer_config('temp', self.pattern)

        # Further transform each step if the `step_map_fn` is provided.
        if self.step_map_fn:
            steps_dataset = steps_dataset.map(self.step_map_fn)
        pattern_dataset = reverb.PatternDataset(
            input_dataset=steps_dataset,
            configs=[config],
            respect_episode_boundaries=True,
            is_end_of_episode=lambda x: x[rlds_types.IS_LAST])
        return pattern_dataset


class TrajectoryTransformBuilder:
    """Facilitates creation of the `TrajectoryTransform`."""

    def __init__(self,
                 dataset_spec: RLDS_SPEC,
                 episode_to_steps_map_fn=lambda e: e[rlds_types.STEPS],
                 step_map_fn=None,
                 pattern_fn=None,
                 expected_tensor_spec=None):
        self._rds_dataset_spec = dataset_spec
        self._steps_spec = None
        self._episode_to_steps_map_fn = episode_to_steps_map_fn
        self._step_map_fn = step_map_fn
        self._pattern_fn = pattern_fn
        self._expected_tensor_spec = expected_tensor_spec

    def build(self,
              validate_expected_tensor_spec: bool = True) -> TrajectoryTransform:
        """Creates `TrajectoryTransform` from a `TrajectoryTransformBuilder`."""

        if validate_expected_tensor_spec and self._expected_tensor_spec is None:
            raise ValueError('`expected_tensor_spec` must be set.')

        episode_ds = zero_episode_dataset_from_spec(self._rds_dataset_spec)

        steps_ds = episode_ds.flat_map(self._episode_to_steps_map_fn)

        episode_to_steps_fn_dataset_spec = self._rds_dataset_spec

        if self._step_map_fn is not None:
            steps_ds = steps_ds.map(self._step_map_fn)

        zeros_spec = transformations.zeros_from_spec(steps_ds.element_spec)  # pytype: disable=wrong-arg-types

        ref_step = reverb.structured_writer.create_reference_step(zeros_spec)

        pattern = self._pattern_fn(ref_step)

        steps_ds_spec = steps_ds.element_spec

        target_tensor_structure = create_reverb_table_signature(
            'temp_table', steps_ds_spec, pattern)

        if (validate_expected_tensor_spec and
                self._expected_tensor_spec != target_tensor_structure):
            raise RuntimeError(
                'The tensor spec of the TrajectoryTransform doesn\'t '
                'match the expected spec.\n'
                'Expected:\n%s\nActual:\n%s\n' %
                (str(self._expected_tensor_spec).replace('TensorSpec',
                                                         'tf.TensorSpec'),
                 str(target_tensor_structure).replace('TensorSpec', 'tf.TensorSpec')))

        return TrajectoryTransform(
            episode_dataset_spec=self._rds_dataset_spec,
            episode_to_steps_fn_dataset_spec=episode_to_steps_fn_dataset_spec,
            steps_dataset_spec=steps_ds_spec,
            pattern=pattern,
            episode_to_steps_map_fn=self._episode_to_steps_map_fn,
            step_map_fn=self._step_map_fn,
            expected_tensor_spec=target_tensor_structure)


def zero_episode_dataset_from_spec(rlds_spec: RLDS_SPEC):
    """Creates a zero valued dataset of episodes for the given RLDS Spec."""

    def add_steps(episode, step_spec):
        episode[rlds_types.STEPS] = transformations.zero_dataset_like(
            tf.data.DatasetSpec(step_spec))
        if 'fake' in episode:
            del episode['fake']
        return episode

    episode_without_steps_spec = {
        k: v
        for k, v in rlds_spec.episode_tensor_spec().items()
        if k != rlds_types.STEPS
    }

    if episode_without_steps_spec:
        episodes_dataset = transformations.zero_dataset_like(
            tf.data.DatasetSpec(episode_without_steps_spec))
    else:
        episodes_dataset = tf.data.Dataset.from_tensors({'fake': ''})

    episodes_dataset_with_steps = episodes_dataset.map(
        lambda episode: add_steps(episode, rlds_spec.step_tensor_spec()))
    return episodes_dataset_with_steps


def create_reverb_table_signature(table_name: str, steps_dataset_spec,
                                  pattern: reverb.structured_writer.Pattern) -> reverb.reverb_types.SpecNest:
    config = create_structured_writer_config(table_name, pattern)
    reverb_table_spec = reverb.structured_writer.infer_signature(
        [config], steps_dataset_spec)
    return reverb_table_spec


def create_structured_writer_config(table_name: str,
                                    pattern: reverb.structured_writer.Pattern) -> Any:
    config = reverb.structured_writer.create_config(
        pattern=pattern, table=table_name, conditions=[])
    return config


def n_step_pattern_builder(n: int) -> Any:
    """Creates trajectory of length `n` from all fields of a `ref_step`."""

    def transform_fn(ref_step):
        traj = {}
        for key in ref_step:
            if isinstance(ref_step[key], dict):
                transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n:],
                                                       ref_step[key])
                traj[key] = transformed_entry
            else:
                traj[key] = ref_step[key][-n:]

        return traj

    return transform_fn


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
    for k in ['observations', 'actions', 'rewards', 'terminals', 'masks']:
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
    frame_stack=None,
    dataset_dir=DEFAULT_DATASET_DIR,
    env_only=False,
):
    dataset_dir = os.path.expanduser(dataset_dir)
    env = simpler_env.make(dataset_name)

    if env_only:
        return env

    ds_builder = tfds.builder_from_directory(builder_dir=dataset_dir)
    # train_ds, val_ds = ds_builder.as_dataset(
    #     split=['train[:90%]', 'train[90%:]'])  # 78491 and 8721 episodes
    train_ds, val_ds = ds_builder.as_dataset(
        split=['train[:10]', 'train[10:14]'])  # 10 and 4 episodes

    # The RLDSSpec for the dataset.
    rlds_spec = RLDSSpec(
        observation_info=ds_builder.info.features['steps']['observation'],
        action_info=ds_builder.info.features['steps']['action'],
        reward_info=ds_builder.info.features['steps']['reward'],
    )

    def step_map_fn(step):
        return {
            'observation': tf.cast(
                tf.image.resize(step['observation']['image'], (64, 64)),
                tf.uint8
            ),
            'action': tf.cast(tf.concat([
                step['action']['world_vector'],
                step['action']['rotation_delta'],
                step['action']['gripper_closedness_action'],
            ], axis=-1), tf.float32),
            'reward': tf.cast(step['reward'], tf.float32),
            'terminal': tf.cast(step['is_last'], tf.bool),
            'mask': tf.cast(1.0 - step['reward'], tf.bool),
        }

    def stacked_step_map_fn(step):
        return {
            'observations': step['observation'][:frame_stack],
            'next_observation': step['observation'][1:frame_stack + 1],
            'action': step['action'][frame_stack],
            'next_actions': step['action'][frame_stack + 1],
            'rewards': step['reward'][frame_stack],
            'terminals': step['terminal'][frame_stack],
            'masks': step['mask'][frame_stack],
        }

    # The following will create a trajectories of length 2.
    frame_stack = frame_stack or 1
    trajectory_transform = TrajectoryTransformBuilder(
        rlds_spec, step_map_fn=step_map_fn,
        pattern_fn=n_step_pattern_builder(frame_stack + 1)).build(
        validate_expected_tensor_spec=False)

    train_dataset = trajectory_transform.transform_episodic_rlds_dataset(
        train_ds)
    train_dataset = train_dataset.map(
        stacked_step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = trajectory_transform.transform_episodic_rlds_dataset(
        val_ds)
    val_dataset = val_dataset.map(
        stacked_step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # train_ds = train_ds.batch(len(train_ds))
    # train_ds = train_ds.unbatch()

    # def episode_map_fn(traj):
    #     episode = traj['steps']
    #
    #     return {
    #         'observation': tf.cast(
    #             tf.image.resize(episode['observation']['image'], (64, 64)),
    #             tf.uint8
    #         ),
    #         'action': tf.cast(tf.concat([
    #             episode['action']['world_vector'],
    #             episode['action']['rotation_delta'],
    #             episode['action']['gripper_closedness_action'],
    #         ], axis=-1), tf.float32),
    #         'reward': tf.cast(episode['reward'], tf.float32),
    #         'terminal': tf.cast(episode['is_last'], tf.bool),
    #         'mask': tf.cast(1.0 - episode['reward'], tf.bool),
    #     }
    #
    # # train_ds = train_ds.map(
    # #     episode_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # #
    # # train_ds_iter = iter(tfds.as_numpy(train_ds))
    # # elems = defaultdict(list)
    # # for episode in tqdm.tqdm(train_ds_iter):
    # #     for k, v in episode.items():
    # #         elems[k].append(v)
    # #
    # # for k, v in elems.items():
    # #     elems[k] = np.array(v)
    # #
    # # print()
    #
    # def step_map_fn(step):
    #     return {
    #         'observation': tf.cast(
    #             tf.image.resize(step['observation']['image'], (64, 64)),
    #             tf.uint8
    #         ),
    #         'action': tf.cast(tf.concat([
    #             step['action']['world_vector'],
    #             step['action']['rotation_delta'],
    #             step['action']['gripper_closedness_action'],
    #         ], axis=-1), tf.float32),
    #         'reward': tf.cast(step['reward'], tf.float32),
    #         'terminal': tf.cast(step['is_last'], tf.bool),
    #         'mask': tf.cast(1.0 - step['reward'], tf.bool),
    #     }
    #
    # # convert RLDS episode dataset to individual steps & reformat
    # train_ds = train_ds.map(
    #     lambda episode: episode['steps'],
    #     num_parallel_calls=tf.data.AUTOTUNE
    # ).flat_map(lambda x: x)
    # train_ds = train_ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    #
    # # shuffle, repeat, pre-fetch, batch
    # # train_ds = train_ds.cache()  # optionally keep full dataset in memory
    # # train_ds = train_ds.shuffle(1024)  # set shuffle buffer size
    # # train_ds = train_ds.repeat()  # ensure that data never runs out
    # train_ds = train_ds.batch(1024)
    #
    # train_ds_iter = iter(tfds.as_numpy(train_ds))
    # elems = defaultdict(list)
    # for elem in tqdm.tqdm(train_ds_iter):
    #     for k, v in elem.items():
    #         elems[k].append(v)
    #
    # for k, v in elems.items():
    #     elems[k] = np.concatenate(v, axis=0)

    print()

    # train_ds = train_ds.prefetch(3).batch(256).as_numpy_iterator()

    # train_dataset = load_dataset(
    #     train_dataset_path,
    #     ob_dtype=ob_dtype,
    #     action_dtype=action_dtype,
    # )
    # val_dataset = load_dataset(
    #     val_dataset_path,
    #     ob_dtype=ob_dtype,
    #     action_dtype=action_dtype,
    # )

    return env, train_dataset, val_dataset
