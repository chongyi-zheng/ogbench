import abc
import dataclasses
import functools
import os
from typing import Any, Union, Dict, Optional, Callable

import cv2 as cv
import gymnasium
import numpy as np
import reverb
import rlds
import simpler_env
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from rlds import rlds_types
from rlds import transformations
from transforms3d.euler import euler2axangle

StepFnMapType = Callable[[rlds_types.Step, rlds_types.Step], None]

DEFAULT_DATASET_DIR = '~/tensorflow_datasets/dataset_dir_name/0.1.0'

"""
The following dataset construction code is adapted from: 
    https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Minimal_Training_Example.ipynb
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

    # def get_for_cached_trajectory_transform(self):
    #     """Creates a copy of this traj transform to use with caching.
    #
    #     The returned TrajectoryTransfrom copy will be initialized with the default
    #     version of the `episode_to_steps_map_fn`, because the effect of that
    #     function has already been materialized in the cached copy of the dataset.
    #     Returns:
    #       trajectory_transform: A copy of the TrajectoryTransform with overridden
    #         `episode_to_steps_map_fn`.
    #     """
    #     traj_copy = dataclasses.replace(self)
    #     traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
    #     traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
    #     return traj_copy

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

    # def create_test_dataset(
    #         self,
    # ) -> tf.data.Dataset:
    #     """Creates a test dataset of trajectories.
    #
    #     It is guaranteed that the structure of this dataset will be the same as
    #     when flowing real data. Hence this is a useful construct for tests or
    #     initialization of JAX models.
    #     Returns:
    #       dataset: A test dataset made of zeros structurally identical to the
    #         target dataset of trajectories.
    #     """
    #     zeros = transformations.zeros_from_spec(self.expected_tensor_spec)
    #
    #     return tf.data.Dataset.from_tensors(zeros)

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
    """
    References: https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb.
    
    Facilitates creation of the `TrajectoryTransform`.
    """

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


# def episode_to_steps_map_fn(episode):
#     steps = episode[rlds_types.STEPS]
#     batched_steps = rlds.transformations.batch(steps, size=2, shift=1)
#
#     def batched_steps_to_transition(batch):
#         """Converts a pair of consecutive steps to a custom transition format."""
#         return {'observations': batch[rlds.OBSERVATION][0],
#                 'next_observations': batch[rlds.OBSERVATION][1],
#                 'actions': batch[rlds.ACTION][0],
#                 'next_actions': batch[rlds.ACTION][1],
#                 'rewards': batch[rlds.REWARD][0],
#                 'terminals': tf.cast(batch[rlds.IS_LAST][0], tf.float32),
#                 'masks': 1.0 - batch[rlds.REWARD][0]}
#
#     return batched_steps.map(batched_steps_to_transition)


def rescale_action_with_bound(
        actions: np.ndarray | tf.Tensor,
        low: float,
        high: float,
        safety_margin: float = 0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
) -> np.ndarray | tf.Tensor:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    if isinstance(actions, tf.Tensor):
        clip_by_value = tf.clip_by_value
    else:
        clip_by_value = np.clip

    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def pad_initial_zero_episode(episode: tf.data.Dataset, num_zero_step: int) -> tf.data.Dataset:
    zero_steps = episode[rlds.STEPS].take(1)
    zero_steps = zero_steps.map(lambda x: tf.nest.map_structure(tf.zeros_like, x),
                                num_parallel_calls=tf.data.AUTOTUNE)
    zero_steps = zero_steps.repeat(num_zero_step)

    episode[rlds.STEPS] = rlds.transformations.concatenate(zero_steps, episode[rlds.STEPS])
    return episode


def base_step_map_fn(step: rlds.Step, map_action: StepFnMapType,
                     width: int = 128, height: int = 128):
    transformed_step = {}
    transformed_step[rlds.REWARD] = tf.cast(
        step[rlds.REWARD], tf.float32)
    transformed_step[rlds.IS_FIRST] = tf.cast(
        step[rlds.IS_FIRST], tf.bool)
    transformed_step[rlds.IS_LAST] = tf.cast(
        step[rlds.IS_LAST], tf.bool)
    transformed_step[rlds.IS_TERMINAL] = tf.cast(
        step[rlds.IS_TERMINAL], tf.bool)

    # (chongyi): use resize with pad
    transformed_step[rlds.OBSERVATION] = tf.cast(
        tf.image.resize_with_pad(step[rlds.OBSERVATION]['image'],
                                 target_width=width, target_height=height),
        tf.uint8
    )
    # transformed_step[rlds.OBSERVATION] = tf.cast(
    #     tf.image.resize(step[rlds.OBSERVATION]['image'], (width, height)),
    #     tf.uint8
    # )
    transformed_step[rlds.ACTION] = {
        'world_vector': tf.zeros(3, dtype=tf.float32),
        'rotation_delta': tf.zeros(3, dtype=tf.float32),
        'gripper_closedness_action': tf.zeros(1, dtype=tf.float32),
    }

    map_action(transformed_step, step)

    transformed_step[rlds.ACTION] = tf.cast(tf.concat([
        transformed_step[rlds.ACTION][k] for k in ['world_vector', 'rotation_delta', 'gripper_closedness_action']
    ], axis=-1), tf.float32)

    return transformed_step


def google_robot_map_action(to_step: rlds.Step, from_step: rlds.Step):
    # (chongyiz): remember to put all keys from the 'from_step' into 'to_step'.
    to_step[rlds.ACTION]['world_vector'] = from_step[rlds.ACTION]['world_vector']
    to_step[rlds.ACTION]['rotation_delta'] = rescale_action_with_bound(
        from_step[rlds.ACTION]['rotation_delta'],
        low=-np.pi / 2, high=np.pi / 2,
        post_scaling_max=1.0, post_scaling_min=-1.0,
    )
    # to_step[rlds.ACTION]['rotation_delta'] = from_step[rlds.ACTION]['rotation_delta']
    to_step[rlds.ACTION]['gripper_closedness_action'] = from_step[rlds.ACTION]['gripper_closedness_action']


def bridge_map_action(to_step: rlds.Step, from_step: rlds.Step):
    to_step[rlds.ACTION]['world_vector'] = rescale_action_with_bound(
        from_step[rlds.ACTION]['world_vector'],
        low=-0.05, high=0.05,
        safety_margin=0.01,
        post_scaling_max=1.0, post_scaling_min=-1.0,
    )
    to_step[rlds.ACTION]['rotation_delta'] = rescale_action_with_bound(
        from_step[rlds.ACTION]['rotation_delta'],
        low=-0.25, high=0.25,
        safety_margin=0.01,
        post_scaling_max=1.0, post_scaling_min=-1.0,
    )

    open_gripper = from_step[rlds.ACTION]['open_gripper']
    possible_values = tf.constant([True, False], dtype=tf.bool)
    eq = tf.equal(possible_values, open_gripper)
    assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

    with tf.control_dependencies([assert_op]):
        to_step[rlds.ACTION]['gripper_closedness_action'] = tf.cond(
            # for open_gripper in bridge dataset,
            # 0 is fully closed and 1 is fully open
            open_gripper,
            # for Fractal data,
            # gripper_closedness_action = -1 means opening the gripper and
            # gripper_closedness_action = 1 means closing the gripper.
            lambda: tf.constant([-1.0], dtype=tf.float32),
            lambda: tf.constant([1.0], dtype=tf.float32),
        )
    # to_step[rlds.ACTION]['gripper_closedness_action'] = tf.cond(
    #     # for open_gripper in bridge dataset,
    #     # 0 is fully closed and 1 is fully open
    #     from_step[rlds.ACTION]['open_gripper'],
    #     # for Fractal data,
    #     # gripper_closedness_action = -1 means opening the gripper and
    #     # gripper_closedness_action = 1 means closing the gripper.
    #     lambda: tf.constant([-1.0], dtype=tf.float32),
    #     lambda: tf.constant([1.0], dtype=tf.float32),
    # )


def base_stacked_step_map_fn(step: rlds.Step, frame_stack: int = 1):
    observations = tf.transpose(step['observation'][:frame_stack], perm=[1, 2, 0, 3])
    next_observations = tf.transpose(step['observation'][1:frame_stack + 1], perm=[1, 2, 0, 3])
    observations = tf.reshape(observations, [*observations.shape[:-2], -1])
    next_observations = tf.reshape(next_observations, [*next_observations.shape[:-2], -1])

    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': step['action'][frame_stack - 1],
        'next_actions': step['action'][frame_stack],
        'rewards': step['reward'][frame_stack - 1],
        'terminals': tf.cast(step['is_last'][frame_stack - 1], tf.float32),
        'masks': 1.0 - step['reward'][frame_stack - 1],
    }


class SimplerEnvWrapper(gymnasium.Wrapper):
    """Environment wrapper for simpler environments."""

    def __init__(self, env, width=128, height=128):
        super().__init__(env)

        self.width = width
        self.height = height
        self.robot_uid = env.get_wrapper_attr('robot_uid')

        if 'google_robot' in self.robot_uid:
            self.camera_name = 'overhead_camera'
        elif 'widowx' in self.robot_uid:
            self.camera_name = '3rd_view_camera'
        else:
            raise NotImplementedError("Unknown robot_uid: {}".format(self.robot_uid))

        image_obs_space = self.observation_space['image'][self.camera_name]['rgb']
        self.observation_space = gymnasium.spaces.Box(
            low=image_obs_space.low[0, 0, 0], high=image_obs_space.high[0, 0, 0],
            shape=(self.width, self.height, 3),
            dtype=image_obs_space.dtype,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=self.action_space.shape,
            dtype=self.action_space.dtype,
        )

    def process_action(self, action):
        """
        Reference: https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/policies/rt1/rt1_model.py.
        """

        processed_action = action.copy().astype(np.float64)
        if 'google_robot' in self.robot_uid:
            processed_action[-1] = np.where(
                np.abs(processed_action[-1]) < 1e-2,
                0.0,
                processed_action[-1],
            )

            action_rotation_delta = rescale_action_with_bound(
                processed_action[3:6],
                low=-1.0, high=1.0,
                post_scaling_max=np.pi / 2, post_scaling_min=-np.pi / 2,
            )
            action_rotation_angle = np.linalg.norm(action_rotation_delta)
            action_rotation_ax = (
                action_rotation_delta / action_rotation_angle
                if action_rotation_angle > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )
            processed_action[3:6] = action_rotation_ax * action_rotation_angle
        elif 'widowx' in self.robot_uid:
            processed_action[0:3] = rescale_action_with_bound(
                processed_action[0:3],
                low=-1.0, high=1.0,
                post_scaling_max=0.05, post_scaling_min=-0.05,
            )
            processed_action[3:6] = rescale_action_with_bound(
                processed_action[3:6],
                low=-1.0, high=1.0,
                post_scaling_max=0.25, post_scaling_min=-0.25,
            )

            roll, pitch, yaw = processed_action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            processed_action[3:6] = action_rotation_ax * action_rotation_angle

            # rt1 policy output is uniformized such that -1 is open gripper, 1 is close gripper;
            # thus we need to invert the rt1 output gripper action for some embodiments like WidowX, since for these embodiments -1 is close gripper, 1 is open gripper
            processed_action[-1] = -processed_action[-1]

            # binarize gripper action to be -1 or 1
            processed_action[-1] = 2.0 * (processed_action[-1] > 0.0) - 1.0
        else:
            raise NotImplementedError("Unknown robot_uid: {}".format(self.robot_uid))

        return processed_action

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation['image'][self.camera_name]['rgb']
        observation = cv.resize(observation, (self.width, self.height))

        return observation, info

    def step(self, action):
        action = self.process_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = observation['image'][self.camera_name]['rgb']
        observation = cv.resize(observation, (self.width, self.height))

        return observation, reward, terminated, truncated, info


def make_env_and_datasets(
    dataset_name,
    frame_stack=None,
    dataset_dir=DEFAULT_DATASET_DIR,
    env_only=False,
    width=128,
    height=128,
):
    if 'google_robot' in dataset_name:
        dataset_dir_name = 'fractal20220817_data'
    elif 'widowx' in dataset_name:
        dataset_dir_name = 'bridge'
    else:
        raise NotImplementedError("Unknown dataset_name: {}".format(dataset_name))

    dataset_dir = os.path.expanduser(dataset_dir.replace('dataset_dir_name', dataset_dir_name))
    env = simpler_env.make(dataset_name)
    env = SimplerEnvWrapper(env, width=width, height=height)

    if env_only:
        return env

    ds_builder = tfds.builder_from_directory(builder_dir=dataset_dir)
    # 82851 / 4361 episodes for google_robot dataset
    # 24187 / 1273 episodes for bridge_v2 dataset
    # TODO (chongyiz): bridge_v2 already contains 'train' and 'test' splits.
    train_ds, val_ds = ds_builder.as_dataset(
        split=['train[:95%]', 'train[95%:]'])

    # We need pad_initial_zero_episode because reverb.PatternDataset will skip
    # constructing trajectories where the first trajectory_length - 1 steps are
    # the final step in a trajectory. As such, without padding, the policies will
    # not be trained to predict the actions in the first trajectory_length - 1
    # steps.
    # We are padding with num_zero_step = trajectory_length - 1 steps.
    frame_stack = frame_stack or 1
    trajectory_length = frame_stack + 1
    train_ds = train_ds.map(
        functools.partial(pad_initial_zero_episode, num_zero_step=trajectory_length - 1),
        num_parallel_calls=tf.data.AUTOTUNE)

    # The RLDSSpec for the dataset.
    rlds_spec = RLDSSpec(
        observation_info=ds_builder.info.features['steps']['observation'],
        action_info=ds_builder.info.features['steps']['action'],
        reward_info=ds_builder.info.features['steps']['reward'],
    )

    if 'google_robot' in env.get_wrapper_attr('robot_uid'):
        step_map_fn = functools.partial(base_step_map_fn,
                                        map_action=google_robot_map_action)
    elif 'widowx' in env.get_wrapper_attr('robot_uid'):
        step_map_fn = functools.partial(base_step_map_fn,
                                        map_action=bridge_map_action)
    else:
        raise NotImplementedError()

    # The following will create a trajectories of length 'trajectory_length'.
    stacked_step_map_fn = functools.partial(
        base_stacked_step_map_fn, frame_stack=frame_stack)
    trajectory_transform = TrajectoryTransformBuilder(
        rlds_spec,
        # episode_to_steps_map_fn=episode_to_steps_map_fn,
        step_map_fn=step_map_fn,
        pattern_fn=n_step_pattern_builder(trajectory_length)).build(
        validate_expected_tensor_spec=False)

    train_dataset = trajectory_transform.transform_episodic_rlds_dataset(
        train_ds)
    train_dataset = train_dataset.map(
        stacked_step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = trajectory_transform.transform_episodic_rlds_dataset(
        val_ds)
    val_dataset = val_dataset.map(
        stacked_step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return env, train_dataset, val_dataset
