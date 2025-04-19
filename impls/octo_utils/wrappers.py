from collections import deque

import gymnasium
import jax
import numpy as np
import tensorflow as tf
import torch

from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils import common
from octo.model.octo_model import _verify_shapes
from simpler_env.utils.action.action_ensemble import ActionEnsembler


class SimplerOctoWrapper(gymnasium.Wrapper):
    """Wrapper for the simpler environments using octo models.

    Adapt from https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/policies/octo/octo_model.py.
    
    """

    def __init__(self, env,
                 example_batch,
                 unnormalization_statistics,
                 text_processor=None,
                 window_size=2,
                 pred_action_horizon=4,
                 image_size=256,
                 action_ensemble_temp=0.0,
                 ):
        super().__init__(env)

        self.example_batch = example_batch
        self.unnormalization_statistics = unnormalization_statistics
        self.text_processor = text_processor
        self.window_size = window_size
        self.pred_action_horizon = pred_action_horizon
        self.image_size = image_size
        self.action_ensemble_temp = action_ensemble_temp

        # dataset_id = "bridge_dataset" if dataset_id is None else dataset_id
        # released huggingface octo models
        # self.model_type = f"hf://rail-berkeley/octo-small"
        # self.tokenizer, self.tokenizer_kwargs = None, None
        # self.model = OctoModel.load_pretrained(self.model_type)
        # self.action_mean = self.unnormalization_statistics["mean"]
        # self.action_std = self.unnormalization_statistics["std"]
        self.action_ensembler = ActionEnsembler(
            self.pred_action_horizon, self.action_ensemble_temp)

        if "google_robot" in env.unwrapped.robot_uids.uid:
            self.sticky_gripper_num_repeat = 15
            self.camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uids.uid:
            self.sticky_gripper_num_repeat = 1
            self.camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError("Unknown robot uid: {}".format(env.unwrapped.robot_uids.uid))
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.window_size)
        self.num_image_history = 0

    def _resize_image(self, image):
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image):
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.window_size)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=bool)
        pad_mask[:horizon - min(horizon, self.num_image_history)] = False
        return images, pad_mask

    def _create_tasks(
        self, goals=None, texts=None
    ):
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: np.ones(v.shape[:1], dtype=bool) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update(
                {
                    k: np.zeros(batch_size, dtype=bool)
                    for k in tasks.keys()
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = np.ones(
                len(texts), dtype=bool
            )
        else:
            batch_size = jax.tree_leaves(goals)[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = np.zeros(
                batch_size, dtype=bool
            )

        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(
                tasks["language_instruction"]
            )
        else:
            del tasks["language_instruction"]

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks

    def _reset_model(self, task_description):
        self.task = self._create_tasks(texts=[task_description])
        self.task_description = task_description
        self.image_history.clear()
        self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _process_observation(self, observation, first_obs=False):
        image = observation["sensor_data"][self.camera_name]["rgb"]
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        if first_obs:
            self.num_image_history += 1
            self.image_history.extend([image] * self.window_size)
        else:
            self._add_image_to_history(image)
        images, timestep_pad_mask = self._obtain_image_history_and_mask()
        assert len(images) == self.window_size

        processed_observation = dict(
            image_primary=images,
            timestep_pad_mask=timestep_pad_mask,
            pad_mask_dict=dict(
                image_primary=np.ones_like(timestep_pad_mask, dtype=bool),
            )
        )

        return processed_observation

    def _process_action(self, action):
        action = (action.copy()[None] * self.unnormalization_statistics["std"][None, None] +
                  self.unnormalization_statistics["mean"][None, None])

        assert action.shape == (1, self.pred_action_horizon, 7)
        action = self.action_ensembler.ensemble_action(action).squeeze()

        # process action to obtain the processed_action to be sent to the maniskill3 environment
        processed_action = dict()
        processed_action["world_vector"] = np.asarray(action[:3], dtype=np.float64)

        rot_axangle = rotation_conversions.matrix_to_axis_angle(
            rotation_conversions.euler_angles_to_matrix(
                common.to_tensor(np.asarray(action[3:6])), "XYZ"))
        processed_action["rot_axangle"] = common.to_numpy(rot_axangle, dtype=np.float64)
        current_gripper_action = np.asarray(action[6:7], dtype=np.float64)
        if "google_robot" in self.env.unwrapped.robot_uids.uid:
            # action_rotation_delta = np.asarray(action[3:6], dtype=np.float64)
            # roll, pitch, yaw = action_rotation_delta
            # action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            # action_rotation_axangle = action_rotation_ax * action_rotation_angle
            # processed_action["rot_axangle"] = action_rotation_axangle

            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0], dtype=np.float64)
            else:
                relative_gripper_action = (
                        self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
        elif "widowx" in self.env.unwrapped.robot_uids.uid:
            relative_gripper_action = (
                2.0 * (current_gripper_action > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
        else:
            raise NotImplementedError("Unknown robot uid: {}".format(self.env.unwrapped.robot_uids.uid))

        processed_action["gripper"] = relative_gripper_action
        processed_action = np.concatenate([
            processed_action["world_vector"],
            processed_action["rot_axangle"],
            processed_action["gripper"]
        ])

        return processed_action

    def step(self, action):
        instruction = self.env.unwrapped.get_language_instruction()[0]
        if instruction != self.task_description:
            # task description has changed; reset the policy state
            self._reset_model(instruction)
            first_obs = True
        else:
            first_obs = False
        processed_action = self._process_action(action)

        observation, reward, terminated, truncated, info = self.env.step(processed_action)
        observation = jax.tree.map(
            lambda x: x[0].cpu().detach().numpy() if isinstance(x, torch.Tensor) else x,
            observation)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        info = jax.tree.map(lambda x: x.item() if isinstance(x, torch.Tensor) else x, info)
        processed_observation = self._process_observation(observation, first_obs=first_obs)

        return processed_observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        observation = jax.tree.map(
            lambda x: x[0].cpu().detach().numpy() if isinstance(x, torch.Tensor) else x,
            observation)
        info = jax.tree.map(lambda x: x.item() if isinstance(x, torch.Tensor) else x, info)
        instruction = self.env.unwrapped.get_language_instruction()[0]
        self._reset_model(instruction)
        processed_observation = self._process_observation(observation, first_obs=True)

        return processed_observation, info
