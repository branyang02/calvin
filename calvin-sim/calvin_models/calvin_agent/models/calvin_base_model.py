import numpy as np
from typing import Any, Dict, Tuple


class CalvinBaseModel:
    """
    Base class for all models that can be evaluated on the CALVIN challenge.
    If you want to evaluate your own model, implement the class methods.
    """

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        raise NotImplementedError

    def step(
        self, obs: Dict[str, Any], goal: str
    ) -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
        """
        Compute the relative action based on the current observation and goal.

        Args:
            obs (dict): The current observation of the environment. A dictionary containing:
                - rgb_obs (dict): A dictionary with RGB image observations:
                    - rgb_static (np.ndarray): Static camera RGB image with shape (200, 200, 3).
                    - rgb_gripper (np.ndarray): Gripper camera RGB image with shape (84, 84, 3).
                    - rgb_tactile (np.ndarray): Tactile sensor RGB image with shape (160, 120, 6).
                - depth_obs (dict): A dictionary with depth map observations:
                    - depth_static (np.ndarray): Static camera depth map with shape (200, 200).
                    - depth_gripper (np.ndarray): Gripper camera depth map with shape (84, 84).
                    - depth_tactile (np.ndarray): Tactile sensor depth map with shape (160, 120, 2).
                - robot_obs (np.ndarray): A NumPy array representing the robot's proprioceptive state with shape (15,):
                    - tcp position (3): x,y,z in world coordinates
                    - tcp orientation (3): euler angles x,y,z in world coordinates
                    - gripper opening width (1): in meter
                    - arm_joint_states (7): in rad
                    - gripper_action (1): binary (close = -1, open = 1)
                - scene_obs (np.ndarray): A NumPy array representing the observed scene information with shape (24,):
                    - sliding door (1): joint state
                    - drawer (1): joint state
                    - button (1): joint state
                    - switch (1): joint state
                    - lightbulb (1): on=1, off=0
                    - green light (1): on=1, off=0
                    - red block (6): (x, y, z, euler_x, euler_y, euler_z)
                    - blue block (6): (x, y, z, euler_x, euler_y, euler_z)
                    - pink block (6): (x, y, z, euler_x, euler_y, euler_z)                

            goal (str): The goal of the current task in the form of a string, e.g. "pick up the red block".

        Returns:
            Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
                A 7-tuple containing the following relative actions, normalized and clipped to the range (-1, 1):

                - x (np.float32): TCP (Tool Center Point) position in the x-direction, normalized and clipped with a scaling factor of 50.
                - y (np.float32): TCP position in the y-direction, normalized and clipped with a scaling factor of 50.
                - z (np.float32): TCP position in the z-direction, normalized and clipped with a scaling factor of 50.
                - euler_x (np.float32): TCP orientation in Euler angle x (roll), normalized and clipped with a scaling factor of 20.
                - euler_y (np.float32): TCP orientation in Euler angle y (pitch), normalized and clipped with a scaling factor of 20.
                - euler_z (np.float32): TCP orientation in Euler angle z (yaw), normalized and clipped with a scaling factor of 20.
                - gripper (np.float32): Gripper action, a binary value where -1 indicates closing the gripper and 1 indicates opening it.
        """
        raise NotImplementedError
