import numpy as np

from gym_xarm.tasks import Base


class Push(Base):
    """Xarm Push environment where the goal is to push the cube to a target location on a table.

    The action space is the desired relative target for the end-effector (x (forward), y (left), z (up)). The gripper
    is fixed in place.

    TODO: Document the units of x, y, and z.
    """

    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 50,
        "description": "Push the cube to a target location on a table",
    }

    def __init__(self, movement_penalty_coeff: float = 0.15, **kwargs):
        """
        Args:
            movement_penalty_coeff: The reward is calculated as (distance_to_goal + c * movement_velocity_magnitude),
                where c is this parameter. Defaults to 0.15 to match the original implementation (although this is
                really meaningless as the action space units are different).
        """
        self._movement_penalty_coeff = movement_penalty_coeff
        super().__init__("push", **kwargs)

    def is_success(self):
        return np.linalg.norm(self.obj - self.target_loc) <= 0.05

    def get_reward(self):
        # dist = np.linalg.norm(self.obj - self.goal)
        dist = np.linalg.norm(self.obj - self.target_loc)
        # Add penalty based on last action magnitude. We don't want there to be a success condition if the block is
        # being pushed through the goal without stopping.
        return -(dist + self._movement_penalty_coeff * self._act_magnitude**2)

    def _get_obs(self):
        """TODO(now): What should this return?"""
        # return np.concatenate(
        #     [
        #         # self.eef,
        #         # self.eef_velp,
        #         self.obj,
        #         self.obj_rot,
        #         self.obj_velp,
        #         self.obj_velr,
        #         self.eef - self.obj,
        #         np.array(
        #             [
        #                 np.linalg.norm(self.eef - self.obj),
        #                 np.linalg.norm(self.eef[:-1] - self.obj[:-1]),
        #                 self.z_target,
        #                 self.z_target - self.obj[-1],
        #                 self.z_target - self.eef[-1],
        #             ]
        #         ),
        #         self.gripper_angle,
        #     ],
        #     axis=0,
        # )

    def get_obs(self):
        if self.obs_type == "state":
            return self._get_obs()
        pixels = self._render(renderer=self.observation_renderer)
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": self.robot_state[:3],
            }
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, pixels_agent_pos]"
            )

    def _sample_goal(self):
        # TODO(now): This is confusing because the getters in Base change co-ordinate system to be relative to the
        # center of the table.
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = self.center_of_table - np.array([0.25, 0, 0.07])
        object_pos[0] += self.np_random.uniform(-0.08, 0.08)
        object_pos[1] += self.np_random.uniform(-0.08, 0.08)
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)
        # Goal
        goal = np.array([1.600, 0.200, 0.545])
        goal[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
        self.target_loc = goal
        return self.target_loc

    def reset(
        self,
        seed=None,
        options: dict | None = None,
    ):
        self._act_magnitude = 0
        self._action = np.zeros(4)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """Action should by just the xyz delta component. The gripper action is fixed to be 0."""
        assert action.shape == (3,)
        action = np.append(action, np.zeros(1, dtype=np.float32))
        self._act_magnitude = np.linalg.norm(action[:3])
        self._action = action.copy()
        return super().step(action)
