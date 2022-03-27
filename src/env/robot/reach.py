import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path



class ReachEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.sample_large = 1
		self.statefull_dim = (11,) if use_xyz else (8,)
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz
		)
		
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
		d = self.goal_distance(achieved_goal, goal, self.use_xyz)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return np.around(-3*d - 0.5*np.square(self._pos_ctrl_magnitude), 4)

	def _get_state_obs(self):
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, values
		], axis=0)

	def _get_achieved_goal(self):
		return self.sim.data.get_site_xpos('grasp').copy()

	def _sample_goal(self, new=True):
		site_id = self.sim.model.site_name2id('target0')

		if new:
			goal = np.array([1.605, 0.3, 0.58])
			goal[0] += self.np_random.uniform(-0.05  - 0.05 * self.sample_large, 0.05 + 0.05 * self.sample_large, size=1)
			goal[1] += self.np_random.uniform(-0.1 - 0.1 * self.sample_large, 0.1 + 0.1 * self.sample_large, size=1)
		else:
			goal = self.sim.data.get_site_xpos('target0')
		

		self.sim.model.site_pos[site_id] = goal
		self.sim.forward()

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.2561169, 0.3, 0.62603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(-0.05, 0.1, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)
