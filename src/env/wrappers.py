import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gym.wrappers import TimeLimit
from env.robot import registration
import utils
from collections import deque
from mujoco_py import modder


def make_env(
		domain_name,
		task_name,
		seed=0,
		episode_length=50,
		n_substeps=20,
		frame_stack=3,
		image_size=84,
		cameras=['third_person', 'first_person'],
		mode='train',
		observation_type='image',
		action_space='xyzw'
	):
	"""Make environment for experiments"""
	assert domain_name == 'robot', f'expected domain_name "robot", received "{domain_name}"'
	assert action_space in {'xy', 'xyz', 'xyzw'}, f'unexpected action space "{action_space}"'

	registration.register_robot_envs(
		n_substeps=n_substeps,
		observation_type=observation_type,
		image_size=image_size,
		use_xyz=action_space.replace('w', '') == 'xyz')
	randomizations = {}
	env_id = 'Robot' + task_name.capitalize() + '-v0'
	env = gym.make(env_id, cameras=cameras, render=False, observation_type=observation_type)
	env.seed(seed)
	env = TimeLimit(env, max_episode_steps=episode_length)
	env = SuccessWrapper(env, any_success=True)
	
	env = ObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size, cameras=cameras)
	env = ActionSpaceWrapper(env, action_space=action_space)
	env = FrameStack(env, frame_stack)

	return env


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		if len(shp) == 3:
			self.observation_space = gym.spaces.Box(
				low=0,
				high=1,
				shape=((shp[0] * k,) + shp[1:]),
				dtype=env.observation_space.dtype
			)
		else:
			self.observation_space = gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=(shp[0] * k,),
				dtype=env.observation_space.dtype
			)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs, state_obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs(), state_obs

	def step(self, action):
		obs, state_obs, reward, done, info = self.env.step(action)
		self._frames.append(obs)
		return self._get_obs(), state_obs, reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return utils.LazyFrames(list(self._frames))


class SuccessWrapper(gym.Wrapper):
	def __init__(self, env, any_success=True):
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.any_success = any_success
		self.success = False

	def reset(self):
		self.success = False
		return self.env.reset()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		if self.any_success:
			self.success = self.success or bool(info['is_success'])
		else:
			self.success = bool(info['is_success'])
		info['is_success'] = self.success
		return obs, reward, done, info


class ObservationSpaceWrapper(gym.Wrapper):
	def __init__(self, env, observation_type, image_size, cameras):
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.observation_type = observation_type
		self.image_size = image_size
		self.cameras = cameras
		self.num_cams = len(self.cameras)

		if self.observation_type in {'image', 'state+image'}:
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3*self.num_cams, image_size, image_size), dtype=np.uint8)

		elif self.observation_type == 'state':
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim, dtype=np.float32)


	def reset(self):
		obs = self.env.reset()
		return self._get_obs(obs), obs['state'] if 'state' in obs else None

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return self._get_obs(obs), obs['state'] if 'state' in obs else None, reward, done, info

	def _get_obs(self, obs_dict):
		if self.observation_type in {'image', 'state+image'}:
			if self.num_cams == 1:
				return obs_dict['observation'][0].transpose(2, 0, 1)
			obs = np.empty((3*self.num_cams, self.image_size, self.image_size), dtype=obs_dict['observation'][0].dtype)
			for ob in range(obs_dict['observation'].shape[0]):
				obs[3*ob:3*(ob+1)] = obs_dict['observation'][ob].transpose(2, 0, 1)

		elif self.observation_type == 'state':
			obs = obs_dict['observation']

		return obs


class ActionSpaceWrapper(gym.Wrapper):
	def __init__(self, env, action_space):
		assert action_space in {'xy', 'xyz', 'xyzw'}, 'task must be one of {xy, xyz, xyzw}'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.action_space_dims = action_space
		self.use_xyz = 'xyz' in action_space
		self.use_gripper = 'w' in action_space
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2+self.use_xyz+self.use_gripper,), dtype=np.float32)
	
	def step(self, action):
		assert action.shape == self.action_space.shape, 'action shape must match action space'
		action = np.array([action[0], action[1], action[2] if self.use_xyz else 0, action[3] if self.use_gripper else 1], dtype=np.float32)
		return self.env.step(action)
