import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random
import subprocess
import platform
from datetime import datetime


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'host': platform.node(),
		'cwd': os.getcwd(),
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def prefill_memory(capacity, obs_shape):
	obses = []
	if len(obs_shape) > 1:
		c,h,w = obs_shape
		for _ in range(capacity):
			if c==6:
				frame = np.ones((6,h,w), dtype=np.uint8)
			else:
				frame = np.ones((3,h,w), dtype=np.uint8)
			obses.append(frame)
	else:
		for _ in range(capacity):
			obses.append(np.ones(obs_shape[0], dtype=np.float32))

	return obses


class ReplayBuffer(Dataset):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size):
		self.capacity = capacity
		self.batch_size = batch_size
		self.state_shape = state_shape
		self._obses = prefill_memory(capacity, obs_shape)
		if self.state_shape is not None:
			self._states = prefill_memory(capacity, state_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.idx = 0
		self.full = False

	def __len__(self):
		return self.capacity if self.full else self.idx

	def __getitem__(self, idx):
		obs, next_obs = self._encode_obses(idx)
		if self.state_shape is not None:
			state, next_state = self._encode_states(idx)
		obs = torch.as_tensor(obs).cuda().float()
		if self.state_shape is not None:
			state = torch.as_tensor(state).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		if self.state_shape is not None:
			next_state = torch.as_tensor(next_state).cuda().float()
		actions = torch.as_tensor(self.actions[idx]).cuda()
		rewards = torch.as_tensor(self.rewards[idx]).cuda()

		if self.state_shape is not None:
			return obs, state, actions, rewards, next_obs, next_state
		else:
			return obs, None, actions, rewards, next_obs, None

	def add(self, obs, state, action, reward, next_obs, next_state):
		self._obses[self.idx] = (obs, next_obs)
		if self.state_shape is not None:
			self._states[self.idx] = (state, next_state)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(0, len(self), size=n)

	def _encode_obses(self, idxs):
		obses, next_obses = zip(*[self._obses[i] for i in idxs])
		return np.array(obses), np.array(next_obses)

	def _encode_states(self, idxs):
		states, next_states = zip(*[self._states[i] for i in idxs])
		return np.array(states), np.array(next_states)

	def sample_drq(self, n=None, pad=4):
		raise NotImplementedError('call sample() and apply aug in the agent.update() instead')

	def sample(self, n=None):
		idxs = self._get_idxs(n)
		return self[idxs]
		

class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]
