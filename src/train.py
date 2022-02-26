import torch
import os

import numpy as np
import gym
import utils
import time
import wandb
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def evaluate(env, agent, video, num_episodes, L, step, test_env=False, use_wandb=False):
	episode_rewards = []
	success_rate = []
	for i in range(num_episodes):
		obs, state = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.select_action(obs, state)
			obs, state, reward, done, info = env.step(action)
			video.record(env)
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
			if args.wandb:
				wandb.log({'eval/episode_reward':episode_reward})
			if 'is_success' in info:
				L.log(f'eval/sucess_rate', success, step)
		episode_rewards.append(episode_reward)

	return np.nanmean(episode_rewards), np.nanmean(success_rate)


def main(args):
	# init wandb
	if args.wandb:
		wandb.init(project=args.wandb_project, name=args.wandb_name, \
		group=args.wandb_group, job_type=args.wandb_job)

	# Set seed
	utils.set_seed_everywhere(args.seed)
	if args.cameras==0:
		cameras=['third_person']
	elif args.cameras==1:
		cameras=['first_person']
	elif args.cameras==2:
		cameras = ['third_person', 'first_person']
	else:
		raise Exception('Current Camera Pose Not Supported.')

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		mode='train',
		cameras=cameras, #['third_person', 'first_person']
		observation_type=args.observation_type,
		action_space=args.action_space
	)
	
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		mode=args.eval_mode,
		cameras=cameras, #['third_person', 'first_person']
		observation_type=args.observation_type,
		action_space=args.action_space,
	) if args.eval_mode is not None else None

	dir_name = 'crossview_attention'

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, dir_name ,str(args.seed))
	print('Working directory:', work_dir)
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448, fps=15 if args.domain_name == 'robot' else 25)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	print('Observations:', env.observation_space.shape)
	print('Action space:', f'{args.action_space} ({env.action_space.shape[0]})')
	agent = make_agent(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	# UNCOMMENT NEXT LINE TO LOAD A TRAINED AGENT
	#agent = torch.load(os.path.join(model_dir, str(args.load_steps)+'.pt'))

	start_step, episode, episode_reward, info, done, episode_success = 0, 0, 0, {}, True, 0
	L = Logger(work_dir)
	start_time = time.time()
	
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step, use_wandb=args.wandb)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True, use_wandb=args.wandb)
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			L.log('train/success_rate', episode_success/args.episode_length, step)
			if args.wandb:
				wandb.log({'train/episode_reward':episode_reward, \
				'train/success_rate':episode_success/args.episode_length})

			obs, state = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1
			episode_success = 0

			L.log('train/episode', episode, step)

		# Sample action and update agent
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.sample_action(obs, state, step)
			num_updates = args.init_steps//args.update_freq if step == args.init_steps else 1
			for i in range(num_updates):
				agent.update(replay_buffer=replay_buffer, L=L, step=step)

		# Take step
		next_obs, next_state, reward, done, info = env.step(action)
		replay_buffer.add(obs, state, action, reward, next_obs, next_state)
		episode_reward += reward
		obs = next_obs
		state = next_state
		episode_success+=float(info['is_success'])
		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
