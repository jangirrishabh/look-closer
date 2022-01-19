import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
#from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations
import cv2


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

		episode_rewards.append(episode_reward)

	return np.nanmean(episode_rewards), np.nanmean(success_rate)


def main(args):

	seed_rewards, seed_success = [], []
	for s in range(args.num_seeds):
		# Set seed
		utils.set_seed_everywhere(args.seed + s)
		if args.cameras==0:
			cameras=['third_person']
		elif args.cameras==1:
			cameras=['first_person']
		elif args.cameras==2:
			cameras = ['third_person', 'first_person']

		# Initialize environments
		gym.logger.set_level(40)

		env = make_env(
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
		work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, dir_name ,str(args.seed + s))
		print('Working directory:', work_dir)

		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		video_dir = utils.make_dir(os.path.join(work_dir, 'video'))

		if not os.path.exists(os.path.join(model_dir, str(args.train_steps)+'.pt')):
			print("Skipping evaluation for ", work_dir)
			continue

		video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

		print("working IDR", work_dir, args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix)
		# Check if evaluation has already been run
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
		assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

		# Prepare agent
		assert torch.cuda.is_available(), 'must have cuda enabled'
		print('Observations:', env.observation_space.shape)
		print('Action space:', f'{args.action_space} ({env.action_space.shape[0]})')
		print("STATE ", env.state_space_shape)
		agent = make_agent(
			obs_shape=env.observation_space.shape,
			state_shape=env.state_space_shape,
			action_shape=env.action_space.shape,
			args=args
		)
		agent = torch.load(os.path.join(model_dir, str(args.train_steps)+'.pt'))
		agent.train(False)

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward, success_rate = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, args.image_size)
		print('Reward:', int(reward))
		print('Success Rate:', success_rate)
		seed_rewards.append(int(reward))
		seed_success.append(success_rate)


	print('Average Reward over all the seeds:', int(np.nanmean(seed_rewards)), np.nanmean(seed_success))


if __name__ == '__main__':
	args = parse_args()
	main(args)
