import argparse
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='robot')
	parser.add_argument('--task_name', default='reach')
	parser.add_argument('--frame_stack', default=1, type=int)
	parser.add_argument('--episode_length', default=50, type=int)
	parser.add_argument('--n_substeps', default=20, type=int)
	parser.add_argument('--eval_mode', default='none', type=str)
	parser.add_argument('--action_space', default='xy', type=str) # Reach, Push: 'xy'.  Pegbox, Hammerall: 'xyz'
	parser.add_argument('--cameras', default= 0, type=int) # 0: 3rd person, 1: 1st person, 2: both
	parser.add_argument('--observation_type', default='image', type=str) # 'image', 'state+image' # only 'state' not supported
	
	# agent
	parser.add_argument('--algorithm', default='sac', type=str)
	parser.add_argument('--train_steps', default='500k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--hidden_dim_state', default=16, type=int)
	parser.add_argument('--image_size', default=84, type=int)
	parser.add_argument('--attention', default=1, type=int)
	parser.add_argument('--concat', default=0, type=int) # Either concatenate observations (=0) or Add (=1)
	parser.add_argument('--context1', default=1, type=int)
	parser.add_argument('--context2', default=1, type=int)
	parser.add_argument('--load_steps', default='500k', type=str)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=3, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# sac
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--update_freq', default=2, type=int)
	parser.add_argument('--tau', default=0.01, type=float)
	parser.add_argument('--num_expl_steps', default=1000, type=int)
	parser.add_argument('--std_schedule', default='linear(1.0,0.1,0.25)', type=str)
	parser.add_argument('--std_clip', default=0.3, type=float)
	parser.add_argument('--mean_zero', default=False, action='store_true')

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='10k', type=str)
	parser.add_argument('--eval_episodes', default=1, type=int)

	# misc
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--exp_suffix', default='default', type=str)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--num_seeds', default=1, type=int) # Number of seeds to evaluate on
	
	# wandb's setting
	parser.add_argument('--wandb', default=False, action='store_true')
	parser.add_argument('--wandb_project', default='robot_project', type=str)
	parser.add_argument('--wandb_name', default=None, type=str)
	parser.add_argument('--wandb_group', default=None, type=str)
	parser.add_argument('--wandb_job', default=None, type=str)


	args = parser.parse_args()

	assert args.image_size in {64, 84, 128}, f'image size = {args.image_size} is strongly discouraged, use one of [64, 84, 128] (default: 84)'
	assert args.action_space in {'xy', 'xyz', 'xyzw'}, f'specified action_space "{args.action_space}" is not supported'
	assert args.eval_mode in {'train', 'test' ,'color_easy', 'color_hard', 'video_easy', 'video_hard', 'none', None}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.exp_suffix is not None, 'must provide an experiment suffix for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))
	
	if args.eval_mode == 'none':
		args.eval_mode = None
	
	return args
