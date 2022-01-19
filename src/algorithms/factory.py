from algorithms.sac import SAC

algorithm = {
	'sac': SAC,
}

def make_agent(obs_shape, state_shape , action_shape, args):
	return algorithm[args.algorithm](obs_shape, state_shape, action_shape, args)
