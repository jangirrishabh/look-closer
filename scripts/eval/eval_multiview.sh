export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
  --algorithm sac \
  --domain_name robot \
  --task_name  reach \
  --episode_length 50 \
  --exp_suffix ours \
  --eval_mode 'test' \
  --save_video \
  --eval_freq 250k \
  --train_steps 1000k \
  --save_freq 250k \
  --log_dir logs \
  --seed 99 \
  --num_seeds 3 \
  --cameras 2 \
  --action_space xy \
  --attention 1 \
  --concat 0 \
  --observation_type image \
  --context1 1 \
  --context2 1 \