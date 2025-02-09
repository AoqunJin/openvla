export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python experiments/robot/metaworld/run_metaworld_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name metaworld_ml10_50e \
  --center_crop True
