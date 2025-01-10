export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl

python experiments/robot/metaworld/run_metaworld_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /data/xtydata/openvla/metaworld_ml10_40e/openvla-7b+metaworld_ml10_40e+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name metaworld_ml10_40e \
  --center_crop True