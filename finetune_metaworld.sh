export CUDA_VISIBLE_DEVICES=3

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name metaworld_ml10_100e \
  --run_root_dir /data/xtydata/openvla/metaworld_ml10_100e \
  --adapter_tmp_dir /data/xtydata/openvla/tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla \
  --wandb_entity hbnu_ai \
  --max_steps 4000 \
  --save_steps 2000
