export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name metaworld_ml10_50e \
  --run_root_dir /path/to/output \
  --adapter_tmp_dir /path/to/adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 40000 \
  --save_steps 20000
