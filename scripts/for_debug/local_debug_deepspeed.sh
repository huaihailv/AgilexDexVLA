#!/bin/bash
LLM=qwen2_vl
LLM_MODEL_SIZE=2B

ACTION_HEAD=scale_dp_policy  # unet_diffusion_policy or scale_dp_policy

DIT_PRETRAIN=/mnt/hpfs/baaiei/lvhuaihai/model/scaledp_l/open_scale_dp_l_backbone.ckpt
MNOP=/mnt/hpfs/baaiei/lvhuaihai/model/qwenvla2_2b # official qwen2_vl weights
TASKNAME=multi-task

OUTPUT=/mnt/hpfs/baaiei/lvhuaihai/DexVLA/qwen2_lora
touch $OUTPUT/log.txt

deepspeed --master_port 29604 --num_gpus=1 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning False \
  --lora_enable True \
  --action_dim 14 \
  --state_dim 12 \
  --flash_attn False \
  --chunk_size 50 \
  --lora_module "vit llm" \
  --using_film True \
  --using_ema False \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_L" \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --episode_first False \
  --task_name $TASKNAME \
  --model_name_or_path $MNOP \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 80000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.001 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log | tee $OUTPUT/log.log

for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done

mv ./60030.log $OUTPUT
echo $OUTPUT