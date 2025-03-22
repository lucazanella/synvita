#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>
#SBATCH --output=output/finetune_lora_synvita_%A.out

export TRANSFORMERS_CACHE="<REPLACE_WITH_PATH_TO_TRANSFORMERS_CACHE>"
export MS_CACHE_HOME="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"
export MODELSCOPE_CACHE="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"

# Activate the virtual environment
VENV_DIR="<REPLACE_WITH_VIDEOLLAVA_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

CKPTS_DIR="<REPLACE_WITH_CHECKPOINTS_DIR>"
# shellcheck disable=SC2006
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

exp_name="train_llm_mix_entail_feedback_cogvideox"

lora_r=32
lora_alpha=32
model_max_length=256
learning_rate=5e-5
warmup_steps=200
per_device_train_batch_size=4
gradient_accumulation_steps=1
num_train_epochs=1

EXP_NAME="${DATETIME}_videollava-7b-lora_${exp_name}"
SAVE_PATH="${CKPTS_DIR}/${EXP_NAME}"

# resume from checkpoint
# pattern="_videollava-7b-lora_${exp_name}"
# SAVE_PATH=$(find "${DIR}" -type d -name "*${pattern}*" | sort | tail -n 1)
# if [ -z "$SAVE_PATH" ]; then
#     echo "No checkpoint found for: ${pattern}"
#     exit 1
# fi
# checkpoint_folder=$(find "$SAVE_PATH" -type d -name "checkpoint-*" | head -n 1)
# if [ -z "$checkpoint_folder" ]; then
#     echo "No 'checkpoint-*' folder found inside: $SAVE_PATH"
#     exit 1
# fi

deepspeed lmms/Video_LLaVA/videollava/train/train_mem_synvita.py \
    --lora_enable True --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --freeze_mm_mlp_adapter \
    --deepspeed ./lmms/Video_LLaVA/scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path "data/${exp_name}.csv" \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder "<REPLACE_WITH_DATASET_DIR>" \
    --features_folder "<REPLACE_WITH_FEATURES_DIR>" \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter "${CKPTS_DIR}/LanguageBind/Video-LLaVA-Pretrain-7B/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "$SAVE_PATH" \
    --num_train_epochs "$num_train_epochs" \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate "${learning_rate}" \
    --weight_decay 0.0001 \
    --warmup_steps "${warmup_steps}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${model_max_length}" \
    --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${EXP_NAME}" \
    --cache_dir "${TRANSFORMERS_CACHE}"  #\
    # --use_extracted_features
