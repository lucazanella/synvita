#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>
#SBATCH --output=logs/entailment_inference_videocon_llm_%A.out

export TRANSFORMERS_CACHE="<REPLACE_WITH_PATH_TO_TRANSFORMERS_CACHE>"
export MS_CACHE_HOME="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"
export MODELSCOPE_CACHE="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"

CKPTS_DIR="<REPLACE_WITH_CHECKPOINTS_DIR>"

# Activate the virtual environment
VENV_DIR="<REPLACE_WITH_VIDEOLLAVA_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

data_dir="<REPLACE_WITH_DATASET_DIR>"
input_csv="datasets/final_videocon_llm_test.csv"
model_base="${TRANSFORMERS_CACHE}/models--LanguageBind--Video-LLaVA-7B/snapshots/aecae02b7dee5c249e096dcb0ce546eb6f811806"
# ckpt="${CKPTS_DIR}/videollava-7b-lora_train_llm_mix_entail_feedback/checkpoint-5182/"
ckpt="${CKPTS_DIR}/videollava-7b-lora_train_llm_mix_entail_feedback_cogvideox/checkpoint-5181/"

process_ckpt() {
    batch_size=8
    # local trained_ckpt="$1/pytorch_model.bin"
    local model_path
    model_path=$(dirname "$1")
    local output_csv="$1/final_videocon_llm_test_scores.csv"
    local output_file="$1/videocon_llm_test_results.csv"

    echo "Running inference on $input_csv using $model_path"
    python lmms/Video_LLaVA/videollava/entailment_inference.py \
        --data_dir "$data_dir" \
        --input_csv "$input_csv" \
        --output_csv "$output_csv" \
        --model_path "$model_path" \
        --model_base "$model_base" \
        --batch_size "$batch_size" \
        --cache_dir "$TRANSFORMERS_CACHE"

    python src/eval_videocon_llm.py \
        --data_dir "$data_dir" \
        --input_csv_1 "$input_csv" \
        --input_csv_2 "$output_csv" \
        --output_file "$output_file" \
        --extract_caption
}

echo "Processing $ckpt"
process_ckpt "$ckpt"
