#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>
#SBATCH --output=logs/entailment_inference_videocon_llm_val_%A.out

export TRANSFORMERS_CACHE="<REPLACE_WITH_PATH_TO_TRANSFORMERS_CACHE>"
export MS_CACHE_HOME="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"
export MODELSCOPE_CACHE="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"

CKPTS_DIR="<REPLACE_WITH_CHECKPOINTS_DIR>"

# Activate the virtual environment
VENV_DIR="<REPLACE_WITH_MPLUGOWL_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

data_dir="<REPLACE_WITH_DATASET_DIR>"
input_csv="datasets/final_videocon_llm_val.csv"
pretrained_ckpt="${CKPTS_DIR}/MAGAer13/mplug-owl-llama-7b-video/"
ckpt="${CKPTS_DIR}/videocon_model_lora_train_llm_mix_entail_feedback_cogvideox/checkpoint-5180/"

process_ckpt() {
    lora_alpha=32
    lora_r=32

    local trained_ckpt="$1/pytorch_model.bin"
    local output_csv="$1/final_videocon_llm_val_scores.csv"
    local output_file="$1/videocon_llm_val_results.csv"

    echo "Running inference on $input_csv using $trained_ckpt"
    echo "LORA alpha: $lora_alpha, LORA rank: $lora_r"
    python lmms/mPLUG_Owl/pipeline/entailment_inference.py \
        --data_dir "$data_dir" \
        --input_csv "$input_csv" \
        --output_csv "$output_csv" \
        --trained_ckpt "$trained_ckpt" \
        --pretrained_ckpt "$pretrained_ckpt" \
        --use_lora \
        --all-params \
        --lora_alpha "$lora_alpha" \
        --lora_r "$lora_r"  # \
        # --use-extracted-features \

    python src/eval_videocon_llm.py \
        --data_dir "$data_dir" \
        --input_csv_1 "$input_csv" \
        --input_csv_2 "$output_csv" \
        --output_file "$output_file"
}

echo "Processing $ckpt"
process_ckpt "$ckpt"
