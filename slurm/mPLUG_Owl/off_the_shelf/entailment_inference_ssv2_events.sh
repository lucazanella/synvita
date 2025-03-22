#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>
#SBATCH --output=logs/entailment_inference_ssv2_events_%A.out

export TRANSFORMERS_CACHE="<REPLACE_WITH_PATH_TO_TRANSFORMERS_CACHE>"
export MS_CACHE_HOME="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"
export MODELSCOPE_CACHE="<REPLACE_WITH_PATH_TO_MODELSCOPE_CACHE>"

CKPTS_DIR="<REPLACE_WITH_CHECKPOINTS_DIR>"

# Activate the virtual environment
VENV_DIR="<REPLACE_WITH_MPLUGOWL_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

data_dir="<REPLACE_WITH_DATASET_DIR>"
input_csv="datasets/eval_ssv2_events_updated.csv"
pretrained_ckpt="${CKPTS_DIR}/MAGAer13/mplug-owl-llama-7b-video/"

process_ckpt() {
    local output_csv="$1/ssv2_events_scores.csv"
    local output_file="$1/ssv2_events_results.txt"

    echo "Running inference on $input_csv using $pretrained_ckpt"
    python lmms/mPLUG_Owl/pipeline/entailment_inference.py \
        --data_dir "$data_dir" \
        --input_csv "$input_csv" \
        --output_csv "$output_csv" \
        --pretrained_ckpt "$pretrained_ckpt" \
        --use-extracted-features

    python src/calc_ssv2.py \
        --data_dir "$data_dir" \
        --input_file_1 "$input_csv" \
        --input_file_2 "$output_csv" \
        --vid_per_caption 588 \
        --output_file "$output_file"
}

echo "Processing $pretrained_ckpt"
process_ckpt "$pretrained_ckpt"
