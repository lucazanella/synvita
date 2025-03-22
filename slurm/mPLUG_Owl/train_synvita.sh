#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>
#SBATCH --exclusive
#SBATCH --output=output/train_synvita_mplug_owl_%A.out

export OMP_NUM_THREADS=8

# Activate the virtual environment
VENV_DIR="<REPLACE_WITH_MPLUGOWL_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

CKPTS_DIR="<REPLACE_WITH_CHECKPOINTS_DIR>"
# shellcheck disable=SC2006
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

if [ "$MASTER_ADDR" ]; then
	echo "$MASTER_ADDR"
    echo "$MASTER_PORT"
    echo "$WORLD_SIZE"
    echo "$RANK"
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$((RANDOM % 10))$((RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

exp_name="train_llm_mix_entail_feedback_cogvideox"

EXP_NAME="${DATETIME}_videocon_model_lora_${exp_name}"
SAVE_NAME="${DATETIME}_videocon_model_lora_${exp_name}"

echo "EXP_NAME: ${EXP_NAME}"

SAVE_PATH="${CKPTS_DIR}/${SAVE_NAME}"

lora_r=32
lora_alpha=32
max_length=256
micro_batch_size=8
gradient_accumulation_steps=1

train_epochs=1
lr_warmup_iters=200
eval_iter=100000

pretrained_ckpt="${CKPTS_DIR}/MAGAer13/mplug-owl-llama-7b-video/"

mkdir -p "${SAVE_PATH}"

options=" \
	--pretrained-ckpt ${pretrained_ckpt} \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr 1e-4 \
	--min-lr 1e-7 \
	--eval-iters ${eval_iter} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 8 \
	--use-lora \
	--all-params \
	--lora-r ${lora_r} \
	--lora-alpha ${lora_alpha} \
	--gradient-checkpointing \
	--wandb_run_name ${EXP_NAME} \
	--bf16 \
	--loss_objective sequential"  # \
	# --use-extracted-features"

multimodal_options=" \
	--mm-config lmms/mPLUG_Owl/configs/${exp_name}.yaml
    "

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch $DISTRIBUTED_ARGS lmms/mPLUG_Owl/pipeline/train_synvita.py "$@" ${options} ${multimodal_options} 2>&1 | tee "${SAVE_PATH}/train.log"
