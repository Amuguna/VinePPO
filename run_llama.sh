CONFIGSTR="configs/llama_3_1_8b_vineppo_MATH.jsonnet"
APP_DIRECTORY="/ext_hdd/yschoi/reasoning_math/vineppo"

export APP_SEED="2746318213"
export WANDB_RUN_ID="vineppo_llama8b" # Optional

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29611
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  --master_port $MASTER_PORT \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop

# # Run the evaluation
# deepspeed --no_local_rank --num_gpus=$NUM_GPUS   \
#          src/treetune/main.py --configs "$CONFIGSTR" \
#             run_evaluation
