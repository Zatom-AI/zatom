#!/bin/bash -l

# salloc -C "gpu&hbm80g" \
#        --qos=interactive \
#        --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 \
#        --module=gpu,nccl-plugin \
#        --account=m5008 \
#        --nodes=4 \
#        --gpus-per-node=4 \
#        --ntasks-per-node=4 \
#        --time=04:00:00 \
#        --job-name=tfp-80M-joint

# Determine location of the project's directory
# PROJECT_ID="dasrepo"
# PROJECT_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/Repositories/zatom"            # long term storage community drive
PROJECT_DIR="/pscratch/sd/${USER:0:1}/$USER/Repositories/zatom"                   # high-performance storage scratch drive with an 8-week purge policy
cd "$PROJECT_DIR" || exit

# Establish environment variables
# export TORCH_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/torch_cache"             # long term storage community drive
# export HF_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/hf_cache"                   # long term storage community drive
# export WANDB_CACHE_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/wandb_cache"        # long term storage community drive
# export WANDB_ARTIFACT_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/wandb_artifacts" # long term storage community drive
export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache"                    # high-performance storage scratch drive with an 8-week purge policy
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"                          # high-performance storage scratch drive with an 8-week purge policy
export WANDB_CACHE_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_cache"               # high-performance storage scratch drive with an 8-week purge policy
export WANDB_ARTIFACT_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_artifacts"        # high-performance storage scratch drive with an 8-week purge policy

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_ARTIFACT_DIR"

# Define run details
DEFAULT_DATASET="joint"                   # NOTE: Set the dataset to be used, must be one of (`joint`,)
DEFAULT_RUN_ID="igbyenbv"                 # NOTE: Generate a unique ID for each run using `python scripts/generate_id.py`
DEFAULT_RUN_DATE="2026-01-02_11-30-00"    # NOTE: Set this to the initial date and time of the run for unique identification (e.g., ${now:%Y-%m-%d}_${now:%H-%M-%S})
DEFAULT_MODEL="zatom"                     # NOTE: Set the model to be used, must be one of (`zatom`,)
DEFAULT_EXPERIMENT="train"                # NOTE: Set the experiment name to be used, must be one of (`train`, `finetune`, `eval`, `overfit`)
DEFAULT_ARCHITECTURE="tfp_80M"            # NOTE: Set the model architecture to be used, must be one of (`{tft,tfp}_80M`, `{tft,tfp}_160M`, `{tft,tfp}_300M`)

DATASET=${1:-$DEFAULT_DATASET}            # First argument or default dataset if not provided
RUN_ID=${2:-$DEFAULT_RUN_ID}              # Second argument or default ID if not provided
RUN_DATE=${3:-$DEFAULT_RUN_DATE}          # Third argument or default date if not provided
MODEL=${4:-$DEFAULT_MODEL}                # Fourth argument or default model if not provided
EXPERIMENT=${5:-$DEFAULT_EXPERIMENT}      # Fifth argument or default experiment if not provided
ARCHITECTURE=${6:-$DEFAULT_ARCHITECTURE}  # Sixth argument or default architecture if not provided

TASK_NAME="train_fm"                                                   # Name of the task to perform
RUN_NAME="${EXPERIMENT}_model-${MODEL}_arch-${ARCHITECTURE}_joint"     # Name of the model type and dataset configuration

CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}_${RUN_DATE}/checkpoints/" # Path at which to find model checkpoints
mkdir -p "$CKPT_PATH"

# Inform user of job details
echo -e "Job details:\n========================================================================\n"

echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run start time: $RUN_DATE"

echo -e "\nSLURM job name: $SLURM_JOB_NAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM master node: $SLURMD_NODENAME"
echo "SLURM all nodes: $SLURM_NODELIST"
echo "SLURM node count: $SLURM_JOB_NUM_NODES"

echo -e "\nCUDA visible devices: $CUDA_VISIBLE_DEVICES"

echo -e "\nCurrent time: $(date)"
echo "Current directory: $(pwd)"
echo "Current node: $(hostname)"

echo -e "\nExecuting script $TASK_NAME.py:\n========================================================================\n"

# Run script
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 shifter python zatom/$TASK_NAME.py \
    callbacks.last_model_checkpoint.every_n_train_steps=3500 \
    ckpt_path=$CKPT_PATH \
    data=$DATASET \
    date=$RUN_DATE \
    experiment=$EXPERIMENT \
    model=$MODEL \
    model/architecture=$ARCHITECTURE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE
"

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
