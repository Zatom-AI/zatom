#!/bin/bash -l

# salloc -C "gpu&hbm80g" \
#        --qos=shared_interactive \
#        --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 \
#        --module=gpu,nccl-plugin \
#        --account=m5008 \
#        --nodes=1 \
#        --gpus-per-node=1 \
#        --ntasks-per-node=1 \
#        --time=04:00:00 \
#        --job-name=ebm

# Determine location of the project's directory
# PROJECT_ID="dasrepo"
# PROJECT_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/Repositories/zatom" # long term storage community drive
PROJECT_DIR="/pscratch/sd/${USER:0:1}/$USER/Repositories/zatom" # high-performance storage scratch drive with an 8-week purge policy
cd "$PROJECT_DIR" || exit

# Establish environment variables
# export TORCH_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/torch_cache" # long term storage community drive
# export HF_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/hf_cache" # long term storage community drive
export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache" # high-performance storage scratch drive with an 8-week purge policy
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"       # high-performance storage scratch drive with an 8-week purge policy

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

# Select model configuration -> EBT-{S/B/L}
D_MODEL=768  # 384, 768, 1024
NUM_LAYERS=12  # 12, 12, 24
NHEAD=12  # 6, 12, 16
# NOTE: For EBT-L, append the following options to your `python train.py` command: data.datamodule.batch_size.train=55 trainer.accumulate_grad_batches=8

# Define run details
DEFAULT_DATASET="joint"                   # NOTE: Set the dataset to be used, must be one of (`joint`, `qm9_only`, `mp20_only`, `qmof150_only`, `omol25_only`, `geom_only`)
DEFAULT_RUN_ID="5buon1wv"                 # NOTE: Generate a unique ID for each run using `python scripts/generate_id.py`
DEFAULT_RUN_DATE="2025-09-16_14-00-00"    # NOTE: Set this to the initial date and time of the run for unique identification (e.g., ${now:%Y-%m-%d}_${now:%H-%M-%S})

DATASET=${1:-$DEFAULT_DATASET}            # First argument or default dataset if not provided
RUN_NAME="EBT-B__${DATASET}_subset_s1_debugging"       # Name of the model type and dataset configuration
RUN_ID=${2:-$DEFAULT_RUN_ID}              # First argument or default ID if not provided
RUN_DATE=${3:-$DEFAULT_RUN_DATE}          # Second argument or default date if not provided

TASK_NAME="train_ebm"                     # Name of the task to perform

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
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME \
    srun --kill-on-bad-exit=1 shifter python zatom/$TASK_NAME.py \
    data=$DATASET \
    data.datamodule.datasets.mp20.proportion=1.0 \
    data.datamodule.datasets.qm9.proportion=1.0 \
    data.datamodule.datasets.qmof150.proportion=0.0 \
    data.datamodule.datasets.omol25.proportion=0.0 \
    data.datamodule.datasets.geom.proportion=0.0 \
    date=$RUN_DATE \
    ecoder.d_model=$D_MODEL \
    ecoder.num_layers=$NUM_LAYERS \
    ecoder.nhead=$NHEAD \
    logger=wandb \
    name=$RUN_NAME \
    strategy=optimized_ddp \
    task_name=$TASK_NAME \
    trainer=ddp \
    trainer.accumulate_grad_batches=1 \
    trainer.check_val_every_n_epoch=500 \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.overfit_batches=1 \
    ckpt_path=$CKPT_PATH
"

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
