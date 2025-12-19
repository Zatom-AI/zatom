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
#        --job-name=finetune-tft-70M-qm9

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

# Define run details
DEFAULT_DATASET="joint"                   # NOTE: Set the dataset to be used, must be one of (`joint`,)
DEFAULT_RUN_ID="gtb6160q"                 # NOTE: Generate a unique ID for each run using `python scripts/generate_id.py`
DEFAULT_RUN_DATE="2025-12-19_10-30-00"    # NOTE: Set this to the initial date and time of the run for unique identification (e.g., ${now:%Y-%m-%d}_${now:%H-%M-%S})
DEFAULT_MODEL="zatom"                     # NOTE: Set the model to be used, must be one of (`zatom`,)
DEFAULT_EXPERIMENT="finetune"             # NOTE: Set the experiment name to be used, must be one of (`train`, `finetune`, `eval`, `overfit`)
DEFAULT_ARCHITECTURE="tft_70M"            # NOTE: Set the model architecture to be used, must be one of (`{tft,}_70M`, `{tft,}_160M`, `{tft,}_300M`, `{mft,mfp}_80M`, `{mft,mfp}_180M`, `{mft,mfp}_500M`)

DATASET=${1:-$DEFAULT_DATASET}            # First argument or default dataset if not provided
RUN_ID=${2:-$DEFAULT_RUN_ID}              # Second argument or default ID if not provided
RUN_DATE=${3:-$DEFAULT_RUN_DATE}          # Third argument or default date if not provided
MODEL=${4:-$DEFAULT_MODEL}                # Fourth argument or default model if not provided
EXPERIMENT=${5:-$DEFAULT_EXPERIMENT}      # Fifth argument or default experiment if not provided
ARCHITECTURE=${6:-$DEFAULT_ARCHITECTURE}  # Sixth argument or default architecture if not provided

TASK_NAME="finetune_fm"                                                         # Name of the task to perform
RUN_NAME="${EXPERIMENT}_model-${MODEL}_arch-${ARCHITECTURE}_qm9_matbench"       # Name of the model type and dataset configuration

PRETRAINED_CKPT_PATH="logs/train_fm/runs/train_model-zatom_arch-tft_70M_joint_fast_2025-12-15_20-00-00/checkpoints/model-epoch@1399-step@43400-val_qm9_valid_rate@0.9471-val_mp20_valid_rate@0.9003.ckpt"  # Path at which to find (initial) pretrained model checkpoint
CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}_${RUN_DATE}/checkpoints/"  # Path at which to find model checkpoints from which to resume
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

echo -e "\nExecuting script train_fm.py:\n========================================================================\n"

# Run script
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME \
    srun --kill-on-bad-exit=1 shifter python zatom/train_fm.py \
    pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
    ckpt_path=$CKPT_PATH \
    callbacks.model_checkpoint.monitor=val_qm9/aux_global_property_loss \
    data=$DATASET \
    data.datamodule.batch_size.train=512 \
    data.datamodule.batch_size.val=512 \
    data.datamodule.batch_size.test=512 \
    data.datamodule.datasets.qm9.proportion=1.0 \
    data.datamodule.datasets.qm9.global_property=all \
    data.datamodule.datasets.matbench.proportion=0.0 \
    data.datamodule.datasets.matbench.global_property=matbench_mp_gap \
    date=$RUN_DATE \
    experiment=$EXPERIMENT \
    model=$MODEL \
    model/architecture=$ARCHITECTURE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.check_val_every_n_epoch=20
"

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
