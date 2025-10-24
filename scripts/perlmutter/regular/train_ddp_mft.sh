#!/bin/bash -l

######################### Batch Headers #########################
#SBATCH -C gpu&hbm40g                                         # request GPU nodes
#SBATCH --qos=regular                                         # use specified partition for job
#SBATCH --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 # use specified container image
#SBATCH --module=gpu,nccl-plugin                              # load GPU and optimized NCCL plugin modules
#SBATCH --account=m5008                                       # use specified account for billing (e.g., `m5008_g` for AI4Science proposal, `dasrepo` for all else)
#SBATCH --nodes=2                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gpus-per-node=4                                     # request A100 GPU resource(s)
#SBATCH --ntasks-per-node=4                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time=00-20:00:00                                    # time limit for the job (up to 2 days: `02-00:00:00`)
#SBATCH --job-name=mft-80M                                    # job name
#SBATCH --output=scripts/perlmutter/regular/logs/train%j.out  # output log file
#SBATCH --error=scripts/perlmutter/regular/logs/train%j.err   # error log file

# Wait for 5-10 seconds randomly to avoid race condition
sleep $((RANDOM % 6 + 5))

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
DEFAULT_RUN_ID="wnv94pae"                 # NOTE: Generate a unique ID for each run using `python scripts/generate_id.py`
DEFAULT_RUN_DATE="2025-10-24_12-00-00"    # NOTE: Set this to the initial date and time of the run for unique identification (e.g., ${now:%Y-%m-%d}_${now:%H-%M-%S})
DEFAULT_ARCHITECTURE="mft_80M"            # NOTE: Set the model architecture to be used, must be one of (`{mft,met,mfp}_80M`, `{mft,met,mfp}_180M`, `{mft,met,mfp}_500M`)

DATASET=${1:-$DEFAULT_DATASET}            # First argument or default dataset if not provided
RUN_ID=${2:-$DEFAULT_RUN_ID}              # Second argument or default ID if not provided
RUN_DATE=${3:-$DEFAULT_RUN_DATE}          # Third argument or default date if not provided
ARCHITECTURE=${4:-$DEFAULT_ARCHITECTURE}  # Fourth argument or default architecture if not provided

TASK_NAME="train_fm"                           # Name of the task to perform
RUN_NAME="train_arch-${ARCHITECTURE}_QM9"      # Name of the model type and dataset configuration

CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}_${RUN_DATE}/checkpoints/" # Path at which to find model checkpoints
mkdir -p "$CKPT_PATH"

# Inform user of job details
echo -e "Job details:\n==================\n"

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

echo -e "\nExecuting script $TASK_NAME.py:\n==================\n"

# Run script
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME \
    srun --kill-on-bad-exit=1 shifter python zatom/$TASK_NAME.py \
    callbacks.model_checkpoint.monitor=val_qm9/valid_rate \
    ckpt_path=$CKPT_PATH \
    data=$DATASET \
    data.datamodule.datasets.mp20.proportion=0.0 \
    date=$RUN_DATE \
    experiment=train \
    model/architecture=$ARCHITECTURE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE
"

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
