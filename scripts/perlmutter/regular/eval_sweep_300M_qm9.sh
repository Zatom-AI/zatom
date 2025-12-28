#!/bin/bash -l

######################### Batch Headers #########################
#SBATCH -C gpu&hbm40g                                         # request GPU nodes
#SBATCH --qos=shared                                          # use specified partition for job
#SBATCH --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 # use specified container image
#SBATCH --module=gpu,nccl-plugin                              # load GPU and optimized NCCL plugin modules
#SBATCH --account=m5008                                       # use specified account for billing (e.g., `m5008_g` for AI4Science proposal, `dasrepo` for all else)
#SBATCH --nodes=1                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gpus-per-node=1                                     # request A100 GPU resource(s)
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time=00-11:00:00                                    # time limit for the job (up to 2 days: `02-00:00:00`)
#SBATCH --job-name=eval-sweep-300M-qm9                        # job name
#SBATCH --output=scripts/perlmutter/regular/logs/eval_sweep_300M_qm9%j.out  # output log file
#SBATCH --error=scripts/perlmutter/regular/logs/eval_sweep_300M_qm9%j.err   # error log file
#SBATCH --array=0-7                                           # create an array of jobs for the sweep (0-11 or 12 total for finetuning and 0-7 or 8 total for generative evaluation)

# Wait for 5-10 seconds randomly to avoid race condition
sleep $((RANDOM % 6 + 5))

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
DEFAULT_SWEEP_ID="3zx8v4ei"                   # NOTE: Generate a unique ID for each run by running `wandb sweep configs/sweep/{train,eval}_sweep_{joint,}.yaml`
SWEEP_ID=${1:-$DEFAULT_SWEEP_ID}              # First argument or default ID if not provided

# Inform user of job details
echo -e "Job details:\n==================\n"

echo "Sweep ID: $SWEEP_ID"

echo -e "\nSLURM job name: $SLURM_JOB_NAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM master node: $SLURMD_NODENAME"
echo "SLURM all nodes: $SLURM_NODELIST"
echo "SLURM node count: $SLURM_JOB_NUM_NODES"

echo -e "\nCUDA visible devices: $CUDA_VISIBLE_DEVICES"

echo -e "\nCurrent time: $(date)"
echo "Current directory: $(pwd)"
echo "Current node: $(hostname)"

echo -e "\nExecuting sweep:\n==================\n"

# Launch sweep
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 shifter wandb agent zatom/zatom/$SWEEP_ID --count 1
"

# Inform user of sweep completion
echo "Sweep completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
