#!/bin/bash -l

if [ $# -lt 2 ]; then
    echo "Error: At least two arguments required."
    echo "Usage: $0 {qm9|geom} <architecture> [run_id] [run_date] [experiment]"
    echo "       {qm9|geom}     is the dataset to be trained on"
    echo "       <architecture> is an architecture config (e.g. tfp_xxx.yaml)"
    echo "       [run_id]       is the unique ID for the run"
    echo "       [run_date]     is the initial date/time of the run"
    echo "       [experiment]   points to the experiment config (defaults to 'train')"
    exit 1
fi

### Define run details

DEFAULT_RUN_ID="xxxxxxxx"               # NOTE: Generate a unique ID for each run using `python scripts/generate_id.py`
DEFAULT_RUN_DATE="2026-01-01_12-00-00"  # NOTE: Set this to the initial date and time of the run for unique identification
DEFAULT_EXPERIMENT="train"              # NOTE: Set the experiment name to be used, must be one of (`train`, `finetune`, `eval`, `overfit`)

MODEL="zatom"                           # NOTE: Set the model to be used, must be one of (`zatom`,)
DATASET="joint"                         # NOTE: Set the dataset to be used, must be one of (`joint`,)
MOLECULE_DATASET="${1:?Missing first argument for molecule-only dataset (qm9 or geom)}"
ARCHITECTURE="${2:?Missing second argument for TFP model config)}"
RUN_ID=${3:-$DEFAULT_RUN_ID}            # 3rd argument or default ID         if not provided
RUN_DATE=${4:-$DEFAULT_RUN_DATE}        # 4th argument or default date       if not provided
EXPERIMENT=${5:-$DEFAULT_EXPERIMENT}    # 5th argument or default experiment if not provided

TASK_NAME="train_fm"                                                             # Name of the task to perform
RUN_NAME="${EXPERIMENT}_model-${MODEL}_arch-${ARCHITECTURE}_${MOLECULE_DATASET}" # Name of the model type and dataset configuration

CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}_${RUN_DATE}/checkpoints/" # Path at which to find model checkpoints
mkdir -p "$CKPT_PATH"

PROJECT_DIR="/path/to/zatom"
cd "$PROJECT_DIR" || exit

export TORCH_HOME="/path/to/torch_home"
export HF_HOME="/path/to/hf_home"
export WANDB_CACHE_DIR="/path/to/wandb_cache_dir"
export WANDB_ARTIFACT_DIR="/path/to/wandb_artifact_dir"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"


### Inform user of job details
echo
echo "========================================================================"
echo "Job details:"
echo
echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run start time: $RUN_DATE"
echo
echo "SLURM job name: $SLURM_JOB_NAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM master node: $SLURMD_NODENAME"
echo "SLURM all nodes: $SLURM_NODELIST"
echo "SLURM node count: $SLURM_JOB_NUM_NODES"
echo
echo "Current time: $(date)"
echo "Current directory: $(pwd)"
echo "Current node: $(hostname)"
echo
echo "Executing script $TASK_NAME.py"
echo "========================================================================"
echo

# Dataset dependent arguments
case "$MOLECULE_DATASET" in
  "qm9")
    RUN_ARGS_DATASET="\
        data.datamodule.datasets.mp20.proportion=0.0 \
        data.datamodule.datasets.qm9.proportion=1.0 \
        data.datamodule.datasets.geom.proportion=0.0 \
        callbacks.model_checkpoint.monitor=\"val_qm9/valid_rate\" \
    "
    ;;
  "geom")
    RUN_ARGS_DATASET="\
        data.datamodule.datasets.mp20.proportion=0.0 \
        data.datamodule.datasets.qm9.proportion=0.0 \
        data.datamodule.datasets.geom.proportion=1.0 \
        callbacks.model_checkpoint.monitor=\"val_geom/valid_rate\" \
        data.datamodule.batch_size.train=16 \
        data.datamodule.batch_size.val=16 \
        data.datamodule.batch_size.test=16 \
        trainer.check_val_every_n_epoch=5 \
    "
    ;;
  *)
    echo "Unknown molecule dataset: $MOLECULE_DATASET, should be {qm9|geom}" >&2
    exit
    ;;
esac

bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 python zatom/$TASK_NAME.py \
    ckpt_path=$CKPT_PATH \
    data=$DATASET \
    date=$RUN_DATE \
    experiment=$EXPERIMENT \
    model=$MODEL \
    model/architecture=$ARCHITECTURE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    $RUN_ARGS_DATASET \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
"

echo "Run completed for job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
