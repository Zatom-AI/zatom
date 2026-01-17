#!/bin/bash -l

# # Use to install torch_scatter from source on Perlmutter (one-time setup)
# export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
# export PATH="$CUDA_HOME/bin:$PATH"
# export CPATH="$CUDA_HOME/include:${CPATH:-}"
# export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Activate conda environment (locally created in Perlmutter home directory)
# shellcheck source=/dev/null
source /global/homes/a/acmwhb/miniforge3/etc/profile.d/conda.sh
conda activate zatom

# Determine location of the project's directory
# PROJECT_ID="dasrepo"
# PROJECT_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/Repositories/zatom"            # long term storage community drive
PROJECT_DIR="/pscratch/sd/${USER:0:1}/$USER/Repositories/zatom"                   # high-performance storage scratch drive with an 8-week purge policy
cd "$PROJECT_DIR" || exit

# Establish environment variables
# export TORCH_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/torch_cache"             # long term storage community drive
# export HF_HOME="/global/cfs/cdirs/$PROJECT_ID/$USER/hf_cache"                   # long term storage community drive
export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache"                    # high-performance storage scratch drive with an 8-week purge policy
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"                          # high-performance storage scratch drive with an 8-week purge policy

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

# Define run details
export PROJECT_ROOT="$PROJECT_DIR/forks/flowmm"
export PMG_VASP_PSP_DIR="$PROJECT_ROOT/data/vasp_psp"

# Change as needed
# mp20_only_zatom_1_eval_dir="$PROJECT_DIR/logs/eval_fm/runs/eval_tft_80M_MP20-only_hxip8xly_2025-12-21_17-00-00"
# jointly_trained_platom_1_eval_dir="$PROJECT_DIR/logs/eval_fm/runs/eval_tfp_80M_QM9+MP20_odrl69x1_2026-01-01_16-30-00"
eval_dir="$PROJECT_DIR/logs/eval_fm/runs/eval_tft_80M_MP20_izr5qhhf_2025-12-16_20-00-00"
# jointly_trained_zatom_1_l_eval_dir="$PROJECT_DIR/logs/eval_fm/runs/eval_tft_160M_QM9+MP20_zkiysa4a_2025-12-24_06-00-00"
# jointly_trained_zatom_1_xl_eval_dir="$PROJECT_DIR/logs/eval_fm/runs/eval_tft_300M_QM9+MP20_i1upnjuo_2025-12-24_06-00-00"

eval_for_dft_samples="$eval_dir/mp20_test_0"
eval_for_dft_json="$eval_dir/mp20_test_0.json"
eval_log_dir="$eval_dir/chgnet_log_dir"

num_jobs=50
slurm_qos=shared
slurm_account=m5008
slurm_partition=nersc
slurm_additional_parameters='{"constraint": "gpu&hbm40g", "module": "gpu,nccl-plugin"}'

# Consolidate
eval_for_dft_pt=$(python "$PROJECT_DIR/forks/flowmm/scripts_model/evaluate.py" consolidate "$eval_for_dft_samples" --subdir "mp20_test_0" --path_eval_pt eval_for_dft.pt | tail -n 1)

# Pre-relax
unset NCCL_CROSS_NIC
TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME \
python "$PROJECT_DIR/forks/flowmm/scripts_analysis/prerelax.py" \
    "$eval_for_dft_pt" "$eval_for_dft_json" "$eval_log_dir" \
    --num_jobs "$num_jobs" \
    --timeout_min 120 \
    --slurm_qos "$slurm_qos" \
    --slurm_account "$slurm_account" \
    --slurm_additional_parameters "$slurm_additional_parameters" \
    --slurm_partition "$slurm_partition"

# # DFT (ignore error if VASP license is not available)
# dft_dir="$eval_dir/dft"
# mkdir -p "$dft_dir"
# python forks/flowmm/scripts_analysis/dft_create_inputs.py "$eval_for_dft_json" "$dft_dir"

# # Energy above hull (if VASP license is not available)
# # (Option 1: Using less accurate CHGNet prerelaxed energies)
# clean_outputs_dir="$eval_dir/clean_outputs"
# json_e_above_hull="$eval_dir/ehulls.json"
# python forks/flowmm/scripts_analysis/ehull.py "$eval_for_dft_json" "$json_e_above_hull"

# # Energy above hull (if VASP license is available)
# # (Option 2: Using more accurate DFT energies computed above using your VASP license, where `clean_outputs` contains trajectory files with name ######.traj with ###### corresponding to the index of the corresponding row in the `eval_for_dft_json` file)
# clean_outputs_dir="$eval_dir/clean_outputs"
# json_e_above_hull="$eval_dir/ehulls.json"
# python forks/flowmm/scripts_analysis/ehull.py "$eval_for_dft_json" "$json_e_above_hull" --clean_outputs_dir "${clean_outputs_dir}"

# # Corrected energy above hull (if VASP license is available)
# root_dft_clean_outputs="$eval_dir"
# ehulls_corrected_json="$eval_dir/ehulls_corrected.json"
# python forks/flowmm/scripts_analysis/ehull_correction.py "$eval_for_dft_json" "$ehulls_corrected_json" --root_dft_clean_outputs "$root_dft_clean_outputs"

# # S.U.N. and M.S.U.N (if VASP license is not available)
# sun_json="$eval_dir/sun.json"
# msun_json="$eval_dir/msun.json"
# python forks/flowmm/scripts_analysis/novelty.py "$eval_for_dft_json" "$sun_json" --ehulls "$json_e_above_hull" --e_above_hull_column e_above_hull_per_atom_chgnet
# python forks/flowmm/scripts_analysis/novelty.py "$eval_for_dft_json" "$msun_json" --ehulls "$json_e_above_hull" --json_sun_count msun_count.json --e_above_hull_column e_above_hull_per_atom_chgnet --e_above_hull_maximum 0.1

# # S.U.N. and M.S.U.N (if VASP license is available)
# sun_json="$eval_dir/sun.json"
# msun_json="$eval_dir/msun.json"
# python forks/flowmm/scripts_analysis/novelty.py "$eval_for_dft_json" "$sun_json" --ehulls "$ehulls_corrected_json"
# python forks/flowmm/scripts_analysis/novelty.py "$eval_for_dft_json" "$msun_json" --ehulls "$ehulls_corrected_json" --json_sun_count msun_count.json --e_above_hull_maximum 0.1
