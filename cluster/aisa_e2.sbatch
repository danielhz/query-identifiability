#!/usr/bin/env bash
#SBATCH --job-name=qident_e2
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=A40:1
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --array=0-2                    # one task per seed
#SBATCH --output=logs/e2_%A_%a.out
#SBATCH --error=logs/e2_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.hernandez@ki.uni-stuttgart.de

# ---------------------------------------------------------------------------
# Parameter grid: 3 tasks  (one per seed)
# All 5 augmentation steps and all 6 architectures run within each task.
# ---------------------------------------------------------------------------
SEEDS=(0 1 2)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

REPO="${QIDENT_REPO:-$HOME/query-identifiability}"
VENV="${REPO}/.venv"
source "${VENV}/bin/activate"
cd "${REPO}"

RESULTS_DIR="${REPO}/results"
mkdir -p "${RESULTS_DIR}" logs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -m experiments.e2_capability_jump \
    --n-train   5000     \
    --n-val     500      \
    --n-test    1000     \
    --n-tuples  5        \
    --epochs    300      \
    --patience  30       \
    --hidden-dim 128     \
    --seeds     "${SEED}" \
    --device    cuda     \
    --output-dir "${RESULTS_DIR}"
