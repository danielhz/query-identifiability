#!/usr/bin/env bash
#SBATCH --job-name=qident_e1
#SBATCH --partition=gpu_4              # Horeka GPU partition (4 GPUs/node)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --array=0-26                   # 3 n_tuples × 3 n_train × 3 seeds = 27
#SBATCH --output=logs/e1_%A_%a.out
#SBATCH --error=logs/e1_%A_%a.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --account=YOUR_ACCOUNT       # uncomment and fill in

# ---------------------------------------------------------------------------
# Parameter grid: 27 tasks  (n_tuples × n_train × seed)
# ---------------------------------------------------------------------------
N_TUPLES_LIST=(10 30 50)
N_TRAIN_LIST=(1000 5000 50000)
SEEDS=(0 1 2)

TASK=${SLURM_ARRAY_TASK_ID}
N_TUPLES=${N_TUPLES_LIST[$((TASK / 9))]}
N_TRAIN=${N_TRAIN_LIST[$(((TASK / 3) % 3))]}
SEED=${SEEDS[$((TASK % 3))]}

# Derived settings
N_VAL=$((N_TRAIN / 5))
N_TEST=$((N_TRAIN / 5))

echo "Task ${TASK}: n_tuples=${N_TUPLES}  n_train=${N_TRAIN}  seed=${SEED}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load devel/cuda/12.1
module load devel/python/3.11.7
# ── AISA: replace the two lines above with the appropriate module loads

VENV="${HOME}/.venvs/qident"
source "${VENV}/bin/activate"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"

RESULTS_DIR="${WORK:-${HOME}/scratch}/experiments/results"
mkdir -p "${RESULTS_DIR}" logs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -m experiments.e1_error_floor \
    --n-train   "${N_TRAIN}"   \
    --n-val     "${N_VAL}"     \
    --n-test    "${N_TEST}"    \
    --n-tuples  "${N_TUPLES}"  \
    --n-cert    10             \
    --n-noncert 10             \
    --epochs    300            \
    --patience  30             \
    --hidden-dim 128           \
    --seeds     "${SEED}"      \
    --device    cuda           \
    --output-dir "${RESULTS_DIR}"
