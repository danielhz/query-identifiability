#!/usr/bin/env bash
#SBATCH --job-name=qident_e3
#SBATCH --partition=cpuonly            # no GPU needed for MinAug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/e3_%j.out
#SBATCH --error=logs/e3_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --account=YOUR_ACCOUNT
#
# Note: E3 (MinAug) is CPU-only.  On AISA, change --partition accordingly.
# On Horeka, cpuonly is the name of the CPU-only partition.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load devel/python/3.11.7

VENV="${HOME}/.venvs/qident"
source "${VENV}/bin/activate"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"

RESULTS_DIR="${WORK:-${HOME}/scratch}/experiments/results"
mkdir -p "${RESULTS_DIR}" logs

# ---------------------------------------------------------------------------
# Run (single large sweep — fast even on CPU)
# ---------------------------------------------------------------------------
python -m experiments.e3_minaug \
    --n-trials       2000                    \
    --n-attrs-range  4 6 8 10 12 15          \
    --n-fds-range    0 2 4 6 10              \
    --footprint-frac 0.6                     \
    --seed           0                       \
    --output-dir     "${RESULTS_DIR}"
