#!/usr/bin/env bash
#SBATCH --job-name=qident_e3
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/e3_%j.out
#SBATCH --error=logs/e3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.hernandez@ki.uni-stuttgart.de

# E3 (Greedy-MinAug) is CPU-only — no GPU needed.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
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
python -m experiments.e3_minaug \
    --n-trials       2000                    \
    --n-attrs-range  4 6 8 10 12 15          \
    --n-fds-range    0 2 4 6 10              \
    --footprint-frac 0.6                     \
    --seed           0                       \
    --output-dir     "${RESULTS_DIR}"
