#!/usr/bin/env bash
# Submit only the ablation experiment to SLURM.
#
# Usage (from the repo root on the cluster):
#   bash cluster/submit_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

RESULTS_DIR="${WORK:-${HOME}/scratch}/experiments/results"
mkdir -p "${RESULTS_DIR}" logs

JID=$(sbatch --parsable "${SCRIPT_DIR}/horeka_ablation.sbatch")
echo "Ablation (FD completeness): job ${JID}  (3-task array)"
echo ""
echo "Results will appear in: ${RESULTS_DIR}"
echo "Analyze with:"
echo "  python -m analysis.plot_ablation --results-dir ${RESULTS_DIR} --latex"
