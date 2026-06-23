#!/usr/bin/env bash
# Submit all experiments to the AISA cluster.
# Run from the repo root: bash cluster/submit_aisa.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

mkdir -p logs results

echo "=== Submitting experiments to AISA ==="

JOB_E1=$(sbatch --parsable "${SCRIPT_DIR}/aisa_e1.sbatch")
echo "E1 (error floor)      → job array ${JOB_E1}"

JOB_E2=$(sbatch --parsable "${SCRIPT_DIR}/aisa_e2.sbatch")
echo "E2 (capability jump)  → job array ${JOB_E2}"

JOB_E3=$(sbatch --parsable "${SCRIPT_DIR}/aisa_e3.sbatch")
echo "E3 (MinAug)           → job       ${JOB_E3}"

JOB_E4=$(sbatch --parsable "${SCRIPT_DIR}/aisa_e4.sbatch")
echo "E4 (real-world)       → job array ${JOB_E4}"

JOB_ABL=$(sbatch --parsable "${SCRIPT_DIR}/aisa_ablation.sbatch")
echo "Ablation              → job array ${JOB_ABL}"

echo ""
echo "Monitor with: squeue -u $(whoami)"
echo "Results will appear in: ${REPO}/results/"
