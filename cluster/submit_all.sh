#!/usr/bin/env bash
# Submit all experiments to SLURM and print job IDs.
#
# Usage (from the repo root on the cluster):
#   bash cluster/submit_all.sh
#
# E3 (CPU-only) is submitted first with no dependency.
# E1 and E2 run independently on GPU; their analysis can start once all
# array tasks in the job finish.
#
# Monitor progress:
#   watch -n 30 squeue -u "$USER" --format="%.10i %.9P %.20j %.8T %.10M %.4C %R"
#
# Re-run a single failed array task (e.g. task 7 of job 12345):
#   sbatch --array=7 cluster/horeka_e1.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

RESULTS_DIR="${WORK:-${HOME}/scratch}/experiments/results"
mkdir -p "${RESULTS_DIR}" logs

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

JID_E3=$(sbatch --parsable "${SCRIPT_DIR}/horeka_e3.sbatch")
echo "E3 (MinAug, CPU):           job ${JID_E3}"

JID_E1=$(sbatch --parsable "${SCRIPT_DIR}/horeka_e1.sbatch")
echo "E1 (error floor, GPU):      job ${JID_E1}  (27-task array)"

JID_E2=$(sbatch --parsable "${SCRIPT_DIR}/horeka_e2.sbatch")
echo "E2 (capability jump, GPU):  job ${JID_E2}  (3-task array)"

JID_ABL=$(sbatch --parsable "${SCRIPT_DIR}/horeka_ablation.sbatch")
echo "Ablation (FD completeness): job ${JID_ABL}  (3-task array)"

JID_E4=$(sbatch --parsable "${SCRIPT_DIR}/horeka_e4.sbatch")
echo "E4 (real-world, GPU):       job ${JID_E4}  (3-task array)"

echo ""
echo "All jobs submitted.  Results will appear in:"
echo "  ${RESULTS_DIR}"
echo ""
echo "Run analysis after all tasks complete:"
echo "  python -m analysis.plot_error_floor     --results-dir ${RESULTS_DIR}"
echo "  python -m analysis.plot_capability_jump --results-dir ${RESULTS_DIR}"
echo "  python -m analysis.plot_minaug          --results-dir ${RESULTS_DIR}"
echo "  python -m analysis.plot_ablation        --results-dir ${RESULTS_DIR} --latex"
echo "  python -m analysis.plot_realworld       --results-dir ${RESULTS_DIR} --latex"
