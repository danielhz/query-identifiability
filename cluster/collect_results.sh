#!/usr/bin/env bash
# Sync experiment results from the cluster to the local machine.
#
# Usage (run locally, not on the cluster):
#   bash cluster/collect_results.sh [horeka|aisa]
#
# Requires ssh key-based access to the cluster.
# Sets up ssh config entries like:
#   Host horeka
#       HostName hk.scc.kit.edu
#       User YOUR_USERNAME
#
# Output lands in results/ relative to the repo root.

set -euo pipefail

CLUSTER="${1:-horeka}"

case "${CLUSTER}" in
    horeka)
        REMOTE_HOST="horeka"
        REMOTE_PATH="\${WORK}/experiments/results/"
        ;;
    aisa)
        REMOTE_HOST="aisa"
        REMOTE_PATH="~/experiments/results/"
        ;;
    *)
        echo "Unknown cluster: ${CLUSTER}.  Usage: $0 [horeka|aisa]" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="${SCRIPT_DIR}/../results"
mkdir -p "${LOCAL_RESULTS}"

echo "Syncing ${REMOTE_HOST}:${REMOTE_PATH} → ${LOCAL_RESULTS}/"
rsync -avz --progress \
    "${REMOTE_HOST}:${REMOTE_PATH}" \
    "${LOCAL_RESULTS}/"

echo ""
echo "JSON files now in ${LOCAL_RESULTS}/"
echo "Run analysis with:"
echo "  python -m analysis.plot_error_floor"
echo "  python -m analysis.plot_capability_jump"
echo "  python -m analysis.plot_minaug"
