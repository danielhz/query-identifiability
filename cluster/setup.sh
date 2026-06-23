#!/usr/bin/env bash
# Set up the Python virtual environment on the cluster.
# Run once after cloning or pulling the repo:
#
#   bash cluster/setup.sh            # default venv path
#   bash cluster/setup.sh /path/venv # custom venv path
#
# The script is written for Horeka (KIT).  AISA differences are noted inline.

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster-specific module loads
# ---------------------------------------------------------------------------

# ── Horeka (KIT) ────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge
    module load devel/cuda/12.1           # A100, CUDA 12.1
    module load devel/python/3.11.7       # CPython 3.11
    # ── AISA (Stuttgart) — replace the two lines above with:
    # module load ...                     # check with: module avail
fi

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

VENV="${1:-${HOME}/.venvs/qident}"
echo "▶ Creating venv at ${VENV}"
python3 -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

pip install --upgrade pip --quiet

# PyTorch: CUDA 12.1 wheel.  Change cu121 → cu118 if your cluster runs CUDA 11.8.
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

# Remaining dependencies (no torch-geometric needed — GNN is hand-rolled)
pip install -r "$(dirname "$0")/../requirements.txt" --quiet

echo ""
echo "✓ Setup complete.  To activate:"
echo "    source ${VENV}/bin/activate"
echo ""
echo "  Quick smoke-test (CPU, ~30 s):"
echo "    python -m pytest tests/ -q"
