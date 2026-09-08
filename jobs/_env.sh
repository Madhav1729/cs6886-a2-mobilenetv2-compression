# Shared setup, sourced by every PBS job. No `set -e` on purpose -- jobs check
# exit codes themselves so a failed step still prints a summary.

set -uo pipefail

mkdir -p logs checkpoints

# --- AQUA cluster setup -------------------------------------------------
module purge 2>/dev/null || true
module load cuda12.4 2>/dev/null || true

# --- python environment -------------------------------------------------
# AQUA defaults (override via qsub -v CONDA_ENV=...,CONDA_BASE=...)
: "${CONDA_BASE:=/lfs/usrhome/btech/cs23b035/miniconda3}"
: "${CONDA_ENV:=cs6886_a2}"

if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif [ -n "${VENV_PATH:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

# --- fail fast, with a clear message ------------------------------------
echo "[pbs] host      : $(hostname)"
echo "[pbs] workdir   : $PWD"
echo "[pbs] python    : $($PYTHON -V 2>&1)  ($(command -v "$PYTHON" || echo NOT-FOUND))"
if ! "$PYTHON" - <<'PYCHK'
import sys
if sys.version_info < (3, 9):
    sys.exit(f"[pbs] FATAL: need Python 3.9+, got {sys.version.split()[0]}")
try:
    import torch, torchvision
except Exception as e:
    sys.exit(f"[pbs] FATAL: cannot import torch/torchvision: {e}")
print(f"[pbs] torch     : {torch.__version__} | torchvision {torchvision.__version__}")
if torch.cuda.is_available():
    print(f"[pbs] cuda      : yes | {torch.cuda.get_device_name(0)}")
else:
    print("[pbs] cuda      : NO -- running on CPU, this will be very slow")
PYCHK
then
    echo "[pbs] environment check FAILED -- aborting before wasting the allocation"
    exit 1
fi
echo "[pbs] started   : $(date)"
echo

run_step() {
    local label="$1"; shift
    echo "########## ${label} ##########"
    local t0 status
    t0=$(date +%s)
    "$@"
    status=$?
    echo "[pbs] ${label}: exit=${status}  elapsed=$(( $(date +%s) - t0 ))s"
    echo
    return $status
}
