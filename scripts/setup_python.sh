#!/usr/bin/env bash
# Create .venv with the pinned encode/decode deps (uv sync).
# AC encode/decode needs a CUDA PyTorch build that matches the driver.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if command -v uv >/dev/null 2>&1; then
  echo "uv sync --extra dev  (pyproject.toml + uv.lock)"
  uv sync --extra dev
  PY="${ROOT}/.venv/bin/python"
else
  echo "uv not found; falling back to python -m venv + pip"
  PY="${PY:-python3}"
  VENV="${ROOT}/.venv"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    "${PY}" -m venv "${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install -U pip
  python -m pip install -r "${ROOT}/requirements.txt"
  python -m pip install pytest
  PY="${VENV}/bin/python"
fi

"${PY}" - <<'EOF'
import sys
print(f"python {sys.version.split()[0]}")
for name in ("numpy", "safetensors", "tqdm", "torch"):
    try:
        mod = __import__(name)
        extra = ""
        if name == "torch":
            extra = f"  cuda={mod.cuda.is_available()}"
        print(f"{name:12} {getattr(mod, '__version__', '?')}{extra}")
    except ImportError:
        print(f"{name:12} MISSING")
try:
    import triton
    print(f"{'triton':12} {getattr(triton, '__version__', '?')}")
except ImportError:
    print(f"{'triton':12} (not on this platform; required on Linux/CUDA)")
EOF
echo "OK ${ROOT}/.venv"
