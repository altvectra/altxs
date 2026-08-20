from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "model"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
