#!/usr/bin/env python3
"""Compatibility facade for the project-owned keepalive worker.

Operational launches must go through ``motion_eval``'s keepalive controller so
the reservation/record/ready handshake exists before this worker touches CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from motion_eval.runtime.keepalive_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
