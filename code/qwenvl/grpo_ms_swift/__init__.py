"""GRPO modules for Motion-r1 ms-swift integration.

The path shim exists only for legacy ``external_plugins`` file loading. New
code should install the package and import ``motionllm.grpo`` directly.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
