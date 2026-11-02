"""Keep the machine awake until process <pid> exits (for a job started without pixie.awake).

Usage: python scripts/keep_awake.py <pid>
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie.awake import keep_awake  # noqa: E402

SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF

if __name__ == "__main__":
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(SYNCHRONIZE, False, int(sys.argv[1]))
    if not h:
        sys.exit(f"no process {sys.argv[1]}")
    keep_awake()
    k32.WaitForSingleObject(h, INFINITE)
