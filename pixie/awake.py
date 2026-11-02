"""Keep the GPU machine from going to sleep while a computation runs.

This asks Windows not to sleep for as long as the calling process lives, the way a video player does.
Power settings are left alone: once the process exits, the normal idle timer applies again.
"""

from __future__ import annotations

import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
