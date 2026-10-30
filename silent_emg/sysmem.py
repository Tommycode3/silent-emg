"""Wait for free memory before starting a training run (macOS `memory_pressure`)."""

from __future__ import annotations

import re
import shutil
import subprocess
import time


def free_percent() -> float | None:
    if shutil.which("memory_pressure") is None:
        return None
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def wait_for_memory(min_free: float = 25.0, poll_s: float = 30.0, log=print) -> None:
    while True:
        f = free_percent()
        if f is None or f >= min_free:
            return
        log(f"[sysmem] free memory {f:.0f}% < {min_free:.0f}%, waiting {poll_s:.0f}s")
        time.sleep(poll_s)
