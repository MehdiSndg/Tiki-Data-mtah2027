"""Keep system and display awake until the existing local job controller exits.

No training, network calls, power-plan edits or mouse/key simulation. The execution request
is released when the controller ends (or this helper ends). Modern Standby sessions may
include mere screen-off time: their event IDs alone do not prove a suspended training job.
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    if os.name != "nt" or Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project interpreter on Windows")
    job = Path(args.job_dir).resolve()
    if not job.is_relative_to((ROOT / "outputs").resolve()):
        raise RuntimeError("Job must be inside workspace outputs")
    state = json.loads((job / "state.json").read_text(encoding="utf-8"))
    if state["status"] not in ("running", "production_complete"):
        raise RuntimeError("No active job to keep awake")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x00100000, False, int(state["pid"]))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    lock = job / "awake_guard.lock"
    active = False
    owned = False
    metadata = {"pid": os.getpid(), "controller_pid": state["pid"], "display_required": True,
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def record(status):
        (job / "awake_guard.json").write_text(json.dumps(dict(metadata, status=status), indent=2), encoding="utf-8")

    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        owned = True
        if kernel.WaitForSingleObject(handle, 0) != 258:
            raise RuntimeError("Controller is already stopped")
        active = bool(kernel.SetThreadExecutionState(0x80000003))
        if not active:
            raise ctypes.WinError(ctypes.get_last_error())
        record("active")
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise RuntimeError("Waiting for controller failed")
    finally:
        if active:
            kernel.SetThreadExecutionState(0x80000000)
            record("released")
        kernel.CloseHandle(handle)
        if owned:
            lock.unlink()


if __name__ == "__main__":
    main()
