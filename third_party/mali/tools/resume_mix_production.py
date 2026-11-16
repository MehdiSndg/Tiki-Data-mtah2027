"""Adopt one surviving production worker; finish eight folds, defer replay, then screen short-team CE.

The old controller must be stopped by PID only, leaving its active worker alive.
Training always uses the original pinned production snapshot; the paired screen uses a new pinned snapshot.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import run_mix_promotion as promotion

ROOT = Path(__file__).resolve().parents[1]


def process_handle(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return kernel, None
        raise ctypes.WinError(error)
    return kernel, handle


def assert_stopped(pid):
    kernel, handle = process_handle(pid)
    if handle:
        try:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                raise RuntimeError("Original controller is still active; do not start another controller")
        finally:
            kernel.CloseHandle(handle)


def validate_adopted_worker(pid, code):
    query = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object CommandLine | ConvertTo-Json -Compress"
    result = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", query], text=True).strip()
    command = json.loads(result or "null")
    if not command or "--worker fold" not in command["CommandLine"] or str(code).lower() not in command["CommandLine"].lower():
        raise RuntimeError("Adopted PID is not a fold worker from the pinned production snapshot")


def defer_reproduction(job, source, state, sealed):
    seal = dict(sealed, reproduction="deferred_by_user")
    promotion.write_json(job / "production_seal.json", sealed)
    promotion.write_json(source / "outputs" / "main" / "official_seal.json", seal)
    state.update(status="production_complete", phase="production_complete", sha256=sealed["sha256"],
                 reproduction="deferred_by_user", production_finished=promotion.now())
    promotion.write_json(job / "state.json", state)
    (job / "RESULT.md").write_text(
        f"# Production complete\n\nRevision: {sealed['revision']}\n\nSHA256: {sealed['sha256']}\n\n"
        "Clean-clone replay deferred by the user; byte-identical reproduction has not been verified.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--worker-pid", required=True, type=int)
    args = parser.parse_args()
    if os.name != "nt" or Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project interpreter on Windows")
    job = Path(args.job_dir).resolve()
    if not job.is_relative_to((ROOT / "outputs").resolve()):
        raise RuntimeError("Job must be inside workspace outputs")
    state = json.loads((job / "state.json").read_text(encoding="utf-8"))
    if state["phase"] != "production" or state["status"] != "running":
        raise RuntimeError("Adoption is restricted to an active production phase")
    assert_stopped(state["pid"])
    code = job / "production_code"
    if promotion.git(code, "rev-parse", "HEAD") != state["revision"] or promotion.git(code, "status", "--porcelain"):
        raise RuntimeError("Production snapshot changed")
    validate_adopted_worker(args.worker_pid, code)
    kernel, worker_handle = process_handle(args.worker_pid)
    if not worker_handle:
        raise RuntimeError("Worker exited before adoption; inspect completion artifacts before resuming")
    lock = ROOT / "outputs/mix_promotion/running.lock"
    if lock.exists():
        assert_stopped(int(lock.read_text(encoding="utf-8")))
    lock.write_text(str(os.getpid()), encoding="utf-8")
    awake = False
    try:
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not awake:
            raise RuntimeError("Could not keep Windows awake")
        state.update(pid=os.getpid(), adopted_worker_pid=args.worker_pid, reproduction="deferred_by_user",
                     next_experiment="short_team_main_screen", updated=promotion.now())
        promotion.write_json(job / "state.json", state)
        revision = promotion.git(ROOT, "rev-parse", "HEAD")
        if promotion.git(ROOT, "status", "--porcelain"):
            raise RuntimeError("Commit experiment code before adoption")
        experiment_code = job / "experiments_code"
        promotion.clone_local(ROOT, experiment_code, revision)
        state["experiment_revision"] = revision
        promotion.write_json(job / "state.json", state)
        while True:
            wait = kernel.WaitForSingleObject(worker_handle, 1000)
            if wait == 0:
                break
            if wait != 258:
                raise RuntimeError("Waiting for adopted worker failed")
        name = state["current_fold"]
        marker = ROOT / "outputs" / name / "night_complete.json"
        log = job / f"production_{name}.log"
        if not marker.is_file() or "OFFLINE_WORKER_COMPLETE" not in log.read_text(encoding="utf-8"):
            raise RuntimeError("Adopted worker has no complete artifacts; do not overwrite its partial output")
        data = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "more-than-a-hack-2027/data").resolve()
        sealed = promotion.run_phase(code, data, ROOT, job, "production", state["revision"], state)
        defer_reproduction(job, ROOT, state, sealed)
        promotion.ensure_no_training()
        state.update(status="running", phase="short_team_main_screen", updated=promotion.now())
        promotion.write_json(job / "state.json", state)
        env = promotion.offline_env(experiment_code, data, ROOT)
        env["MTAH_PYTHON"] = sys.executable
        with (job / "short_team_screen.log").open("w", encoding="utf-8") as log:
            child = subprocess.run([sys.executable, "-u", str(experiment_code / "tools/run_short_team_screen.py"),
                                    "--main-backbone", "--wait-minutes", "30"], cwd=experiment_code,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        screen = ROOT / "outputs/short_team_main_screen"
        result = json.loads((screen / "state.json").read_text(encoding="utf-8"))
        if result["status"] != "complete" or not (screen / "comparison.json").is_file():
            raise RuntimeError(f"Paired screen did not complete (exit {child.returncode}); inspect local logs")
        state.update(status="complete", phase="short_team_screen_complete", finished=promotion.now())
        promotion.write_json(job / "state.json", state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), updated=promotion.now())
        promotion.write_json(job / "state.json", state)
        traceback.print_exc()
        raise
    finally:
        kernel.CloseHandle(worker_handle)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


if __name__ == "__main__":
    main()
