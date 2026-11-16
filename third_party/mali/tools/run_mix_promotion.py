r"""Detached, offline promotion: eight production folds, seal, then a fresh-clone replay.

Use .venv\Scripts\python.exe tools/run_mix_promotion.py --job-dir outputs/mix_promotion/20261027_night
The controller and all workers run locally; the controller keeps Windows awake while active.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

CODE_ROOT = Path(__file__).resolve().parents[1]
MATCHES = ("J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    # All allowed Git transport is local, including both official-repository copies.
    return subprocess.check_output(["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always",
                                    "-C", str(root), *args], text=True, encoding="utf-8").strip()


def clone_local(source, dest, revision):
    official_rev = git(source, "ls-tree", revision, "more-than-a-hack-2027").split()[2]
    if dest.exists():
        if git(dest, "rev-parse", "HEAD") != revision:
            raise RuntimeError("Existing snapshot has a different revision")
        if git(dest / "more-than-a-hack-2027", "rev-parse", "HEAD") != official_rev:
            raise RuntimeError("Official dependency snapshot has a different revision")
        return
    subprocess.run(["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "clone",
                    "--no-hardlinks", "--no-checkout", str(source), str(dest)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    git(dest, "checkout", "--detach", revision)
    official = dest / "more-than-a-hack-2027"
    subprocess.run(["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "clone",
                    "--no-hardlinks", "--no-checkout", str(source / "more-than-a-hack-2027"), str(official)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    git(official, "checkout", "--detach", official_rev)


def offline_env(code_root, data_root, storage=None):
    env = dict(os.environ)
    for key in ("MTAH_CACHE_DIR", "MTAH_OUTPUTS_DIR", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(PYTHONIOENCODING="utf-8", MTAH_DATA_DIR=str(data_root), MTAH_OFFLINE="1",
               PYTHONPATH=str(code_root / "tools" / "offline_site"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", WANDB_MODE="disabled")
    if storage is not None:
        env.update(MTAH_CACHE_DIR=str(storage / "cache"), MTAH_OUTPUTS_DIR=str(storage / "outputs"))
    return env


def preflight(root, data_root):
    """Inspect only paths/sizes and installed versions; no match data is printed or sampled."""
    required = [data_root / "tactical" / f"BAR-ATH_tactical{s}"
                for s in ("_atd.xml", "_atd.txt", "_smart_tagging.xml")]
    required += [data_root / "external" / "idsse" / m / f for m in MATCHES
                 for f in ("meta.xml", "positions.xml")]
    for game in (1, 2):
        base = data_root / "external" / "metrica" / "data" / f"Sample_Game_{game}"
        required += [base / f"Sample_Game_{game}_{suffix}.csv" for suffix in
                     ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")]
    base = data_root / "external" / "metrica" / "data" / "Sample_Game_3"
    required += [base / f"Sample_Game_3_{s}" for s in ("metadata.xml", "tracking.txt", "events.json")]
    missing = [str(p.relative_to(data_root)) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Offline raw files missing: " + ", ".join(missing))
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scipy", "pyarrow", "lightgbm", "PyYAML", "numba", "torch", "kloppy")}
    if shutil.disk_usage(root).free < 30 * (1 << 30):
        raise RuntimeError("Less than 30 GiB of free disk space")
    return {"raw_files_present": len(required), "packages": versions,
            "free_disk_gib": round(shutil.disk_usage(root).free / (1 << 30), 1)}


def ensure_no_training():
    if os.name == "nt":
        query = (r"@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' "
                 r"-and $_.CommandLine -notmatch 'run_mix_promotion\.py' "
                 r"-and $_.CommandLine -match 'run\.py|run_short_team_screen\.py|seqnet|train' }).Count")
        count = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", query], text=True).strip()
        if int(count) > 0:
            raise RuntimeError("Another training process is present; promotion did not start a new fit")
    names = subprocess.check_output(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                                    text=True)
    if any("python" in s.lower() for s in names.splitlines()):
        raise RuntimeError("Another Python GPU process is present")


def output_summary(output_root, name):
    folder = output_root / name
    for f in ("summary.json", "submission.csv", "probs.npy", "bar_bnd_adapted.npy"):
        if not (folder / f).is_file():
            raise RuntimeError(f"Incomplete output: {name}/{f}")
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


def worker(args):
    import socket
    if os.environ.get("MTAH_OFFLINE") != "1" or socket.socket.connect.__module__ != "sitecustomize":
        raise RuntimeError("The offline network guard must be active before any training imports")
    sys.path.insert(0, str(CODE_ROOT / "src"))
    from mtah import config
    from mtah.paths import OUTPUTS
    from mtah.pipeline import regionrun
    from mtah.io import submission
    plan = [pair for row in regionrun.fold_configs("main") for pair in row]
    revision = git(CODE_ROOT, "rev-parse", "HEAD")
    if args.worker == "fold":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing an unattended CPU training fallback")
        name, cfg = plan[args.index]
        regionrun.train_fold(name, cfg, bootstrap=0)
        output_summary(OUTPUTS, name)
        if submission.validate(OUTPUTS / name / "submission.csv") is not None:
            raise RuntimeError("Fold submission failed the official format check")
        write_json(OUTPUTS / name / "night_complete.json",
                   {"revision": revision, "config_hash": config.digest(cfg), "finished": now(),
                    "sha256": sha(OUTPUTS / name / "submission.csv")})
    else:
        cfg = config.load("main")
        sets = regionrun.fold_configs("main")
        _, summary = regionrun.stitch(cfg, [[n for n, _ in row] for row in sets], "main")
        if summary["official_format_check"] != "OK":
            raise RuntimeError("Final submission failed the official format check")
        write_json(OUTPUTS / "main" / "night_complete.json",
                   {"revision": revision, "config_hash": config.digest(cfg), "finished": now(),
                    "sha256": summary["sha256"]})
    print("OFFLINE_WORKER_COMPLETE", flush=True)


def archive_previous(storage, job):
    backup = job / "previous_official"
    backup.mkdir(parents=True, exist_ok=True)
    outputs = (storage / "outputs").resolve()
    # Keep main available as a diagnostic reference until final stitching replaces it.
    if (outputs / "main").is_dir() and not (backup / "main").exists():
        shutil.copytree(outputs / "main", backup / "main")
    names = ["main_lgbm"]
    for seed in (2027, 7027):
        for k in range(4):
            name = f"main_rf{k}" + ("" if seed == 2027 else f"_s{seed}")
            names.extend((name, name + "_lgbm", name + "_aux_lgbm"))
    for name in names:
        source, dest = (outputs / name).resolve(), (backup / name).resolve()
        if not source.is_relative_to(outputs) or not dest.is_relative_to(job.resolve()):
            raise RuntimeError("Archive path left the intended workspace")
        if source.exists() and not dest.exists():
            source.rename(dest)


def run_phase(code, data, storage, job, phase, revision, state):
    out = (storage or code) / "outputs"
    env = offline_env(code, data, storage)
    for index in range(8):
        marker = out / (f"main_rf{index % 4}" + ("_s7027" if index >= 4 else "")) / "night_complete.json"
        name = marker.parent.name
        if marker.exists():
            done = json.loads(marker.read_text(encoding="utf-8"))
            if done["revision"] == revision:
                output_summary(out, name)
                state["completed_folds"].setdefault(phase, {})[name] = done
                continue
            raise RuntimeError("Existing completion marker belongs to another revision")
        ensure_no_training()
        state.update(phase=phase, current_fold=name, updated=now())
        write_json(job / "state.json", state)
        log_path = job / f"{phase}_{name}.log"
        cmd = [sys.executable, "-u", str(code / "tools" / "run_mix_promotion.py"), "--worker", "fold",
               "--index", str(index)]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        if not marker.exists() or "OFFLINE_WORKER_COMPLETE" not in log_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"{phase}/{name} did not complete; inspect its local log")
        done = json.loads(marker.read_text(encoding="utf-8"))
        if done["revision"] != revision:
            raise RuntimeError("Worker revision mismatch")
        output_summary(out, name)
        state["completed_folds"].setdefault(phase, {})[name] = dict(done, exit_code=result.returncode)
        write_json(job / "state.json", state)
    state.update(phase=phase + "_stitch", updated=now())
    write_json(job / "state.json", state)
    log_path = job / f"{phase}_stitch.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", str(code / "tools" / "run_mix_promotion.py"),
                                 "--worker", "stitch"], cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
    if "OFFLINE_WORKER_COMPLETE" not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"{phase} stitching did not complete")
    summary = output_summary(out, "main")
    actual = sha(out / "main" / "submission.csv")
    if summary["sha256"] != actual or summary["official_format_check"] != "OK":
        raise RuntimeError("Sealed SHA or format check mismatch")
    return {"sha256": actual, "revision": revision, "finished": now(), "exit_code": result.returncode,
            **{k: summary[k] for k in ("n_possessions", "dead_share", "home_share", "official_format_check")}}


def controller(args):
    source = CODE_ROOT
    if Path(sys.executable).resolve() != (source / ".venv" / "Scripts" / "python.exe").resolve():
        raise RuntimeError("Use the project's .venv interpreter")
    data = Path(os.environ.get("MTAH_DATA_DIR") or source / "more-than-a-hack-2027" / "data").resolve()
    checks = preflight(source, data)
    if args.preflight:
        print(json.dumps(checks), flush=True)
        return
    ensure_no_training()
    if git(source, "status", "--porcelain"):
        raise RuntimeError("Commit the source changes before starting the pinned overnight run")
    revision = git(source, "rev-parse", "HEAD")
    job = Path(args.job_dir).resolve()
    if not job.is_relative_to((source / "outputs").resolve()):
        raise RuntimeError("Job directory must be inside this workspace's outputs directory")
    job.mkdir(parents=True, exist_ok=True)
    lock = source / "outputs" / "mix_promotion" / "running.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state_path = job / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "status": "running", "started": now(), "revision": revision, "completed_folds": {}, "offline": True}
    awake = False
    try:
        if state["revision"] != revision:
            raise RuntimeError("Resume requires the original source revision")
        state.update(status="running", preflight=checks, pid=os.getpid(), updated=now())
        write_json(state_path, state)
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            if not awake:
                raise RuntimeError("Could not request Windows to stay awake")
        production, clean = job / "production_code", job / "clean_clone"
        clone_local(source, production, revision)
        clone_local(source, clean, revision)
        preflight(clean, data)
        if not state["completed_folds"].get("production"):
            archive_previous(source, job)
        sealed = run_phase(production, data, source, job, "production", revision, state)
        write_json(job / "production_seal.json", sealed)
        write_json(source / "outputs" / "main" / "official_seal.json", dict(sealed, reproduction="pending"))
        if not state.get("clean_start_verified"):
            if (clean / "cache").exists() or (clean / "outputs").exists():
                raise RuntimeError("The clean-clone replay must start without caches or outputs")
        state["clean_start_verified"] = True
        write_json(state_path, state)
        replay = run_phase(clean, data, None, job, "reproduction", revision, state)
        write_json(job / "reproduction_seal.json", replay)
        if replay["sha256"] != sealed["sha256"]:
            raise RuntimeError("Clean clone did not reproduce the production SHA")
        verified = dict(sealed, reproduction="byte_identical", reproduction_finished=replay["finished"],
                        clean_clone=str(clean))
        write_json(source / "outputs" / "main" / "official_seal.json", verified)
        state.update(status="complete", phase="verified", sha256=sealed["sha256"], finished=now())
        write_json(state_path, state)
        (job / "RESULT.md").write_text(
            f"# Offline promotion complete\n\nRevision: {revision}\n\nSHA256: {sealed['sha256']}\n\n"
            f"Clean clone: byte-identical.\n\nPossessions: {sealed['n_possessions']}; D: {sealed['dead_share']:.6f}.\n",
            encoding="utf-8")
        print(json.dumps(verified), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), updated=now())
        write_json(state_path, state)
        traceback.print_exc()
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", default="outputs/mix_promotion/20261027_night")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", choices=("fold", "stitch"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    main()
