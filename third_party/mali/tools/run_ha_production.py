r"""Resumable, offline, day-split production of the H<->A candidate (configs/main_ha*.yaml).

Eight region folds (4 regions x seed sets 2027/7027) -> outputs/main_ha_rf*, then stitch -> outputs/main_ha.
The official outputs/main is never touched; promotion is a separate, user-approved step.
A fold starts only if its declared maximum runtime plus five minutes fits before 22:55 Istanbul;
re-run the same command on a later day to continue (finished folds are skipped by revision marker).

  .venv\Scripts\python.exe tools/run_ha_production.py --job-dir outputs/ha_production

--clean: fresh-clone replay of the sealed official file. The snapshot is pinned to the seal's revision and
uses its own empty cache/ and outputs/ inside <job>/code (raw data only is shared, read-only); the final
SHA is compared with outputs/main/official_seal.json. The official outputs are never written.
"""

import argparse
import ctypes
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import run_mix_promotion as common
from run_day_screen import ISTANBUL, budget_seconds, fit

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "main_ha"
MARKER = "ha_complete.json"


def plan():
    sys.path.insert(0, str(CODE_ROOT / "src"))
    from mtah.pipeline import regionrun
    return regionrun.fold_configs(CONFIG)


def worker(args):
    import socket
    if os.environ.get("MTAH_OFFLINE") != "1" or socket.socket.connect.__module__ != "sitecustomize":
        raise RuntimeError("The offline network guard must be active before any training imports")
    sets = plan()
    from mtah import config
    from mtah.io import submission
    from mtah.paths import OUTPUTS
    from mtah.pipeline import regionrun
    revision = common.git(CODE_ROOT, "rev-parse", "HEAD")
    if args.worker == "fold":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing an unattended CPU training fallback")
        name, cfg = [pair for row in sets for pair in row][args.index]
        regionrun.train_fold(name, cfg, bootstrap=0)
        common.output_summary(OUTPUTS, name)
        if submission.validate(OUTPUTS / name / "submission.csv") is not None:
            raise RuntimeError("Fold submission failed the official format check")
        common.write_json(OUTPUTS / name / MARKER, {"revision": revision, "config_hash": config.digest(cfg),
                                                     "finished": common.now(),
                                                     "sha256": common.sha(OUTPUTS / name / "submission.csv")})
    else:
        cfg = config.load(CONFIG)
        _, summary = regionrun.stitch(cfg, [[n for n, _ in row] for row in sets], CONFIG)
        if summary["official_format_check"] != "OK":
            raise RuntimeError("Final submission failed the official format check")
        common.write_json(OUTPUTS / CONFIG / MARKER, {"revision": revision, "config_hash": config.digest(cfg),
                                                       "finished": common.now(), "sha256": summary["sha256"]})
    print("OFFLINE_WORKER_COMPLETE", flush=True)


def controller(args):
    if os.name != "nt" or Path(sys.executable).resolve() != (CODE_ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project interpreter on Windows")
    if not 0 < args.max_fold_minutes <= 180:
        raise ValueError("Fold runtime must be between zero and 180 minutes")
    hour, minute = (int(x) for x in args.cutoff.split(":"))
    now = datetime.now(ISTANBUL)
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if hour < 12:                    # an after-midnight cutoff (user-approved exception) belongs to the next day
        cutoff += timedelta(days=1)
    if not timedelta(0) < cutoff - now <= timedelta(hours=16) or (hour, minute) > (22, 55) and hour >= 12:
        raise ValueError("Cutoff must be 22:55 or earlier, or an explicit after-midnight exception (HH:MM < 12:00)")
    if not budget_seconds(datetime.now(ISTANBUL), cutoff, args.max_fold_minutes * 60):
        raise RuntimeError("No new fold: not enough time before the 23:00 boundary")
    job = Path(args.job_dir).resolve()
    outputs = (CODE_ROOT / "outputs").resolve()
    if not job.is_relative_to(outputs):
        raise RuntimeError("Job must be inside workspace outputs")
    state_path = job / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    seal = json.loads((outputs / "main" / "official_seal.json").read_text(encoding="utf-8")) if args.clean else None
    if state is None:
        if common.git(CODE_ROOT, "status", "--porcelain"):
            raise RuntimeError("Commit source changes before pinning the production revision")
        if args.clean and seal.get("config") != CONFIG:
            raise RuntimeError("The official seal is not a main_ha production")
        state = {"revision": seal["revision"] if args.clean else common.git(CODE_ROOT, "rev-parse", "HEAD"),
                 "config": CONFIG, "clean": bool(args.clean), "created": common.now(), "folds": {}, "sessions": []}
        if args.clean:
            state["expected_sha256"] = seal["sha256"]
    if bool(state.get("clean")) != bool(args.clean):
        raise RuntimeError("This job directory was created with a different --clean setting")
    revision = state["revision"]
    common.ensure_no_training()
    job.mkdir(parents=True, exist_ok=True)
    lock = outputs / "day_training.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    awake = False
    state.update(status="initializing", pid=os.getpid(), cutoff_istanbul=cutoff.isoformat(), updated=common.now())
    state["sessions"].append({"started": common.now()})
    try:
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000003))
        if not awake:
            raise RuntimeError("Could not request system wakefulness")
        code = job / "code"
        common.clone_local(CODE_ROOT, code, revision)
        data = Path(os.environ.get("MTAH_DATA_DIR") or CODE_ROOT / "more-than-a-hack-2027/data").resolve()
        state["preflight"] = common.preflight(CODE_ROOT, data)
        env = common.offline_env(code, data, None if args.clean else CODE_ROOT)
        out = code / "outputs" if args.clean else outputs   # clean: the snapshot's own empty storage
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                       env=env, check=True, timeout=90)
        names = [n for row in plan() for n, _ in row]
        state.update(status="running")
        common.write_json(state_path, state)
        for index, name in enumerate(names):
            marker = out / name / MARKER
            if marker.exists():
                done = json.loads(marker.read_text(encoding="utf-8"))
                if done["revision"] != revision:
                    raise RuntimeError(f"{name}: completion marker belongs to another revision")
                state["folds"][name] = done
                continue
            if (out / name).exists():      # a fold stopped mid-way: keep it for inspection, never reuse
                dest = job / "abandoned" / f"{name}_{datetime.now(ISTANBUL):%Y%m%d_%H%M%S}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(out / name), str(dest))
            seconds = budget_seconds(datetime.now(ISTANBUL), cutoff, args.max_fold_minutes * 60)
            if not seconds:
                state.update(status="deferred_time_budget", next_fold=name, updated=common.now())
                common.write_json(state_path, state)
                return
            common.ensure_no_training()
            state.update(current=name, updated=common.now())
            common.write_json(state_path, state)
            log = job / f"{name}.log"
            started = common.now()
            exit_code = fit([sys.executable, "-u", str(code / "tools/run_ha_production.py"), "--worker", "fold",
                             "--index", str(index)], code, env, log, seconds)
            if not marker.exists() or "OFFLINE_WORKER_COMPLETE" not in log.read_text(encoding="utf-8"):
                raise RuntimeError(f"{name} did not complete; inspect its local log")
            done = json.loads(marker.read_text(encoding="utf-8"))
            if done["revision"] != revision:
                raise RuntimeError("Worker revision mismatch")
            state["folds"][name] = dict(done, started=started, exit_code=exit_code)
            common.write_json(state_path, state)
        state.update(current="stitch", updated=common.now())
        common.write_json(state_path, state)
        log = job / "stitch.log"
        fit([sys.executable, "-u", str(code / "tools/run_ha_production.py"), "--worker", "stitch"],
            code, env, log, 900)
        if "OFFLINE_WORKER_COMPLETE" not in log.read_text(encoding="utf-8"):
            raise RuntimeError("Stitching did not complete")
        summary = common.output_summary(out, CONFIG)
        actual = common.sha(out / CONFIG / "submission.csv")
        if summary["sha256"] != actual or summary["official_format_check"] != "OK":
            raise RuntimeError("SHA or format check mismatch")
        state.update(status="complete", sha256=actual, finished=common.now(),
                     result={k: summary[k] for k in ("n_possessions", "dead_share", "home_share",
                                                     "official_format_check")})
        if args.clean:
            state["matches_official"] = actual == state["expected_sha256"]
        common.write_json(state_path, state)
    except subprocess.TimeoutExpired:
        state.update(status="stopped_time_budget", updated=common.now())
        common.write_json(state_path, state)
        raise
    except BaseException as exc:
        state.update(status="failed", error=str(exc), updated=common.now())
        common.write_json(state_path, state)
        traceback.print_exc()
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", default="outputs/ha_production")
    parser.add_argument("--max-fold-minutes", type=float, default=85)
    parser.add_argument("--clean", action="store_true", help="fresh-clone replay of the sealed official file")
    parser.add_argument("--cutoff", default="22:55", help="Istanbul stop time; after midnight only as a user-approved exception")
    parser.add_argument("--worker", choices=("fold", "stitch"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    main()
