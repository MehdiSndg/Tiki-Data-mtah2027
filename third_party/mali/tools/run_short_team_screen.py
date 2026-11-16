r"""Run a paired, two-seed screen only after the existing Q.4 chain has finished.

    .venv\Scripts\python.exe -u tools/run_short_team_screen.py --wait-minutes 120

Child processes use the same project interpreter. Only scalar/aggregate results are printed.
No public scores are used in the comparison. All generated files stay under outputs/.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("MTAH_OUTPUTS_DIR") or ROOT / "outputs").resolve()
sys.path.insert(0, str(ROOT / "src"))

from mtah import config  # noqa: E402

TRIALS = ("ev_short_base_rf0", "ev_short_team_rf0")
PREREQUISITES = ("ev_mix_rf0", "ev_ref2b_rf0", "ev_mix_rf1", "ev_ref2b_rf1")


def finished_summary(name):
    path = OUTPUT_ROOT / name / "summary.json"
    if not path.is_file():
        return None
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not s.get("dfl_lomo") or not s.get("dfl_lomo_eval") or len(s.get("metrica_eval", [])) != 3:
        return None
    return s


def gpu_idle():
    """Check nvidia-smi before each training, in addition to the explicit completion gates."""
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20, check=True)
    busy = any(float(v.strip()) >= 10 for v in util.stdout.splitlines() if v.strip())
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20, check=True)
    named_training = any("python" in line.lower() for line in procs.stdout.splitlines())
    return not busy and not named_training


def wait_for_previous_chain(deadline):
    last = None
    while True:
        pending = tuple(name for name in PREREQUISITES if finished_summary(name) is None)
        if pending != last:
            print(json.dumps({"waiting_for": pending}), flush=True)
            last = pending
        if not pending and gpu_idle():
            # Require repeated idle observations; a single gap between training batches is insufficient.
            time.sleep(5)
            if gpu_idle():
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing training has not demonstrably completed; no new training started")
        time.sleep(30)


def scores(s):
    row = next(r for r in s["dfl_lomo"] if r["net"] == "adapted" and r["w_boost"] == 0.25)
    ev = next(r for r in s["dfl_lomo_eval"] if r["w_boost"] == 0.25)
    bar = next(r for r in s["bar"] if r["net"] == "adapted" and r["w_boost"] == 0.25)
    return {"dfl_v1": row["spell_gt35"], "dfl_v2b": ev["spell_gt35"],
            "metrica": sum(r["spell"] for r in s["metrica_eval"]) / 3,
            "dfl_v1_per_match": row["per_match_gt35"], "dfl_v2b_per_match": ev["per_match_gt35"],
            "metrica_per_match": {r["match"]: r["spell"] for r in s["metrica_eval"]},
            "bar": {k: bar[k] for k in ("n_possessions", "dead_share", "home_share")}}


def main():
    global TRIALS, PREREQUISITES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-minutes", type=float, default=120)
    parser.add_argument("--main-backbone", action="store_true")
    args = parser.parse_args()
    expected_python = Path(os.environ.get("MTAH_PYTHON") or ROOT / ".venv" / "Scripts" / "python.exe")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Use the project's .venv interpreter")
    if args.main_backbone:
        TRIALS = ("ev_short_main_base_rf0", "ev_short_main_team_rf0")
        PREREQUISITES = tuple(f"main_rf{k}" + suffix for suffix in ("", "_s7027") for k in range(4))
    out = OUTPUT_ROOT / ("short_team_main_screen" if args.main_backbone else "short_team_screen")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "running.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    state = {"status": "initializing"}
    state_path = out / "state.json"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        state = {"revision": revision, "status": "waiting", "prerequisites": PREREQUISITES,
                 "trials": {name: {"config_hash": config.digest(config.load(name))} for name in TRIALS}}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        deadline = time.monotonic() + args.wait_minutes * 60
        wait_for_previous_chain(deadline)
        summaries = {}
        for name in TRIALS:
            if finished_summary(name) is not None:
                raise RuntimeError(f"{name} already has a result; inspect it before rerunning")
            wait_for_previous_chain(deadline)
            state["status"], state["current"] = "running", name
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            cmd = [sys.executable, "-u", str(ROOT / "run.py"), "adapt", "--config", name, "--name", name]
            print(json.dumps({"starting": name, "revision": revision}), flush=True)
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            env.update(MTAH_OFFLINE="1", PYTHONPATH=str(ROOT / "tools" / "offline_site"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", WANDB_MODE="disabled")
            log_path = out / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            summary = finished_summary(name)
            complete = (summary is not None and "done in " in log_path.read_text(encoding="utf-8")
                        and all((OUTPUT_ROOT / name / f).is_file()
                                for f in ("probs.npy", "bar_bnd_adapted.npy", "submission.csv")))
            state["trials"][name]["exit_code"] = child.returncode
            if not complete:
                raise RuntimeError(f"{name} has no verified complete output; see its local log")
            summaries[name] = summary
            state["trials"][name]["scores"] = scores(summary)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(json.dumps({"finished": name, **scores(summary)}), flush=True)
        ref, trial = (scores(summaries[name]) for name in TRIALS)
        result = {"revision": revision, "protocol": "2 seeds; 3 booster-unseen DFL folds; Metrica never trained",
                  "reference": ref, "candidate": trial,
                  "delta": {k: trial[k] - ref[k] for k in ("dfl_v1", "dfl_v2b", "metrica")},
                  "note": "screening only; no automatic promotion; public scores are excluded"}
        (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        state["status"] = "complete"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({"comparison": result}), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
