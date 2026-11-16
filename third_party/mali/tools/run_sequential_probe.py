"""Frozen-prediction CPU experiment: joint HSMM, joint minimum-dwell control, staged dead/team.

Protected arrays are consumed only by local algorithms. Only aggregate scores/counts are
written or printed. No video, frame labels, probabilities, public scoring or model training.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

import run_mix_promotion as common
from run_day_screen import fit

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("ev_ha_confirm_swap_rf0", "ev_ha_confirm_swap_rf1")
METHODS = ("reference", "joint_minimum", "staged")


def worker(job):
    import socket
    if os.environ.get("MTAH_OFFLINE") != "1" or socket.socket.connect.__module__ != "sitecustomize":
        raise RuntimeError("Offline guard is required")
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from mtah import config, constants as C
    from mtah.decoding import contact, sequential
    from mtah.io.broadcast import bar_data
    from mtah.labels.align import suppress_short_returns
    from mtah.labels.masking import gap_lengths
    from mtah.paths import OUTPUTS
    from mtah.pipeline import adapt, dfl_referee, engine, metrica_src
    from mtah.validation import sanity
    from mtah.validation.official import pm
    from mtah.validation.possession import diagnostics
    from run_short_team_screen import scores

    results = {}
    for source in SOURCES:
        cfg = config.load(source)
        dc = adapt.blend_cfg(cfg, adapt.params(cfg)[0])
        folder = OUTPUTS / source
        expected = scores(json.loads((folder / "summary.json").read_text(encoding="utf-8")))
        data = bar_data(cfg)
        lengths = gap_lengths(data)
        result = {m: {g: {} for g in ("dfl_v1", "dfl_v2b", "metrica")} for m in METHODS}
        for group, probs_file, boundary_file in (
                ("dfl_v1", "dfl_lomo_blend.npz", "dfl_lomo_bnd.npz"),
                ("dfl_v2b", "dfl_lomo_eval_blend.npz", "dfl_lomo_eval_bnd.npz"),
                ("metrica", "metrica_eval_blend.npz", "metrica_eval_bnd.npz")):
            with np.load(folder / probs_file) as probabilities, np.load(folder / boundary_file) as boundaries:
                for match in probabilities.files:
                    if group == "metrica":
                        prep = metrica_src.prepare(int(match.split("_")[-1]), cfg)
                        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
                    else:
                        effective = dfl_referee.eval_cfg(cfg) if group == "dfl_v2b" else cfg
                        prep = dfl_referee.prepare(match, effective, lengths)
                        clean = prep["data"]
                        gt = np.where(clean.period > 0, suppress_short_returns(clean.labels, clean.runs(), 35,
                                      clean, True), C.X).astype(np.int8)
                    probs, boundary = probabilities[match], boundaries[match]
                    regime = contact.regime_for(prep.get("corrupted", prep["data"]), dc)
                    for method in METHODS:
                        if method == "reference":
                            labels = engine.decode_runs(probs, prep["runs"], dc, boundary=boundary, regime=regime)
                        else:
                            labels = sequential.decode_runs(probs, prep["runs"], dc, method, boundary, regime)
                        score = pm.evaluate(gt, labels)
                        result[method][group][match] = {
                            "spell": score["spell_f1_mean"], "macro": score["macro_f1"],
                            "segments": diagnostics(gt, labels, prep["runs"], prep["data"].fps)}
                    print(json.dumps({"source": source, "group": group, "match": match,
                                      "spell": {m: result[m][group][match]["spell"] for m in METHODS}}), flush=True)
                    common.write_json(job / "progress.json", {"source": source, "group": group, "match": match})
        probs = np.load(folder / "probs.npy")
        boundary = np.load(folder / "bar_bnd_adapted.npy")
        regime = contact.regime_for(data, dc)
        for method in METHODS:
            labels = (engine.decode_runs(probs, list(C.HALVES.values()), dc, C.FORCED_DEAD,
                                         boundary=boundary, regime=regime) if method == "reference" else
                      sequential.decode_runs(probs, list(C.HALVES.values()), dc, method, boundary, regime, C.FORCED_DEAD))
            result[method]["bar_health"] = sanity.stats(labels)
            result[method]["mean_scores"] = {g: float(np.mean([r["spell"] for r in result[method][g].values()]))
                                              for g in ("dfl_v1", "dfl_v2b", "metrica")}
        for group, value in result["reference"]["mean_scores"].items():
            if abs(value - expected[group]) > 1e-8:
                raise RuntimeError(f"Reference reproduction mismatch: {source}/{group}")
        if result["reference"]["bar_health"]["n_possessions"] != expected["bar"]["n_possessions"]:
            raise RuntimeError("Reference full-match health does not reproduce")
        results[source] = result
    comparisons = {}
    for method in ("joint_minimum", "staged"):
        comparisons[method] = {g: float(np.mean([r[method]["mean_scores"][g] - r["reference"]["mean_scores"][g]
                                               for r in results.values()]))
                               for g in ("dfl_v1", "dfl_v2b", "metrica")}
    common.write_json(job / "result.json", {"sources": results, "mean_delta_vs_reference": comparisons,
                      "note": "Frozen H/A five-seed predictions; same 3 DFL and 3 Metrica matches across two regions; no training/public selection/promotion"})
    print("CPU_PROBE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    job = Path(args.job_dir).resolve()
    if args.worker:
        worker(job)
        return
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project interpreter")
    if not job.is_relative_to((ROOT / "outputs").resolve()):
        raise RuntimeError("Job must be in workspace outputs")
    if common.git(ROOT, "status", "--porcelain"):
        raise RuntimeError("Commit source before pinning the CPU probe")
    if (job / "state.json").exists():
        raise RuntimeError("Do not overwrite a previous probe")
    job.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "pid": os.getpid(), "started": common.now(), "device": "cpu", "offline": True}
    awake = False
    try:
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000003))
        revision = common.git(ROOT, "rev-parse", "HEAD")
        common.clone_local(ROOT, job / "code", revision)
        state["revision"] = revision
        common.write_json(job / "state.json", state)
        data = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "more-than-a-hack-2027/data").resolve()
        env = common.offline_env(job / "code", data, ROOT)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        exit_code = fit([sys.executable, "-u", str(job / "code/tools/run_sequential_probe.py"),
                         "--worker", "--job-dir", str(job)], job / "code", env, job / "worker.log", 1200)
        if not (job / "result.json").exists() or "CPU_PROBE_COMPLETE" not in (job / "worker.log").read_text(encoding="utf-8"):
            raise RuntimeError(f"CPU probe incomplete, exit {exit_code}")
        state.update(status="complete", finished=common.now(), exit_code=exit_code)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), updated=common.now())
        raise
    finally:
        common.write_json(job / "state.json", state)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
