"""Pinned, offline daytime H/A screening, with a strict Istanbul 23:00 boundary.

Only one child fit at a time. Stop owned child trees by 22:55; never start a fit whose
declared maximum runtime plus a five-minute margin will not fit. No automatic promotion.
"""

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import run_mix_promotion as common
from run_short_team_screen import scores

ROOT = Path(__file__).resolve().parents[1]
ISTANBUL = timezone(timedelta(hours=3), "Europe/Istanbul")
TRIALS = ("ev_ha_base_rf0", "ev_ha_repeat_rf0", "ev_ha_swap_rf0")
CONFIRM_TRIALS = tuple(f"ev_ha_confirm_{kind}_rf{region}" for region in (0, 1) for kind in ("base", "swap"))
BOUNDARY_TRIALS = ("ev_bmask_base_rf0", "ev_bmask_team0_rf0")
SUPPORT_TRIALS = ("ev_support_base_rf0", "ev_support_scale_rf0", "ev_support_known_rf0")
SET_TRIALS = ("ev_set_base_rf0", "ev_set_gru_rf0", "ev_set_pool_rf0")
EVENING_TRIALS = tuple(f"ev_evening_{kind}_rf1" for kind in ("base", "known", "context128", "ft4", "ft12"))
CTX_TRIALS = tuple(f"ev_ctx_confirm_{kind}_rf{region}" for region in (0, 1) for kind in ("base", "c128"))
G11_TRIALS = tuple(f"ev_g11_{kind}_rf{region}" for region in (0, 1) for kind in ("base", "gt11"))
TTA_TRIALS = ("ev_tta_rf0", "ev_tta_rf1")
SYM_TRIALS = ("ev_symft_rf0", "ev_symft_rf1")
EXT_TRIALS = ("ev_ext",)
EXTMET_TRIALS = ("ev_ext_met",)   # reference: the finished ev_ext run (same net seeds)
EXTMETSC_TRIALS = ("ev_ext_met_sc",)   # reference: the finished ev_ext_met run (same net seeds)
EXTMETSC20_TRIALS = ("ev_ext_met_sc20",)   # same reference
EXTMETSCD_TRIALS = ("ev_ext_met_scd", "ev_ext_met_sc20d")   # SkillCorner dead/alive only; same reference
# plans scored pairwise against the finished ev_ext_met (same folds, referees and net seeds)
EXTMET_VS = {"ext-met-scd": EXTMETSCD_TRIALS, "ext-met-dr": ("ev_ext_met_dr3",), "ext-met-big": ("ev_ext_met_big",), "ext-met-wcsf": ("ev_ext_met_wcsf",)}
SYMC_TRIALS = tuple(f"ev_symft_confirm_{kind}_rf{region}" for region in (0, 1) for kind in ("base", "sym"))      # references: the plain outputs of ev_tta_rf{0,1} (same seeds)
# User decision 31 Oct (docs/research/rapor9_inceleme_ve_plan_2026-11-01.md section 4): selection = 4 gates.
GATES = dict(weights={"dfl_v2b": 0.45, "metrica": 0.35, "dfl_v1": 0.20}, screen_gain=0.004, production_gain=0.0035,
             floor={"dfl_v2b": -0.002, "metrica": -0.002, "dfl_v1": -0.004}, min_positive=5,
             possessions=(220, 285), dead=(0.425, 0.475), home=(0.585, 0.645))


def four_gates(pairs, stage="screen"):
    """pairs: {region: {"reference": scores, "candidate": scores}} over the same matches."""
    keys = tuple(GATES["weights"])
    comp = lambda r: sum(w * r[k] for k, w in GATES["weights"].items())  # noqa: E731
    delta = {k: sum(p["candidate"][k] - p["reference"][k] for p in pairs.values()) / len(pairs) for k in keys}
    d_comp = sum(comp(p["candidate"]) - comp(p["reference"]) for p in pairs.values()) / len(pairs)
    per_match = {}
    for kind in ("dfl_v2b", "metrica"):
        for m in next(iter(pairs.values()))["reference"][f"{kind}_per_match"]:
            per_match[f"{kind}:{m}"] = sum(p["candidate"][f"{kind}_per_match"][m] - p["reference"][f"{kind}_per_match"][m]
                                           for p in pairs.values()) / len(pairs)
    bar = [p["candidate"]["bar"] for p in pairs.values()]
    inside = lambda v, lo_hi: lo_hi[0] <= v <= lo_hi[1]  # noqa: E731
    gates = {"1_composite": d_comp >= GATES[f"{stage}_gain"],
             "2_non_inferiority": all(delta[k] >= GATES["floor"][k] for k in keys),
             "3_match_signs": sum(v > 0 for v in per_match.values()) >= GATES["min_positive"],
             "4_label_free": all(inside(b["n_possessions"], GATES["possessions"]) and inside(b["dead_share"], GATES["dead"])
                                 and inside(b["home_share"], GATES["home"]) for b in bar)}
    return {"stage": stage, "delta": delta, "delta_composite": d_comp, "per_match_delta": per_match,
            "positive_matches": sum(v > 0 for v in per_match.values()), "gates": gates, "pass": all(gates.values()),
            "note": "short-segment recall and fixed D region are checked separately; promotion still needs the user"}


def met_scores(s):
    """Scores of an external run whose folds include Metrica LOMO (Metrica trained on)."""
    row = next(r for r in s["dfl_lomo"] if r["net"] == "adapted" and r["w_boost"] == 0.25)
    ev = next(r for r in s["dfl_lomo_eval"] if r["w_boost"] == 0.25)
    bar = next(r for r in s["bar"] if r["net"] == "adapted" and r["w_boost"] == 0.25)
    met = {m: v for m, v in row["per_match_gt35"].items() if m.startswith("metrica_")}
    out = {"dfl_v1": row["spell_gt35"], "dfl_v2b": ev["spell_gt35"], "metrica": sum(met.values()) / len(met),
           "dfl_v1_per_match": {m: v for m, v in row["per_match_gt35"].items() if m not in met},
           "dfl_v2b_per_match": ev["per_match_gt35"], "metrica_per_match": met,
           "bar": {k: bar[k] for k in ("n_possessions", "dead_share", "home_share")}}
    # 5 Nov: the WCSF match (Groningen) is scored like a DFL match but reported apart, so the DFL-3
    # referees of the gates keep exactly their matches
    if "WCSF" in out["dfl_v1_per_match"]:
        out["wcsf_v1"] = out["dfl_v1_per_match"].pop("WCSF")
        out["wcsf_v2b"] = out["dfl_v2b_per_match"].pop("WCSF", None)
        mean = lambda d: sum(d.values()) / len(d)  # noqa: E731
        out["dfl_v1"], out["dfl_v2b"] = mean(out["dfl_v1_per_match"]), mean(out["dfl_v2b_per_match"])
    return out


def compare(plan, summaries, revision):
    keys = ("dfl_v1", "dfl_v2b", "metrica")
    if plan in EXTMET_VS:
        reference = met_scores(common.output_summary(ROOT / "outputs", "ev_ext_met"))
        pairs = {}
        for cand_name in EXTMET_VS[plan]:
            cand = met_scores(summaries[cand_name])
            pairs[cand_name] = {"reference": reference, "candidate": cand,
                                "delta": {k: cand[k] - reference[k] for k in keys},
                                "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}
        return {"revision": revision, "pairs": pairs,
                "four_gates": {n: four_gates({n: p}) for n, p in pairs.items()},
                "note": f"external data only; {', '.join(EXTMET_VS[plan])} vs ev_ext_met; same folds, referees and 5 net seeds"}
    if plan in ("ext-met", "ext-met-sc", "ext-met-sc20"):
        ref_name, cand_name = {"ext-met": ("ev_ext", "ev_ext_met"), "ext-met-sc": ("ev_ext_met", "ev_ext_met_sc"),
                               "ext-met-sc20": ("ev_ext_met", "ev_ext_met_sc20")}[plan]
        ref_summary = common.output_summary(ROOT / "outputs", ref_name)
        # ev_ext_met trains on Metrica: its Metrica scores are Metrica-LOMO folds inside dfl_lomo, read
        # them like the candidate's (5 Nov fix: scores() left them under dfl_v1 and Metrica at 0)
        reference = scores(ref_summary) if ref_name == "ev_ext" else met_scores(ref_summary)
        cand = met_scores(summaries[cand_name])
        pairs = {"rf_ext": {"reference": reference, "candidate": cand,
                            "delta": {k: cand[k] - reference[k] for k in keys},
                            "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}}
        return {"revision": revision, "pairs": pairs, "mean_delta": pairs["rf_ext"]["delta"],
                "four_gates": four_gates(pairs),
                "note": ("external data only; Metrica 1-3 trained (raw + BAR-like corrupted + H/A twins), scored "
                         "leave-one-Metrica-game-out vs ev_ext where Metrica was never trained on; same 5 net seeds"
                         if plan == "ext-met" else
                         f"external data only; + SkillCorner open games ({cand_name}) as training data (phase-of-play "
                         "labels, raw + BAR-like corrupted + H/A twins) vs ev_ext_met; same folds, referees and 5 net seeds")}
    if plan == "ext":
        cand = scores(summaries["ev_ext"])
        pairs = {}
        for region in (0, 1):
            reference = scores(common.output_summary(ROOT / "outputs", f"ev_tta_rf{region}"))
            pairs[f"rf{region}"] = {"reference": reference, "candidate": cand,
                                    "delta": {k: cand[k] - reference[k] for k in keys},
                                    "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}
        return {"revision": revision, "pairs": pairs,
                "mean_delta": {k: sum(p["delta"][k] for p in pairs.values()) / 2 for k in keys},
                "four_gates": four_gates(pairs),
                "note": "external-only recipe (no provided-game training: DFL-only booster without Smart Tagging, "
                        "DFL-only net, no BAR fine-tuning) vs the official recipe's plain outputs, same 5 net seeds; "
                        "compliance comparison, not a gain screen"}
    if plan == "symft":
        pairs = {}
        for region in (0, 1):
            reference = scores(common.output_summary(ROOT / "outputs", f"ev_tta_rf{region}"))
            cand = scores(summaries[f"ev_symft_rf{region}"])
            pairs[f"rf{region}"] = {"reference": reference, "candidate": cand,
                                    "delta": {k: cand[k] - reference[k] for k in keys},
                                    "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}
        return {"revision": revision, "pairs": pairs,
                "mean_delta": {k: sum(p["delta"][k] for p in pairs.values()) / 2 for k in keys},
                "four_gates": four_gates(pairs),
                "note": "same 5 network seeds 42027..46027 as the plain ev_tta_rf{0,1} outputs (official H/A recipe); "
                        "only change: BAR weak anchors also fine-tuned as team-exchanged twins; no promotion"}
    if plan == "tta":
        pairs = {}
        for region in (0, 1):
            s = summaries[f"ev_tta_rf{region}"]
            reference, cand = scores(s), scores(s["tta"])
            pairs[f"rf{region}"] = {"reference": reference, "candidate": cand,
                                    "delta": {k: cand[k] - reference[k] for k in keys},
                                    "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}
        return {"revision": revision, "pairs": pairs,
                "mean_delta": {k: sum(p["delta"][k] for p in pairs.values()) / 2 for k in keys},
                "four_gates": four_gates(pairs),
                "note": "in-run pairing: the SAME 5 network seeds 42027..46027 predict the plain and the team-exchanged "
                        "views; official H/A recipe; 2 regions over the same 3 DFL + 3 Metrica; no promotion"}
    paired = {"ctx-confirm": ("ev_ctx_confirm", "c128", "3 fresh paired network seeds 22027..24027; H/A backbone; "
                                                      "2 regions over the SAME 3 DFL + 3 Metrica; context 128 vs 512; no promotion"),
              "symft-confirm": ("ev_symft_confirm", "sym", "5 fresh paired network seeds 52027..56027; official H/A recipe; "
                                "2 regions over the SAME 3 DFL + 3 Metrica; BAR anchors also as team-exchanged twins; no promotion"),
              "gt11": ("ev_g11", "gt11", "3 fresh paired network seeds 32027..34027; official H/A recipe; 2 regions over the "
                                         "SAME 3 DFL + 3 Metrica; DFL net target gt20 vs gt11; no promotion")}
    if plan in paired:
        prefix, kind, note = paired[plan]
        pairs = {}
        for region in (0, 1):
            reference = scores(summaries[f"{prefix}_base_rf{region}"])
            cand = scores(summaries[f"{prefix}_{kind}_rf{region}"])
            pairs[f"rf{region}"] = {"reference": reference, kind: cand,
                                    "delta": {k: cand[k] - reference[k] for k in keys},
                                    "bar_possessions": [reference["bar"]["n_possessions"], cand["bar"]["n_possessions"]]}
        out = {"revision": revision, "pairs": pairs,
               "mean_delta": {k: sum(p["delta"][k] for p in pairs.values()) / 2 for k in keys}, "note": note}
        if plan == "symft-confirm":
            out["four_gates"] = four_gates({r: {"reference": p["reference"], "candidate": p[kind]} for r, p in pairs.items()})
        return out
    if plan == "evening":
        reference = scores(summaries[EVENING_TRIALS[0]])
        candidates = {name: scores(summaries[name]) for name in EVENING_TRIALS[1:]}
        return {"revision": revision, "reference": reference, "candidates": candidates,
                "delta": {name: {k: row[k] - reference[k] for k in keys}
                          for name, row in candidates.items()},
                "note": "2 paired fresh network seeds 4027/5027; region 1; known-support confirmation, shorter context, adaptation dose 4/8/12; no promotion"}
    if plan == "player-set":
        reference, gru, pooled = [scores(summaries[n]) for n in SET_TRIALS]
        return {"revision": revision, "reference": reference, "gru": gru, "pooled": pooled,
                "pooled_minus_reference": {k: pooled[k] - reference[k] for k in keys},
                "pooled_minus_gru": {k: pooled[k] - gru[k] for k in keys},
                "note": "2 paired seeds; H/A backbone; scalar BiGRU control and team DeepSets-BiGRU member; fixed blends separate; no promotion"}
    if plan == "weak-support":
        reference, scale, known = [scores(summaries[n]) for n in SUPPORT_TRIALS]
        return {"revision": revision, "reference": reference, "scale": scale, "known": known,
                "known_minus_reference": {k: known[k] - reference[k] for k in keys},
                "known_minus_scale": {k: known[k] - scale[k] for k in keys},
                "scale_minus_reference": {k: scale[k] - reference[k] for k in keys},
                "note": "2 paired seeds on H/A backbone; weak-window TMSE support versus equal-gate-weight strength control; no promotion"}
    if plan == "boundary-mask":
        reference, candidate = [scores(summaries[n]) for n in BOUNDARY_TRIALS]
        return {"revision": revision, "reference": reference, "candidate": candidate,
                "delta": {k: candidate[k] - reference[k] for k in keys},
                "note": "2 paired seeds on H/A backbone; external true H/A-edge TMSE mask only; unchanged BAR loss/decoder; no promotion"}
    if plan == "ha-confirm":
        pairs = {}
        for region in (0, 1):
            reference = scores(summaries[f"ev_ha_confirm_base_rf{region}"])
            swap = scores(summaries[f"ev_ha_confirm_swap_rf{region}"])
            pairs[f"rf{region}"] = {"reference": reference, "swap": swap,
                                    "delta": {k: swap[k] - reference[k] for k in keys}}
        return {"revision": revision, "pairs": pairs,
                "mean_delta": {k: sum(p["delta"][k] for p in pairs.values()) / 2 for k in keys},
                "note": "5 fresh network seeds 12027..16027; 2 region folds over the SAME 3 DFL + 3 Metrica matches; no promotion"}
    reference, repeat, swap = [scores(summaries[n]) for n in TRIALS]
    return {"revision": revision, "reference": reference, "repeat": repeat, "swap": swap,
            "swap_minus_reference": {k: swap[k] - reference[k] for k in keys},
            "swap_minus_repeat": {k: swap[k] - repeat[k] for k in keys},
            "note": "2 seeds, 3 booster-unseen DFL matches, untouched Metrica; no public selection; no promotion"}


def budget_seconds(current, cutoff, maximum, margin=300):
    remaining = (cutoff - current).total_seconds()
    return maximum if remaining >= maximum + margin else 0


def stop_owned_child(child):
    """The handle/PID comes exclusively from this controller's own Popen call."""
    if child.poll() is None:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=True,
                       capture_output=True, text=True, timeout=30)
        child.wait(timeout=30)


def fit(cmd, cwd, env, log_path, seconds):
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return child.wait(timeout=seconds)
        except BaseException:
            stop_owned_child(child)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--max-trial-minutes", type=float, default=90)
    parser.add_argument("--plan", choices=("ha-screen", "ha-confirm", "boundary-mask", "weak-support", "player-set", "evening", "ctx-confirm", "gt11", "tta", "symft", "symft-confirm", "ext", "ext-met", "ext-met-sc", "ext-met-sc20", *EXTMET_VS), default="ha-screen")
    parser.add_argument("--overnight-until", help="HH:MM next morning: user-approved overnight exception to the "
                        "23:00 training boundary (Istanbul)")
    parser.add_argument("--allow-parallel", action="store_true", help="user-approved (5 Nov): start even "
                        "while another training process runs (GPU memory allows several screens)")
    args = parser.parse_args()
    if os.name != "nt" or Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project interpreter on Windows")
    if not 0 < args.max_trial_minutes <= 600:      # 5 Nov: larger trials (3 draws, big net); 23:00 boundary unchanged
        raise ValueError("Trial runtime must be between zero and 600 minutes")
    current = datetime.now(ISTANBUL)
    cutoff = current.replace(hour=22, minute=55, second=0, microsecond=0)
    if args.overnight_until:
        from datetime import timedelta
        hh, mm = (int(x) for x in args.overnight_until.split(":"))
        if not 0 <= hh < 12:
            raise ValueError("--overnight-until is a morning time (00:00-11:59)")
        cutoff = current.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cutoff <= current:
            cutoff += timedelta(days=1)
    if not budget_seconds(current, cutoff, args.max_trial_minutes * 60):
        raise RuntimeError("No new fit: not enough time before the 23:00 boundary")
    if common.git(ROOT, "status", "--porcelain"):
        raise RuntimeError("Commit source changes before starting a pinned experiment")
    if not args.allow_parallel:
        common.ensure_no_training()
    job = Path(args.job_dir).resolve()
    if not job.is_relative_to((ROOT / "outputs").resolve()):
        raise RuntimeError("Job must be inside workspace outputs")
    job.mkdir(parents=True, exist_ok=True)
    # parallel screens: one lock per job (trial output names still refuse to overwrite each other)
    lock = job / "day_training.lock" if args.allow_parallel else ROOT / "outputs/day_training.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    awake = False
    state = {"status": "initializing", "pid": os.getpid(), "started": common.now(), "offline": True,
             "cutoff_istanbul": cutoff.isoformat(), "plan": args.plan, "trials": {}}
    try:
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000003))
        if not awake:
            raise RuntimeError("Could not request system and display wakefulness")
        revision = common.git(ROOT, "rev-parse", "HEAD")
        code = job / "code"
        common.clone_local(ROOT, code, revision)
        data = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "more-than-a-hack-2027/data").resolve()
        common.preflight(ROOT, data)
        env = common.offline_env(code, data, ROOT)
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                       env=env, check=True, timeout=90)
        state.update(revision=revision, status="running")
        common.write_json(job / "state.json", state)
        summaries = {}
        trials = {"ha-screen": TRIALS, "ha-confirm": CONFIRM_TRIALS,
                  "boundary-mask": BOUNDARY_TRIALS, "weak-support": SUPPORT_TRIALS,
                  "player-set": SET_TRIALS, "evening": EVENING_TRIALS, "ctx-confirm": CTX_TRIALS, "gt11": G11_TRIALS,
                  "tta": TTA_TRIALS, "symft": SYM_TRIALS, "symft-confirm": SYMC_TRIALS, "ext": EXT_TRIALS, "ext-met": EXTMET_TRIALS, "ext-met-sc": EXTMETSC_TRIALS, "ext-met-sc20": EXTMETSC20_TRIALS, **EXTMET_VS}[args.plan]
        for name in trials:
            seconds = budget_seconds(datetime.now(ISTANBUL), cutoff, args.max_trial_minutes * 60)
            if not seconds:
                state.update(status="deferred_time_budget", next_trial=name)
                common.write_json(job / "state.json", state)
                return
            if not args.allow_parallel:
                common.ensure_no_training()
            if (ROOT / "outputs" / name).exists():
                raise RuntimeError(f"Output {name} already exists; never overwrite a previous/partial trial")
            state.update(current=name, updated=common.now())
            common.write_json(job / "state.json", state)
            log = job / f"{name}.log"
            exit_code = fit([sys.executable, "-u", str(code / "run.py"), "adapt", "--config", name, "--name", name],
                            code, env, log, seconds)
            summary = common.output_summary(ROOT / "outputs", name)
            if "done in " not in log.read_text(encoding="utf-8"):
                raise RuntimeError(f"{name} lacks a completed log")
            summaries[name] = summary
            state["trials"][name] = {"exit_code": exit_code, "scores": scores(summary), "finished": common.now()}
            common.write_json(job / "state.json", state)
        comparison = compare(args.plan, summaries, revision)
        if args.plan == "player-set":
            seconds = min(600, (cutoff - datetime.now(ISTANBUL)).total_seconds() - 300)
            if seconds < 60:
                raise RuntimeError("No time left for fixed-blend evaluation")
            state.update(current="fixed_blends_cpu", updated=common.now())
            common.write_json(job / "state.json", state)
            mix_env = dict(env, CUDA_VISIBLE_DEVICES="-1")
            blend_log = job / "fixed_blends.log"
            fit([sys.executable, "-u", str(code / "tools/run_member_blend.py"), "--job-dir", str(job)],
                code, mix_env, blend_log, seconds)
            if "FIXED_BLENDS_COMPLETE" not in blend_log.read_text(encoding="utf-8"):
                raise RuntimeError("Fixed-blend evaluation incomplete")
            comparison["fixed_blends"] = json.loads((job / "fixed_blends.json").read_text(encoding="utf-8"))
        common.write_json(job / "comparison.json", comparison)
        state.update(status="complete", finished=common.now())
        common.write_json(job / "state.json", state)
    except subprocess.TimeoutExpired:
        state.update(status="stopped_time_budget", finished=common.now())
        common.write_json(job / "state.json", state)
        raise
    except BaseException as exc:
        state.update(status="failed", error=str(exc), updated=common.now())
        common.write_json(job / "state.json", state)
        traceback.print_exc()
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


if __name__ == "__main__":
    main()
