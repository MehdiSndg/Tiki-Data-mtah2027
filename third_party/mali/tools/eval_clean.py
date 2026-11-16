r"""Score a finished external run's held-out DFL matches on the CLEAN DFL track (no ATD-like corruption).

Needs the run's fold nets (outputs/<run>/fold_nets/<match>/net_<seed>.pt, saved by adapt since 5 Nov). For
every held-out DFL match: the booster (the run's own config, which never trained on the match) and the fold
nets predict the uncorrupted track; blend, decode and score exactly as the run does. The same code on the
v1-corrupted track must reproduce the run's own v1 scores (printed as a check).

Clean = ball never masked, no smoothing / offsets / player gaps / ghost tracks. DFL freezes the dead ball
(visible, speed 0): a cue our ATD does not have, so clean DFL is easier than the target. Comparable with
numbers other teams report on clean tracks, not with our corrupted referees.

  .venv\Scripts\python.exe tools/eval_clean.py --run cl_ev_ext_met_dr3 [--booster-met]
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine  # noqa: E402
from mtah.pipeline.hybrid import log_pool, net_boundary, net_probs, step_view  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

OFF = dict(corruption="none", dead_jitter_m=0.0, ball_smooth_frames=0, ball_offset_m=0.0, dead_carry_prob=0.0,
           player_smooth_frames=0, player_gap_frac=0.0, player_gap_live=0.0, player_gap_dead=0.0, ghost_frac=0.0)


def clean_cfg(cfg):
    c = copy.deepcopy(cfg)
    keep = {k: v for k, v in (cfg.get("dfl_referee") or {}).items() if k in ("corruption_seed", "align_max_frames")}
    c["dfl_referee"] = dict(OFF, **keep)
    c.pop("dfl_referee_train", None)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--booster-met", action="store_true",
                    help="also score with the Metrica-trained booster in the blend (tools/booster_met.py)")
    args = ap.parse_args()
    import bar_from_nets
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, p = adapt.params(cfg)
    w = float(ac["w_boost"])
    stride = int(p["stride"])
    dcfg = adapt.blend_cfg(cfg, ac)
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    lengths = cloud.bar_lengths(cfg)
    ccfg = clean_cfg(cfg)
    extra = [m for m in ac["eval_matches"] if not m.startswith(("metrica_", "WCSF"))]
    if not (run / "fold_nets").is_dir():
        raise SystemExit(f"{run} has no fold_nets (runs before 5 Nov did not save them)")
    old_v1 = decoder_tune.unseen_probs(cfg, extra, folder)
    clean_preps = [dfl_referee.prepare(m, ccfg, lengths) for m in extra]
    old_clean = decoder_tune.unseen_probs(cfg, [m + "_clean" for m in extra], folder, preps=clean_preps)
    new = None
    if args.booster_met:
        from booster_met import fit
        new = dict(zip(extra, fit(cfg, [1, 2, 3], [pp["feats"] for pp in clean_preps],
                                  list(clean_preps[0]["feats"].columns), lengths)))
    res = {"run": args.run, "w_boost": w, "rows": []}
    for m, cprep in zip(extra, clean_preps):
        nets, columns = bar_from_nets.load_nets(run / "fold_nets" / m)
        models = [n for _, n in nets]
        gt = np.where(cprep["data"].period > 0,
                      suppress_short_returns(cprep["data"].labels, cprep["runs"], 35, cprep["data"], True),
                      C.X).astype(np.int8)
        row = {"match": m}
        # (view, prep, net-input booster = the run's own, blend booster)
        views = [("v1_check", dfl_referee.prepare(m, cfg, lengths), old_v1[m], old_v1[m]),
                 ("clean", cprep, old_clean[m + "_clean"], old_clean[m + "_clean"])]
        if new is not None:
            views.append(("clean_booster_met", cprep, old_clean[m + "_clean"], new[m]))
        for view, prep, pin, pb in views:
            assert list(prep["feats"].columns) == columns
            steps = step_view(prep["feats"], columns, pin, None, prep["runs"], stride)
            pn = net_probs(models, steps, len(pb), pin, stride, None)
            bnd = net_boundary(models, steps, len(pb), stride)
            reg = contact.regime_for(prep["corrupted"], dcfg)
            for ww in (0.0, w):
                pred = engine.decode_runs(log_pool(pb, pn, ww), prep["runs"], dcfg, boundary=bnd, regime=reg)
                row[f"{view}_w{ww:g}"] = round(pm.evaluate(gt, pred)["spell_f1_mean"], 4)
        res["rows"].append(row)
        print(row, flush=True)
    keys = [k for k in res["rows"][0] if k != "match"]
    res["mean"] = {k: round(float(np.mean([r[k] for r in res["rows"]])), 4) for k in keys}
    try:
        import run_day_screen as screen
        res["run_v1_from_summary"] = screen.met_scores(summ)["dfl_v1_per_match"]
    except Exception as e:      # noqa: BLE001
        print("summary read failed:", e)
    print("mean", res["mean"], "| run's own v1:", res.get("run_v1_from_summary"))
    (run / "eval_clean.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
