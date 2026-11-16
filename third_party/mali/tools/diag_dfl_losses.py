r"""Where does the external spellF1 go? CPU decomposition on a finished run's saved blends (no training).

For every scored external match (DFL-3 under v1 and v2b corruption, Metrica LOGO) of the run:
  score             the run's decoded output (official spell F1, mean over IoU 0.3-0.7)
  argmax            frame-wise argmax of the blend, no decoder
  oracle_dead       re-decoded with the TRUE alive/dead put into the emission (team from the model)
  oracle_team       re-decoded with the TRUE team put into the emission on live frames (alive/dead from the model)
  per-IoU F1, spell counts and recall / precision by spell length.
The gap score -> oracle_dead is the alive/dead share of the loss, score -> oracle_team the team share.
Only aggregate numbers (external games only).

  .venv\Scripts\python.exe tools/diag_dfl_losses.py --run ev_ext_met --out outputs/diag_dfl_losses_ev_ext_met.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, dfl_referee, engine, metrica_src  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

BINS = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 10 ** 9)]      # frames (25 fps): <2 s, 2-4, 4-10, 10-20, >20


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    s = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return s["mean_f1"], {str(k): round(v["f1"], 4) for k, v in s["per_iou"].items()}, s["n_gt"], s["n_pred"]


def spells(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] in (C.H, C.A):
                out.append((s, i, int(lab[s])))
            s = i
    return out


def matched(a, others, thr=0.5):
    s, e, t = a
    for s2, e2, t2 in others:
        if t2 == t and e2 > s and s2 < e:
            inter = min(e, e2) - max(s, s2)
            if inter / (max(e, e2) - min(s, s2)) >= thr:
                return True
    return False


def by_length(gt, pred):
    gs, ps = spells(gt), spells(pred)
    out = {}
    for lo, hi in BINS:
        g = [x for x in gs if lo <= x[1] - x[0] < hi]
        p = [x for x in ps if lo <= x[1] - x[0] < hi]
        out[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = {
            "n_gt": len(g), "n_pred": len(p),
            "recall@0.5": round(np.mean([matched(x, ps) for x in g]), 3) if g else None,
            "precision@0.5": round(np.mean([matched(x, gs) for x in p]), 3) if p else None}
    return out


def oracle(probs, gt, kind):
    p = probs.copy()
    live = np.isin(gt, (C.H, C.A))
    dead = gt == C.D
    if kind == "dead":
        team = p[:, :2] / np.clip(p[:, :2].sum(1, keepdims=True), 1e-9, None)
        p[live, :2], p[live, 2] = team[live] * 0.999, 0.001
        p[dead] = (0.0005, 0.0005, 0.999)
    else:
        alive = 1 - p[:, 2]
        for t in (C.H, C.A):
            m = gt == t
            p[m, t], p[m, 1 - t] = alive[m] * 0.999, alive[m] * 0.001
    return np.clip(p, 1e-6, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, _ = adapt.params(cfg)
    dcfg = adapt.blend_cfg(cfg, ac)
    lengths = cloud.bar_lengths(cfg)
    ecfg = dfl_referee.eval_cfg(cfg)
    views = {"v1": (np.load(run / "dfl_lomo_blend.npz"), np.load(run / "dfl_lomo_bnd.npz"), cfg),
             "v2b": (np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz"), ecfg)}
    res = {"run": args.run, "rows": []}
    for view, (blend, bnd, vcfg) in views.items():
        for m in blend.files:
            if m.startswith("metrica_"):
                if view != "v1":
                    continue
                prep = metrica_src.prepare(int(m.split("_")[1]), cfg)
                gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
                src = "metrica"
            else:
                prep = dfl_referee.prepare(m, vcfg, lengths)
                gt = np.where(prep["data"].period > 0,
                              suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                              C.X).astype(np.int8)
                src = f"dfl_{view}"
            pr, b = blend[m], bnd[m]
            reg = contact.regime_for(prep["corrupted"], dcfg)

            def dec(p):
                return engine.decode_runs(p, prep["runs"], dcfg, boundary=b, regime=reg)

            pred = dec(pr)
            am = np.where(np.isin(gt, (C.H, C.A, C.D)), np.argmax(pr, 1), C.X).astype(np.int8)
            f, per_iou, n_gt, n_pred = spell_f1(gt, pred)
            graded = np.isin(gt, (C.H, C.A, C.D))
            live_gt, live_pr = np.isin(gt, (C.H, C.A)), np.isin(pred, (C.H, C.A))
            row = {"src": src, "match": m, "score": round(f, 4), "per_iou": per_iou, "n_gt": n_gt, "n_pred": n_pred,
                   "argmax": round(spell_f1(gt, am)[0], 4),
                   "oracle_dead": round(spell_f1(gt, dec(oracle(pr, gt, "dead")))[0], 4),
                   "oracle_team": round(spell_f1(gt, dec(oracle(pr, gt, "team")))[0], 4),
                   "frame_dead_as_live": round(float(np.mean(live_pr[graded & (gt == C.D)])), 3),
                   "frame_live_as_dead": round(float(np.mean(pred[graded & live_gt] == C.D)), 3),
                   "frame_team_err_both_live": round(float(np.mean(pred[live_gt & live_pr] != gt[live_gt & live_pr])), 3),
                   "by_length": by_length(gt, pred)}
            res["rows"].append(row)
            print(f"{src:<8} {m:<10} score {row['score']:.4f} argmax {row['argmax']:.4f} oracle_dead {row['oracle_dead']:.4f} "
                  f"oracle_team {row['oracle_team']:.4f} | spells gt {n_gt} pred {n_pred} | IoU {per_iou} | "
                  f"D->live {row['frame_dead_as_live']} live->D {row['frame_live_as_dead']} team {row['frame_team_err_both_live']}",
                  flush=True)
    agg = {}
    for src in ("dfl_v1", "dfl_v2b", "metrica"):
        rs = [r for r in res["rows"] if r["src"] == src]
        agg[src] = {k: round(float(np.mean([r[k] for r in rs])), 4)
                    for k in ("score", "argmax", "oracle_dead", "oracle_team", "n_gt", "n_pred", "frame_dead_as_live",
                              "frame_live_as_dead", "frame_team_err_both_live")}
        bins = {}
        for key in rs[0]["by_length"]:
            cells = [r["by_length"][key] for r in rs]
            bins[key] = {"n_gt": sum(c["n_gt"] for c in cells), "n_pred": sum(c["n_pred"] for c in cells),
                         "recall@0.5": round(float(np.mean([c["recall@0.5"] for c in cells if c["recall@0.5"] is not None])), 3),
                         "precision@0.5": round(float(np.mean([c["precision@0.5"] for c in cells
                                                               if c["precision@0.5"] is not None])), 3)}
        agg[src]["by_length"] = bins
        print(src, json.dumps(agg[src]))
    res["mean"] = agg
    Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
