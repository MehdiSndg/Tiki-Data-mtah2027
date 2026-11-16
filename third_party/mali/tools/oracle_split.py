r"""Where is the remaining spellF1 lost on the external referees? (diagnostic, no training, external data only)

For the official blend (saved nets + Metrica booster, w 0.25, main_ext_met_dt decoder) on DFL-3 (v1 / v2b
corruption) and Metrica 1-3:

  pred          the decoded prediction
  oracle dead   true dead/alive mask, our team inside the live frames (nearest predicted team fills frames we
                called dead)                                   -> what a perfect dead-ball model would give
  oracle team   our dead/alive mask, true team inside our live frames (nearest true team fills true-dead frames)
                                                               -> what a perfect team model would give
  dead segments true vs predicted count, share of true segments found (IoU >= 0.5), frame confusion
                live->D / D->live, and F1 at IoU 0.3 / 0.5 / 0.7 of the possessions.

  .venv\Scripts\python.exe tools/oracle_split.py --run ev_ext_met
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah.paths import OUTPUTS  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

from short_spell_diag import build_refs  # noqa: E402
from timing_union import decode  # noqa: E402


def fill_team(team_src, runs):
    """Per frame the nearest H/A value of team_src inside its run (forward fill, then backward)."""
    out = np.full(len(team_src), -1, np.int8)
    for a, b in runs:
        s = team_src[a:b]
        ok = (s == 0) | (s == 1)
        if not ok.any():
            continue
        idx = np.where(ok, np.arange(b - a), -1)
        f = np.maximum.accumulate(idx)
        nxt = np.where(ok, np.arange(b - a), b - a)
        bk = np.minimum.accumulate(nxt[::-1])[::-1]
        use = np.where(f >= 0, f, np.minimum(bk, b - a - 1))
        both = (f >= 0) & (bk < b - a)
        use = np.where(both & ((bk - np.arange(b - a)) < (np.arange(b - a) - f)), bk, use)
        out[a:b] = s[use]
    return out


def runs_of(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def seg_found(true, pred, thr=0.5):
    n = 0
    for a, b in true:
        best = 0.0
        for c, d in pred:
            if d <= a:
                continue
            if c >= b:
                break
            best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
        n += best >= thr
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    refs, dcfg, w, _, _ = build_refs(args.run, args.decoder_config)
    res = {}
    for ref in refs:
        view, m, prep, gt = ref[:4]
        pred = decode(ref, ref[4], ref[6], dcfg, w)
        graded = gt != 3
        runs = prep["runs"]
        pt, gtt = fill_team(pred, runs), fill_team(gt, runs)
        od = np.where(gt == 2, 2, np.where(pt >= 0, pt, pred)).astype(np.int8)
        od[~graded] = pred[~graded]
        ot = np.where(pred == 2, 2, np.where(gtt >= 0, gtt, pred)).astype(np.int8)
        ev = {k: pm.evaluate(gt, x) for k, x in (("pred", pred), ("oracle_dead", od), ("oracle_team", ot))}
        td, pd_ = runs_of((gt == 2)), runs_of((pred == 2) & graded)
        live, dead = graded & (gt != 2), gt == 2
        r = {k: float(v["spell_f1_mean"]) for k, v in ev.items()}
        r.update(macro=float(ev["pred"].get("macro_f1", np.nan)),
                 dead_true=len(td), dead_pred=len(pd_), dead_found=seg_found(td, pd_) / max(len(td), 1),
                 dead_pred_real=seg_found(pd_, td) / max(len(pd_), 1),
                 live_to_dead=float(np.mean(pred[live] == 2)), dead_to_live=float(np.mean(pred[dead] != 2)),
                 dead_share_true=float(dead.sum() / graded.sum()),
                 dead_share_pred=float(np.mean(pred[graded] == 2)),
                 short_dead_true=int(sum((b - a) < 75 for a, b in td)),
                 short_dead_found=seg_found([s for s in td if s[1] - s[0] < 75], pd_),
                 keys=[k for k in ev["pred"] if "iou" in k.lower() or "f1" in k.lower()][:12])
        res[f"{view}:{m}"] = r
        print(f"{view:<8}{m:<10} pred {r['pred']:.4f} | oracle dead {r['oracle_dead']:.4f} ({r['oracle_dead']-r['pred']:+.3f}) "
              f"| oracle team {r['oracle_team']:.4f} ({r['oracle_team']-r['pred']:+.3f}) | dead seg {r['dead_pred']}/{r['dead_true']} "
              f"found {r['dead_found']:.2f} real {r['dead_pred_real']:.2f} short<3s {r['short_dead_found']}/{r['short_dead_true']} "
              f"| live->D {r['live_to_dead']:.3f} D->live {r['dead_to_live']:.3f} | D share {r['dead_share_pred']:.3f}/{r['dead_share_true']:.3f}",
              flush=True)
    print("metric keys:", next(iter(res.values()))["keys"])
    for v in ("v1", "v2b", "metrica"):
        ks = [k for k in res if k.startswith(v + ":")]
        print(v, {c: round(float(np.mean([res[k][c] for k in ks])), 4) for c in ("pred", "oracle_dead", "oracle_team")})
    (OUTPUTS / f"oracle_split_{args.run}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
