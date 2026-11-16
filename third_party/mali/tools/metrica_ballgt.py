r"""Metrica referee with the dead flag taken from the ball channel (diagnostic, no training, external data only).

Metrica's open tracking leaves the ball empty while it is out of play (P(ball missing | live) 0.00-0.03,
P(dead | ball missing) 0.93-0.96 against our event-derived labels). The competition's ground truth takes its
dead flag from the ball-status channel of the same provider's tracking, so the ball channel is the closer
analogue; our event-derived dead starts at the START of a BALL OUT / FAULT event (the kick), earlier than the
ball channel (median +1...+12 frames, p75 +35...+44).

alt labels: D = ball missing (gaps <= --bridge frames inside a live stretch are bridged), team = event label,
frames the events call dead / loose while the ball is still tracked keep the previous team (next team at a
run's start).  Scores the saved official blend against both label sets.

  .venv\Scripts\python.exe tools/metrica_ballgt.py --run ev_ext_met
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

from oracle_split import fill_team, runs_of  # noqa: E402
from short_spell_diag import build_refs  # noqa: E402
from timing_union import decode, offsets  # noqa: E402


def alt_labels(gt, ball_missing, runs, bridge):
    miss = ball_missing.copy()
    for a, b in runs_of(miss):
        if b - a <= bridge and a > 0 and b < len(gt) and gt[a - 1] in (0, 1) and gt[min(b, len(gt) - 1)] in (0, 1):
            miss[a:b] = False
    team = np.where((gt == 0) | (gt == 1), gt, 3)
    team = np.where(miss, 3, team)
    # previous team first (forward fill), next team only at a run's start
    out = np.full(len(gt), 3, np.int8)
    for a, b in runs:
        s = team[a:b]
        ok = s != 3
        idx = np.maximum.accumulate(np.where(ok, np.arange(b - a), -1))
        nxt = np.minimum.accumulate(np.where(ok, np.arange(b - a), b - a)[::-1])[::-1]
        use = np.where(idx >= 0, idx, np.minimum(nxt, b - a - 1))
        out[a:b] = np.where(ok.any(), s[use], 3)
    out = np.where(miss, 2, out).astype(np.int8)
    out[gt == 3] = np.where(miss[gt == 3], 2, out[gt == 3])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    ap.add_argument("--bridge", type=int, nargs="+", default=[0, 5, 12, 25])
    args = ap.parse_args()
    refs, dcfg, w, _, _ = build_refs(args.run, args.decoder_config)
    res = {}
    for ref in refs:
        view, m, prep, gt = ref[:4]
        if view != "metrica":
            continue
        pred = decode(ref, ref[4], ref[6], dcfg, w)
        per = prep["data"].period > 0
        miss = np.isnan(prep["data"].ball[:, 0]) & per
        base = pm.evaluate(gt, pred)
        res[m] = {"event": float(base["spell_f1_mean"])}
        line = f"{m} event-GT {base['spell_f1_mean']:.4f} macro {base['macro_f1']:.4f}"
        for br in args.bridge:
            alt = alt_labels(gt, miss, prep["runs"], br)
            alt[~per] = 3
            ev = pm.evaluate(alt, pred)
            st = dict(spell=float(ev["spell_f1_mean"]), macro=float(ev["macro_f1"]),
                      dead_share=float(np.mean(alt[per] == 2)), dead_segs=len(runs_of(alt == 2)),
                      d_to_live=float(np.mean(pred[alt == 2] != 2)),
                      live_to_d=float(np.mean(pred[(alt == 0) | (alt == 1)] == 2)))
            pt = fill_team(pred, prep["runs"])
            od = np.where(alt == 2, 2, np.where(pt >= 0, pt, pred)).astype(np.int8)
            st["oracle_dead"] = float(pm.evaluate(alt, od)["spell_f1_mean"])
            off = offsets(alt, pred)
            st["end_abs_median"] = float(np.median(np.abs(off["end"][0]))) if len(off["end"][0]) else None
            st["end_matched"] = len(off["end"][0]) / max(off["end"][1], 1)
            res[m][f"bridge_{br}"] = st
            line += (f"\n   ball-GT bridge {br:>2}: spell {st['spell']:.4f} macro {st['macro']:.4f} oracle-dead {st['oracle_dead']:.4f} "
                     f"| D share {st['dead_share']:.3f} segs {st['dead_segs']} | D->live {st['d_to_live']:.3f} live->D {st['live_to_d']:.3f} "
                     f"| end matched {st['end_matched']:.2f} |off| {st['end_abs_median']}")
        print(line, flush=True)
    for br in args.bridge:
        print(f"bridge {br}: mean spell {np.mean([res[m][f'bridge_{br}']['spell'] for m in res]):.4f} "
              f"(event-GT {np.mean([res[m]['event'] for m in res]):.4f})")
    (OUTPUTS / f"metrica_ballgt_{args.run}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
