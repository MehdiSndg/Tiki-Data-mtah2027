"""Which judge resembles the hidden full-match ground truth? The organisers publish the four reference baselines'
full-match scores (EVALUATION.md); the second teammate tuned their proxy's label merge until the baselines reproduced
them (RMSE 0.003, ranking exact). Here: starter/baseline.py's algorithm (nearest tracked player to the visible ball,
centred majority vote of +-w frames, D where nobody) run on our carried-dead-ball degraded DFL and Metrica, scored
against labels merged at MERGES seconds; the RMSE of the eight numbers (macroF1 and spellF1 of four baselines) to the
published ones says which merge (and judge) is closest. No ground truth of BAR-ATH is used. CPU, light.

    python analysis/p9k_baseline_calibration.py      # -> eda_out/reports/P9k_baseline_calibration.md
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample, merge_short  # noqa: E402

OFFICIAL = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}
MERGES = tuple(float(x) for x in os.environ.get("P9K_MERGES", "0,0.8,1.6,2.0,2.4,3.0").split(","))
CARRY = tuple(float(x) for x in os.environ.get("P9K_CARRY", "0,0.25,0.5").split(","))
JITTER = tuple(float(x) for x in os.environ.get("P9K_JITTER", "1").split(","))     # x jitter_share and drift_p
OUT = os.path.join(ROOT, "eda_out", "reports", "P9k_baseline_calibration{}.md".format(os.environ.get("P9K_TAG", "")))


def nearest_team(m):
    L, W = m["pitch"]
    dx = (m["px"].astype(float) - m["bx"].astype(float)[:, None]) * L
    dy = (m["py"].astype(float) - m["by"].astype(float)[:, None]) * W
    d = np.hypot(dx, dy)
    ok = np.isfinite(m["bx"]) & np.isfinite(d).any(axis=1)
    out = np.full(len(d), -1, np.int8)
    out[ok] = np.asarray(m["slot_team"])[np.nanargmin(np.where(np.isfinite(d[ok]), d[ok], np.inf), axis=1)]
    return out


def smooth(codes, w):
    """starter/baseline.smooth, on codes: H / A / D."""
    if w <= 0:
        return np.where(codes == 0, pm.H, np.where(codes == 1, pm.A, pm.D)).astype(np.int8)
    hc, ac = np.r_[0, np.cumsum(codes == 0)], np.r_[0, np.cumsum(codes == 1)]
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - w), np.minimum(len(codes), i + w + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    return np.where(h + a == 0, pm.D, np.where(h >= a, pm.H, pm.A)).astype(np.int8)


def merged(labels, half, s):
    out = labels.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        out[idx] = merge_short(labels[idx], s)
    return out


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    rows = []
    for carry, jit in [(c, j) for c in CARRY for j in JITTER]:
        rt = lambda k: dict(rates_for(k), smooth_frames=25, dead_carry_p=carry, jitter_share=rates_for(k)["jitter_share"] * jit, drift_p=rates_for(k)["drift_p"] * jit)
        ms = [degrade(from_dfl(i, merge_s=0.0), rt(i), seed=0) for i in ids]
        ms += [degrade(from_metrica_sample(g), rt("metrica"), seed=0) for g in (1, 2, 3)]
        preds = [{w: smooth(nearest_team(m), w) for w in OFFICIAL} for m in ms]
        for mg in MERGES:
            for src, sel in (("DFL", slice(0, len(ids))), ("Metrica", slice(len(ids), None))):
                res = {w: [] for w in OFFICIAL}
                for m, p in list(zip(ms, preds))[sel]:
                    gt = merged(m["labels"], m["half"], mg) if src == "DFL" else m["labels"]
                    for w in OFFICIAL:
                        e = pm.evaluate(gt, p[w], bootstrap_n=0)
                        res[w].append((e["macro_f1"], e["spell_f1_mean"]))
                vals = {w: np.mean(res[w], axis=0) for w in OFFICIAL}
                err = np.array([vals[w] - np.array(OFFICIAL[w]) for w in OFFICIAL])
                order = [w for w, _ in sorted(vals.items(), key=lambda kv: -kv[1][1])] == [w for w, _ in sorted(OFFICIAL.items(), key=lambda kv: -kv[1][1])]
                r = {"carry": carry, "jitter": jit, "merge": mg if src == "DFL" else "-", "src": src, "rmse_all": float(np.sqrt(np.mean(err ** 2))),
                     "rmse_spell": float(np.sqrt(np.mean(err[:, 1] ** 2))), "rmse_macro": float(np.sqrt(np.mean(err[:, 0] ** 2))), "spell_order_ok": order}
                for w in OFFICIAL:
                    r["s{}".format(w)], r["m{}".format(w)] = vals[w][1], vals[w][0]
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["carry", "jitter", "merge", "src"])
    lines = ["# P9k: resmi baseline skorlarına göre hakem ayarı (DFL / Metrica, bozulmuş, tohum 0)", "", "`analysis/p9k_baseline_calibration.py`.", "",
             "Resmi tam maç: spellF1 raw / 25 / 50 / 125 = 0.049 / 0.200 / 0.243 / 0.224; macroF1 0.653 / 0.675 / 0.672 / 0.636.", "",
             "| taşıma | titreme × | birleştirme | kaynak | spellF1 raw / 25 / 50 / 125 | macroF1 raw / 25 / 50 / 125 | RMSE (8) | RMSE spell | sıra |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values("rmse_all").iterrows():
        lines.append("| {} | {} | {} | {} | {:.3f} / {:.3f} / {:.3f} / {:.3f} | {:.3f} / {:.3f} / {:.3f} / {:.3f} | {:.4f} | {:.4f} | {} |".format(
            r.carry, r.jitter, r.merge, r.src, r.s0, r.s25, r.s50, r.s125, r.m0, r.m25, r.m50, r.m125, r.rmse_all, r.rmse_spell, "✓" if r.spell_order_ok else "✗"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
