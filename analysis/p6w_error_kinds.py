"""Where does the GRU team call lose spellF1? Error kinds on DFL LOMO (team call only). GPU.

    python analysis/p6w_error_kinds.py      # -> eda_out/reports/P6w_error_kinds.md

GRU as P6s (1 seed), Viterbi at mtah.grumodel.SWITCH, dead ball from the labels. At IoU 0.5 (the middle of
the scorer's 0.3-0.7): ground-truth possessions recalled per duration bin, and each missed one put in one
kind -- no same-team prediction overlaps it (wrong team), the best overlapping prediction is more than
twice as long (merged into a neighbour), less than half as long (split), or about the right length but
shifted. Predicted possessions without a match per duration bin.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p6v_gru_dead import gru_fit_predict  # noqa: E402

BINS = (0, 2, 5, 10, 20, 1e9)
IOU = 0.5
OUT = os.path.join(ROOT, "eda_out", "reports", "P6w_error_kinds.md")


def bin_of(s):
    return next(i for i in range(len(BINS) - 1) if BINS[i] <= s < BINS[i + 1])


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    T = [features(m) for m in ms]
    gt_rows, pred_rows = [], []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
        p = gru_fit_predict([(T[k], (ms[k]["labels"] == pm.H).astype(np.float32), np.isin(ms[k]["labels"], (pm.H, pm.A)).astype(np.float32),
                              ms[k]["half"] > 0) for k in tr], (T[t], g), seeds=(0,))
        c = decode(p, alive, G.SWITCH)
        runs = pm.valid_runs(m["labels"])
        gs, ps = pm.possession_spells(m["labels"], runs), pm.possession_spells(c, runs)
        ov = pm.spell_overlaps(gs, ps)
        flags = pm.matched_gt_flags(ov, len(gs), IOU)
        matched_pred, used = set(), set()
        for iou, i, j in ov:
            if iou < IOU:
                break
            if i in used or j in matched_pred:
                continue
            used.add(i)
            matched_pred.add(j)
        best = {}
        for iou, i, j in ov:
            best.setdefault(i, j)
        for i, (a, b, k) in enumerate(gs):
            dur = (b - a) / 25
            if flags[i]:
                kind = "bulundu"
            elif i not in best:
                kind = "yanlış takım"
            else:
                pa, pb, _ = ps[best[i]]
                r = (pb - pa) / max(1, b - a)
                kind = "komşuyla birleşmiş" if r > 2 else "bölünmüş" if r < 0.5 else "kaymış"
            gt_rows.append({"match": m["name"], "bin": bin_of(dur), "kind": kind})
        for j, (a, b, k) in enumerate(ps):
            pred_rows.append({"match": m["name"], "bin": bin_of((b - a) / 25), "matched": j in matched_pred})
        print(m["name"], "gt", len(gs), "pred", len(ps), "recall", round(float(flags.mean()), 3), flush=True)
    g = pd.DataFrame(gt_rows)
    p = pd.DataFrame(pred_rows)
    names = ["{}–{} s".format(BINS[i], int(BINS[i + 1]) if BINS[i + 1] < 1e8 else "∞") for i in range(len(BINS) - 1)]
    tab = g.groupby(["bin", "kind"]).size().unstack(fill_value=0)
    kinds = [k for k in ("bulundu", "yanlış takım", "komşuyla birleşmiş", "bölünmüş", "kaymış") if k in tab.columns]
    lines = ["# P6w: GRU'nun hata türleri (DFL, maç dışarıda bırakarak, IoU 0.5)", "", "`analysis/p6w_error_kinds.py`. Sadece takım kararı.", "",
             "## Gerçek possession'lar", "", "| süre | sayı | " + " | ".join(kinds) + " |", "|---|---|" + "---|" * len(kinds)]
    for b, r in tab.iterrows():
        n = int(r.sum())
        lines.append("| {} | {} | ".format(names[b], n) + " | ".join("{} ({:.0%})".format(int(r.get(k, 0)), r.get(k, 0) / n) for k in kinds) + " |")
    lines += ["", "## Tahmin edilen possession'lar", "", "| süre | sayı | eşleşmeyen |", "|---|---|---|"]
    for b, r in p.groupby("bin"):
        lines.append("| {} | {} | {} ({:.0%}) |".format(names[b], len(r), int((~r.matched).sum()), (~r.matched).mean()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
