"""The crop evidence's scale again, for the v9 team GRU (team swap) and dead ball v6. GT-free, our match.

    python analysis/p7p_crop_scale_v9.py      # -> eda_out/reports/P7p_crop_scale_v9.md

P6f's ball-dropout test (analysis/p6f_fuse_crops.py, same blocks, same drop): the reference is the v9 GRU's
call with the ATD ball; each scale's call with the ball removed in stretches is scored against it. The
switch cost is v9's per-frame one. Production scale: mtah.possessor.SCALE.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.possessor import SCALE, calibrate, crop_evidence, evidence_logodds  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p3e_ball_pseudolabels import BLOCK  # noqa: E402
from p6f_fuse_crops import SCALES, drop_ball, logit  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7p_crop_scale_v9.md")


def main():
    from mtah.data import load_tactical
    from mtah.grumodel import external_gru, predict, switch_costs
    from mtah.match import from_tactical
    from p4_common import dead_v6
    m = from_tactical()
    m["hom"] = load_tactical()["hom"]
    dead = dead_v6()
    g = m["half"] > 0
    alive = g & ~dead
    val = np.flatnonzero(g & ((np.arange(len(g)) // BLOCK) % 5 == 4))
    t, p = crop_evidence(m)
    E = evidence_logodds(t, p, calibrate(m, t, p, val))
    gm, bm = external_gru(swap=True), external_gru(target="boundary", swap=True)
    md, drop = drop_ball(m, val)
    ref = decode(predict(gm, m), alive, switch_costs(bm, m))
    based, cost_d = logit(predict(gm, md)), switch_costs(bm, md)
    vmask = np.zeros(len(g), bool)
    vmask[val] = True
    gt_ref = np.where(vmask, ref, pm.X).astype(np.int8)
    rows = []
    for s in SCALES:
        c = np.where(vmask, decode(1 / (1 + np.exp(-(based + s * E))), alive, cost_d), pm.X).astype(np.int8)
        rows.append((s, pm.spell_scores(gt_ref, c, pm.valid_runs(gt_ref))["mean_f1"], np.mean(c[drop & alive] == ref[drop & alive])))
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r[1])[0]
    lines = ["# P7p: kırpıntı kanıtının ölçeği, v9 GRU'su ile (GT'siz top bırakma testi, bizim maç)", "", "`analysis/p7p_crop_scale_v9.py`. Üretimdeki ölçek {:g}.".format(SCALE), "",
             "| ölçek | spellF1 (referansa göre) | silinen oyun içi karelerde uyum |", "|---|---|---|"]
    lines += ["| {:g} | {:.3f} | {:.1%} |".format(s, r, a) for s, r, a in rows]
    lines += ["", "En iyi ölçek {:g}.".format(best)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
