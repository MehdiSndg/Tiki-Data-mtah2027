"""GT-free ablation of the broadcast dead-ball rule (first research report, item 11; in the chain since v3, never
ablated): on our match, v16's dead ball with and without mtah.deadball.broadcast_still_dead (the classifier + GRU
blocks and the not-played periods as in p4_common.dead_v5, then close_short_gaps as dead_v7), the team call decoded
from v16's own probabilities and switch costs (MTAH_DUMP), both chains written for analysis/p6m_compare_subs.py. Also
how many rule-only blocks hold a Smart Tagging set piece. No labels of our match are read. CPU (HistGB retrained),
the dead GRU from models/.

    python analysis/p9aj_broadcast_rule.py cache/v16_dump.npz      # -> eda_out/reports/P9aj_broadcast_rule.md, eda_out/p9aj_*.csv
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
import p4_common as P  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import broadcast_still_dead, features_v2  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.teammodel import decode  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P9aj_broadcast_rule.md")
LETTERS = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def main(dump):
    m = from_tactical()
    g = m["half"] > 0
    F = features_v2(m)
    clf = P.dead_classifier(featurize=features_v2, extra_rates=P.CARRY_RATES, draws=P.DRAWS_V6)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(F[g])[:, 1]
    gm = P.dead_gru(draws=P.DRAWS_V6)
    q = np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0)
    fr = load_frames()
    model = (to_blocks((p + np.where(g, q, 0)) / 2) | np.isin(fr["period"].values, NOT_PLAYED)) & g
    rule = broadcast_still_dead(fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    with_rule, without = P.close_short_gaps(model | rule, g), P.close_short_gaps(model, g)
    u = np.load(dump)
    same = np.mean(with_rule == (u["dead"] & u["g"]))
    out = {}
    for name, dead in (("with", with_rule), ("without", without)):
        codes = np.where(g, decode(u["p"], g & ~dead, u["switch"]), pm.X).astype(np.int8)
        path = os.path.join(ROOT, "eda_out", "p9aj_{}.csv".format(name))
        pd.DataFrame({"frame": np.arange(len(codes)), "label": [LETTERS[c] for c in codes]}).to_csv(path, index=False)
        out[name] = (dead, codes, path)
    only = with_rule & ~without
    bl = [(a, b) for a, b in P_blocks(only)]
    ev = [e for e in load_smart_tagging.load() if e["code"] == "SET PIECES"]
    hold = sum(any(a - 5 * FPS <= e["start_frame"] <= b + 5 * FPS for e in ev) for a, b in bl)
    gmin = g.sum() / FPS / 60
    lines = ["# P9aj: yayın ölü top kuralının GT'siz ablasyonu (bizim maç, v16)", "", "`analysis/p9aj_broadcast_rule.py`.", "",
             "- yeniden kurulan 'kurallı' ölü top v16 dökümüyle kare uyumu: %{:.2f}".format(100 * same),
             "- kuralın tek başına eklediği: {} kare ({:.1f} dk), {} blok; bunların {}'inde ±5 s içinde Smart Tagging duran topu".format(
                 int(only.sum()), only.sum() / FPS / 60, len(bl), hold),
             "- ölü blok / 90 dk: kurallı {:.1f}, kuralsız {:.1f}".format(len(P_blocks(with_rule)) / gmin * 90, len(P_blocks(without)) / gmin * 90)]
    for name, (dead, codes, _) in out.items():
        lines.append("- {}: possession / 90 dk {:.0f}".format(name, len(pm.possession_spells(codes, pm.valid_runs(codes))) / gmin * 90))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def P_blocks(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


if __name__ == "__main__":
    main(*sys.argv[1:])
