"""Short possessions weighted up in the team GRU's loss. DFL LOMO, team call only, flat switch, 3 seeds. GPU.

    python analysis/p7s_short_weight.py      # -> eda_out/reports/P7s_short_weight.md

P7r: most of what v9 misses are possessions under 5 s, which the GRU never hands to the other team. The
training weight of frames inside a true possession shorter than SHORT_S, and within PAD frames of it, is
multiplied by K (K = 1 is production). All with the team swap. Reported: spellF1, and the count of
predicted possessions against the truth's.
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
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SHORT_S, PAD = 5.0, 12
KS = (1.0, 2.0, 4.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P7s_short_weight.md")


def short_mask(labels):
    out = np.zeros(len(labels), bool)
    for a, b, _ in pm.possession_spells(labels, pm.valid_runs(labels)):
        if (b - a) / 25 < SHORT_S:
            out[max(0, a - PAD):b + PAD] = True
    return out


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    S = [short_mask(m["labels"]) for m in ms]
    print("features done; short share", round(float(np.mean([s[m["half"] > 0].mean() for s, m in zip(S, ms)])), 3), flush=True)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        runs = pm.valid_runs(m["labels"])
        r = {"match": m["name"], "n_gt": len(pm.possession_spells(m["labels"], runs))}
        for K in KS:
            def it(mm, F, s):
                w = np.isin(mm["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(s, K, 1.0).astype(np.float32)
                return (F, (mm["labels"] == pm.H).astype(np.float32), w, mm["half"] > 0)
            p = fit_predict([it(ms[k], T[k], S[k]) for k in tr] + [it(sw[k], Ts[k], S[k]) for k in tr], (T[t], g))
            c = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r["k%g" % K] = pm.spell_scores(m["labels"], c, runs)["mean_f1"]
            r["n%g" % K] = len(pm.possession_spells(c, runs))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7s: kısa possession'lara kayıpta ağırlık (DFL, maç dışarıda bırakarak, sadece takım, takım değişimiyle, 3 tohum)", "",
             "`analysis/p7s_short_weight.py`, SHORT_S = {}, PAD = {}.".format(SHORT_S, PAD), "",
             "| maç | gerçek possession | " + " | ".join("K = {:g}".format(K) for K in KS) + " |", "|---|---|" + "---|" * len(KS)]
    lines += ["| {} | {} | ".format(r["match"], r["n_gt"]) + " | ".join("{:.3f} ({})".format(r["k%g" % K], r["n%g" % K]) for K in KS) + " |" for r in rows]
    lines += ["| ortalama | {:.0f} | ".format(df.n_gt.mean()) + " | ".join("{:.3f} ({:.0f})".format(df["k%g" % K].mean(), df["n%g" % K].mean()) for K in KS) + " |", ""]
    lines += ["K = {:g}: K = 1'e fark {:+.3f}, {} / 7 maçta önde.".format(K, (df["k%g" % K] - df.k1).mean(), int((df["k%g" % K] > df.k1).sum())) for K in KS[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
