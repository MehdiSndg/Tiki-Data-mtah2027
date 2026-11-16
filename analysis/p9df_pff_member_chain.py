"""P9df: P9dd's PFF-trained BigGRU (cache/oof_pff/oof_<match>.npz: pp) in v19c's out-of-fold chain, DFL 7 + Metrica 3,
1.6 s / 3 s. The member never saw DFL or Metrica; it reads the same degraded copies as the other members. Arms: ours;
the member alone (our dead ball and switch costs); ours with the member as a seventh member (probability mean) and at
log-odds weight 0.25 / 0.5; the same on top of Mali's team at 0.25 (P9de). Rule (as every member): adopt when DFL is
ahead at 1.6 s and not behind at 3 s, and Metrica not behind at either.

    python analysis/p9df_pff_member_chain.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    rows = {}
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        if not os.path.exists(C("oof_pff", n)):
            print("no member prediction for", n)
            continue
        z, big, t1, d3, q = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f, C("oof_pff", n)))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        assert len(q["pp"]) == len(g) and (q["g"] == g).all(), name
        pp = np.nan_to_num(q["pp"].astype(np.float64), nan=0.5)
        l3 = merged(lab, 3.0)
        sw = HI * (1 - z["b"]) + LO * z["b"]
        p = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        chain = lambda team: np.array([pm.spell_scores(L, np.where(g, decode(team, g & ~dead, sw), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        live = g & np.isin(lab, (pm.H, pm.A))
        acc = lambda x: np.mean((x[live] > 0.5) == (lab[live] == pm.H))
        r = {"ours": chain(p), "member alone": chain(pp), "ours + member as 7th": chain((6 * p + pp) / 7),
             "ours + member w 0.25": chain(sg(0.75 * lg(p) + 0.25 * lg(pp))), "ours + member w 0.5": chain(sg(0.5 * lg(p) + 0.5 * lg(pp)))}
        M = theirs("mehmet_ali_topkara", name, g)
        if M is not None:
            r["ours + mali 0.25"] = chain(sg(0.75 * lg(p) + 0.25 * lg(M["team"])))
            r["ours 0.6 + mali 0.2 + member 0.2"] = chain(sg(0.6 * lg(p) + 0.2 * lg(M["team"]) + 0.2 * lg(pp)))
        for k, v in r.items():
            rows.setdefault(k, {})[name] = v
        print(name, "frame team accuracy ours {:.4f} member {:.4f} mean {:.4f}; agreement {:.4f}".format(acc(p), acc(pp), acc((6 * p + pp) / 7), np.mean((p[live] > 0.5) == (pp[live] > 0.5))), flush=True)
    base = rows["ours"]
    print("\narm | DFL 1.6 s / 3 s | gain (ahead) | Metrica 1.6 s / 3 s | gain (ahead)")
    for k, v in rows.items():
        out = [k]
        for src in ("DFL", "metrica"):
            ms = [m for m in v if src in m]
            a, b = np.array([v[m] for m in ms]), np.array([base[m] for m in ms])
            out.append("{:.4f} / {:.4f} | {:+.4f} / {:+.4f} ({} / {})".format(*a.mean(0), *(a - b).mean(0), int(((a - b)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
