"""P9em: inside the team call's loss (P9el: oracle team +0.21 on DFL). On the realistic copies, v28's chain:
  * oracle boundaries: the labels' spell boundaries kept, each spell's team = the majority of the chain's team
    probability inside it (who, given when) ; and the reverse, oracle team inside the chain's own segments (when, given who)
  * recall of the labels' spells by duration (< 2 s, 2-5 s, 5-15 s, > 15 s) at IoU 0.5, and the predicted spells' count
"On these copies" only.

    /home/mehdi/venvs/mali/bin/python analysis/p9em_team_loss.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402

BINS = ((0, 2), (2, 5), (5, 15), (15, 1e9))


def spells(c):
    return pm.possession_spells(c, pm.valid_runs(c))


def main():
    acc = {k: [] for k in ("chain", "who|when", "when|who")}
    rec = {k: np.zeros((len(BINS), 2)) for k in ("chain",)}
    npred = []
    for f in sorted(glob.glob(C("p9ea", "oof_*.npz"))):
        n = os.path.basename(f)
        name = n[4:-4]
        if "DFL" not in name:
            continue
        z, big, t1, d3, e = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n), f))
        g, lab = z["g"], z["labels"]
        b = np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        M = theirs("mehmet_ali_topkara", name, g)
        team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
        dead = ((e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3).astype(np.float64)
        chain = np.where(g, sm.decode_halves(team, dead, g, b), pm.X).astype(np.int8)
        # who | when: the labels' spells and dead stretches, each spell's team from the chain's probability
        who = lab.copy()
        for a, c, _ in spells(lab):
            who[a:c] = pm.H if np.mean(lg(team[a:c])) > 0 else pm.A
        # when | who: the chain's segments, each predicted spell's team = the labels' majority team inside it
        when = chain.copy()
        for a, c, _ in spells(chain):
            seg = lab[a:c]
            live = seg[np.isin(seg, (pm.H, pm.A))]
            if len(live):
                when[a:c] = pm.H if np.mean(live == pm.H) >= 0.5 else pm.A
        for k, c in (("chain", chain), ("who|when", who), ("when|who", when)):
            acc[k].append(pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"])
        P = spells(chain)
        npred.append((len(P), len(spells(lab))))
        for a, c, t in spells(lab):
            d = (c - a) / 25
            i = next(j for j, (lo, hi) in enumerate(BINS) if lo <= d < hi)
            best = 0.0
            for pa, pc, pt in P:
                if pt == t and pc > a and pa < c:
                    inter = min(c, pc) - max(a, pa)
                    best = max(best, inter / ((c - a) + (pc - pa) - inter))
            rec["chain"][i] += (best >= 0.5, 1)
        print(name, {k: round(v[-1], 4) for k, v in acc.items()}, flush=True)
    print("DFL 7, realistic copies, 1.6 s labels: chain {:.4f} | right boundaries, our team per spell {:.4f} | our segments, right team per segment {:.4f}".format(
        *[np.mean(acc[k]) for k in ("chain", "who|when", "when|who")]))
    r = rec["chain"]
    print("label spells found at IoU 0.5 by duration:", "  ".join("{}-{}s {:.0f}/{:.0f} ({:.2f})".format(lo, hi if hi < 1e9 else "+", r[i, 0], r[i, 1], r[i, 0] / r[i, 1]) for i, (lo, hi) in enumerate(BINS)))
    print("predicted / label spells per match:", np.array(npred).sum(0))


if __name__ == "__main__":
    main()
