"""P9dy: the ball-blind channel (P9ds, cache/oof_blind: GRU + TCN on player-only features) fused into v26's out-of-fold
probabilities (six members + Mali's team at 0.25; three dead models) at log-odds weight w, semi-Markov decoder,
DFL 7 + Metrica 3, our labels at 1.6 s / 3 s.

    .venv/bin/python analysis/p9dy_blind_fuse.py

Registered before the run (13 Nov): w in {0.1, 0.2, 0.3}, picked per match on the other nine (mean of both referees);
adopt only if nested DFL is ahead at 1.6 s and neither DFL at 3 s nor Metrica at either referee is behind.
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
from p9l_rescore_calibrated import merged  # noqa: E402

WS = (0.1, 0.2, 0.3)


def main():
    res = {}
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        if not os.path.exists(C("oof_blind", n)):
            print("missing", n)
            continue
        z, big, t1, d3, bl = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f, C("oof_blind", n)))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        assert np.array_equal(bl["labels"], lab)
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        M = theirs("mehmet_ali_topkara", name, g)
        base = 0.75 * lg(p) + 0.25 * lg(M["team"])
        dead = (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
        pb = np.nan_to_num((bl["pb"] + bl["pbt"]) / 2, nan=0.5)
        alive = np.isin(lab, (pm.H, pm.A)) & g
        acc = lambda q: np.mean((q[alive] > 0.5) == (lab[alive] == pm.H))
        corr = np.corrcoef((sg(base)[alive] > 0.5) != (lab[alive] == pm.H), (pb[alive] > 0.5) != (lab[alive] == pm.H))[0, 1]
        sc = lambda team: [pm.spell_scores(L, np.where(g, sm.decode_halves(team, dead, g, b), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        r = {"base": sc(sg(base))}
        for w in WS:
            r[w] = sc(sg((1 - w) * base + w * lg(pb)))
        res[name] = r
        print(name, "acc fused {:.3f} blind {:.3f}, error correlation {:.2f}".format(acc(sg(base)), acc(pb), corr), {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    names = sorted(res)
    base = np.array([res[m]["base"] for m in names])
    arms = {w: np.array([res[m][w] for m in names]) for w in WS}
    arms["nested"] = np.array([res[m][max(WS, key=lambda w: np.mean([np.mean(res[o][w]) - np.mean(res[o]["base"]) for o in names if o != m]))] for m in names])
    for k, a in arms.items():
        out = [str(k)]
        for src in ("DFL", "metrica"):
            ix = [i for i, m in enumerate(names) if src in m]
            if not ix:
                continue
            d = a[ix] - base[ix]
            out.append("{} {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *a[ix].mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), len(ix)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
