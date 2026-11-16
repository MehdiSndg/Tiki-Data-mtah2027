"""P9ds: a ball-blind team channel (the friend's v21 "fourth channel that never sees the ball"): the production team
features computed with the ball removed (22 player-only columns: team centroids, spreads, velocities, counts), GRU and
TCN, P9a's leave-one-match-out over DFL 7 + Metrica 3 (training copies = P9a's old copies + team swap, short-spell
weight 2; test = P9a's test copy with the ball removed), seeds 3-5. Writes cache/oof_blind/oof_<match>.npz (pb, pbt).
GPU.

    .venv/bin/python analysis/p9ds_ballblind.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "cache", "oof_blind")
SEEDS = (3, 4, 5)


def blind(m):
    b = dict(m)
    b["bx"] = np.full_like(m["bx"], np.nan)
    b["by"] = np.full_like(m["by"], np.nan)
    return b


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    To, Tso = [features(blind(m)) for m in old], [features(blind(m)) for m in so]
    keep = np.flatnonzero(np.nanstd(np.concatenate(To), axis=0) > 1e-9)
    To, Tso = [t[:, keep].astype(np.float32) for t in To], [t[:, keep].astype(np.float32) for t in Tso]
    print("features done", len(keep), flush=True)
    Sh = [G.short_mask(m["labels"]) for m in old]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    for t, (r, k) in enumerate(raws):
        m = degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)
        path = os.path.join(OUT, "oof_{}.npz".format(m["name"].split("+")[0]))
        if os.path.exists(path):
            continue
        g = m["half"] > 0
        Tt = features(blind(m))[:, keep].astype(np.float32)
        ks = [j for j in range(len(old)) if j != t]
        data = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(Sh[j], 2.0, 1.0).astype(np.float32), mm["half"] > 0)
                for j in ks for mm, F in ((old[j], To[j]), (so[j], Tso[j]))]
        pb = fit_predict(data, (Tt, g), seeds=SEEDS)
        pbt = fit_predict(data, (Tt, g), make_net=G._tcn, seeds=SEEDS)
        alive = np.isin(m["labels"], (pm.H, pm.A)) & g
        acc = lambda p: np.mean((p[alive] > 0.5) == (m["labels"][alive] == pm.H))
        np.savez_compressed(path, pb=pb, pbt=pbt, g=g, labels=m["labels"])
        print(m["name"], "live team accuracy gru {:.3f} tcn {:.3f}".format(acc(pb), acc(pbt)), flush=True)


if __name__ == "__main__":
    main()
