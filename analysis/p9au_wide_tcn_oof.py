"""Out-of-fold wide-TCN members for a full-chain test (P9ao: the TCN with dilations up to 128, about +-20 s of context,
beat the production TCN on the team call): per test match, P9a's recipe for the TCN member and its two-head (team +
boundary) twin with dilations 1-128 and v16's 6 seeds (3-8); saved as oof_<match>.npz (pt, mt, mbt, labels, g) in
MTAH_OOF_DIR (default cache/oof_tcn128), to swap into P9al's v16 OOF chains on the CPU. GPU.

    python analysis/p9au_wide_tcn_oof.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_tcn128/oof_<match>.npz
"""

import glob
from functools import partial
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import p9a_oof  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5, 6, 7, 8)
p9a_oof.SEEDS = SEEDS
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_tcn128"))
WIDE = partial(G._tcn, dilations=(1, 2, 4, 8, 16, 32, 64, 128))


def main(source="dfl"):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    Sh = [G.short_mask(m["labels"]) for m in old]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    print("features done", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in (range(nd) if source == "dfl" else range(nd, nd + 3)):
        m = degrade(raws[t][0], dict(rates_for(raws[t][1]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        Tt = features(m)
        ks = [k for k in range(len(old)) if k != t]
        team = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(Sh[k], 2.0, 1.0).astype(np.float32), mm["half"] > 0)
                for k in ks for mm, F in ((old[k], To[k]), (so[k], Tso[k]))]
        pt = fit_predict(team, (Tt, g), make_net=WIDE, seeds=SEEDS)
        mt, mbt = p9a_oof.multitask(To, Tso, WIDE, ks, old, so, Sh, aw, Tt, g)
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"], pt=pt, mt=mt, mbt=mbt)
        print("saved", m["name"], flush=True)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
