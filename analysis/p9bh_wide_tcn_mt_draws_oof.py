"""Out-of-fold two-head wide-TCN member (team + boundary heads, dilations 1-128, v18's two-head TCN) trained on three
degradation draws per training match, the twin of P9bg's single-head test: p9a_oof.multitask's recipe (P8n) with the
training set of draws 0, 1, 2 (each with its teams-swapped twin); mean and std from running sums (no concatenated
feature copy). Test match and its degradation as in P9au. Seeds 3-5. Saved as oof_<match>.npz (mt, mbt, labels, g) in
MTAH_OOF_DIR (default cache/oof_tcn128d3mt). GPU.

    python analysis/p9bh_wide_tcn_mt_draws_oof.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_tcn128d3mt/oof_<match>.npz
"""

import glob
import os
import sys
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
import p8m_multitask as MT  # noqa: E402
from p8n_multitask_ensemble import widened  # noqa: E402

SEEDS = (3, 4, 5)
DRAWS = (0, 1, 2)
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_tcn128d3mt"))


def main(source="dfl"):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []                       # per training match: (features, team target, team weight, boundary target, weight, played) over draws
    for r, k in raws:
        out = []
        for d in DRAWS:
            m = degrade(r, dict(rates_for(k), smooth_frames=25), seed=d)
            sh = G.short_mask(m["labels"])
            for mm in (m, swap_teams(m)):
                out.append((features(mm), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32),
                            G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0))
        rows.append(out)
    print("features done", flush=True)
    MT.mt_net = partial(widened, G.NETS["tcn128"])
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in (range(nd) if source == "dfl" else range(nd, nd + 3)):
        m = degrade(raws[t][0], dict(rates_for(raws[t][1]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        md0 = [x for k in range(len(raws)) if k != t for x in rows[k]]
        mu, sd = G.feature_stats([d[0] for d in md0])
        mod = G.GRUEnsemble([], mu, sd, "cuda")
        md = [(mod.norm(d[0]),) + d[1:] for d in md0]
        p = np.mean([MT.mt_predict(MT.mt_train(md, s, "cuda"), mod.norm(features(m)), g, "cuda") for s in SEEDS], axis=0)
        del md
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"], mt=p[:, 0], mbt=p[:, 1])
        print("saved", m["name"], flush=True)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
