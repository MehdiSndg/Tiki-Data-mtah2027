"""P9bs: test-time team swap for the single-head wide TCN (v19's three-draw member, P9bg's recipe, seeds 3-5). The same
trained nets predict each test match twice, as it is and with home and away swapped (mtah.match.swap_teams), and the
swap's P(home) is turned back (1 - p); pt is the plain prediction, pt_tta the mean of the two. Same nets, so the
comparison is paired. Saved as oof_<match>.npz (pt, pt_tta, labels, g) in MTAH_OOF_DIR (default cache/oof_tcn128d3tta);
judged by analysis/p9bs_chain.py. GPU.

    python analysis/p9bs_tcn_swap_tta_oof.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_tcn128d3tta/oof_<match>.npz
"""

import glob
import os
import sys

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

SEEDS = (3, 4, 5)
DRAWS = (0, 1, 2)
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_tcn128d3tta"))


def fit_predict(train, test, make_net, seeds):
    """p7d_player_set.fit_predict without the concatenated copy of every training feature (WSL ran out of memory
    on 5 Nov): mean and std from running float64 sums (np.nanmean / np.nanstd up to rounding)."""
    nf = train[0][0].shape[1]
    n, s1, s2 = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    for F, *_ in train:
        ok = ~np.isnan(F)
        n += ok.sum(0)
        s1 += np.where(ok, F, 0).sum(0, dtype=np.float64)
        s2 += np.where(ok, F.astype(np.float64) ** 2, 0).sum(0)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    mod = G.GRUEnsemble([], mu, sd, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net) for s in seeds]
    del data
    return [np.mean([G._predict_one(nt, mod.norm(F), test[1], "cuda") for nt in nets], axis=0) for F in test[0]]


def main(source="dfl"):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []                       # per training match: its (features, target, weight, played) rows over all draws
    for r, k in raws:
        out = []
        for d in DRAWS:
            m = degrade(r, dict(rates_for(k), smooth_frames=25), seed=d)
            sh = G.short_mask(m["labels"])
            for mm in (m, swap_teams(m)):
                out.append((features(mm), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        rows.append(out)
    print("features done", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in (range(nd) if source == "dfl" else range(nd, nd + 3)):
        m = degrade(raws[t][0], dict(rates_for(raws[t][1]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        team = [x for k in range(len(raws)) if k != t for x in rows[k]]
        pt, sw = fit_predict(team, ((features(m), features(swap_teams(m))), g), make_net=G.NETS["tcn128"], seeds=SEEDS)
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"],
                            pt=pt, pt_tta=(pt + 1 - sw) / 2)
        print("saved", m["name"], flush=True)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
