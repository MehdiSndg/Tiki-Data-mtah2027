"""Out-of-fold player-set member (mtah.setgru, single head, v19's set member) trained on three degradation draws per
training match (P9bg: three draws made the single-head wide TCN pass the rule, DFL +0.003, Metrica +0.003), for a
full-chain test on v19's out-of-fold chain (judged by analysis/p9bp_chain.py). P9a's recipe for ps (team features plus
the player set, teams-swapped twins, short possessions x2); features kept in float16 (the set member's inputs are ~3x
the team member's; batches are copied to float32 in training). Seeds 3-5. Saved as oof_<match>.npz (ps, labels, g) in
MTAH_OOF_DIR (default cache/oof_set3). GPU.

    python analysis/p9bp_set_draws_oof.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_set3/oof_<match>.npz
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
from mtah.setgru import player_set, set_net  # noqa: E402

SEEDS = (3, 4, 5)
DRAWS = (0, 1, 2)
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_set3"))


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
    data = [(mod.norm(F).astype(np.float16), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net) for s in seeds]
    del data
    return np.mean([G._predict_one(nt, mod.norm(test[0]), test[1], "cuda") for nt in nets], axis=0)


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
                out.append((np.concatenate([features(mm), player_set(mm)], 1).astype(np.float16), (mm["labels"] == pm.H).astype(np.float32),
                            aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        rows.append(out)
    print("features done", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in (range(nd) if source == "dfl" else range(nd, nd + 3)):
        m = degrade(raws[t][0], dict(rates_for(raws[t][1]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        team = [x for k in range(len(raws)) if k != t for x in rows[k]]
        Ft = np.concatenate([features(m), player_set(m)], 1)
        n0 = Ft.shape[1] - player_set(m).shape[1]
        ps = fit_predict(team, (Ft, g), make_net=lambda nf, n0=n0: set_net(n0), seeds=SEEDS)
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"], ps=ps)
        print("saved", m["name"], flush=True)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
