"""Out-of-fold BigGRU single-head member pre-trained on SoccerTrack v2 (P8d / P8j: pre-training then fine-tuning gave the
GRU DFL +0.008 / +0.009; v12 used it; Metrica never measured in the full chain), for a full-chain test: P9at's recipe for
pg (one degradation draw, test match as there); every seed first trained on the usable SoccerTrack matches (degraded with
rates_for("st"), with their teams-swapped twins, standardised with the training matches' statistics), then fine-tuned on
the training matches from those weights, as mtah.grumodel.external_gru(pretrain="st"). Seeds 3-5. Saved as
oof_<match>.npz (pg, labels, g) in MTAH_OOF_DIR (default cache/oof_bigst); judged by
MTAH_ALT_DIR=cache/oof_bigst python analysis/p9ay_chain.py. GPU.

    python analysis/p9bf_big_st_oof.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_bigst/oof_<match>.npz
"""

import glob
import os
import sys

import copy

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
from p9am_big_gru import big_net  # noqa: E402

SEEDS = (3, 4, 5)
DRAWS = (0,)
ST = []
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_bigst"))


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
    pre = [(mod.norm(F), y, w, g) for F, y, w, g in ST]
    nets = []
    for s in seeds:
        warm = G.train(pre, s, "cuda", make_net=make_net).train()
        nets.append(G.train(data, s, "cuda", make_net=lambda nf, wn=warm: copy.deepcopy(wn)))
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
                out.append((features(mm), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        rows.append(out)
    global ST
    from mtah.external import from_soccertrack, soccertrack_ids, soccertrack_usable
    st = [degrade(m, dict(rates_for("st"), smooth_frames=25), seed=0) for m in (from_soccertrack(i) for i in soccertrack_ids()) if soccertrack_usable(m)]
    ST = []
    for m in st:
        sh = G.short_mask(m["labels"])
        for mm in (m, swap_teams(m)):
            ST.append((features(mm), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
    print("features done; SoccerTrack matches", len(st), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in (range(nd) if source == "dfl" else range(nd, nd + 3)):
        m = degrade(raws[t][0], dict(rates_for(raws[t][1]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        team = [x for k in range(len(raws)) if k != t for x in rows[k]]
        pg = fit_predict(team, (features(m), g), make_net=big_net, seeds=SEEDS)
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"], pg=pg)
        print("saved", m["name"], flush=True)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
