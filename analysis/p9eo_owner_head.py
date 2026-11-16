"""P9eo: recipe C's team GRU member with a training-only clean-owner head (mtah.owner; the Barca-Hackathon fork's
idea, +0.007 [+0.002, +0.013] on their held-out Metrica). Same member, copies, seeds and judges as P9ef:
  C   P9ef's arm C (production copy seed 0 of the unglued raws + a realistic-dead copy seed 1; team swap; short x2)
  CO  C with two more outputs, "a home / an away player within 1.5 m of the ball in the clean tracking", BCE at
      weight 0.5 where the clean ball is tracked; prediction reads the team output only
Held out per match (7 DFL + Metrica 1-3); judges "old" and "real" as P9ef. GPU.

    .venv/bin/python analysis/p9eo_owner_head.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9eo_owner_head.py score

Registered before the run (14 Nov): on the "real" judge CO must be ahead of C on DFL at 1.6 s and not behind on DFL
3 s or on Metrica at either referee (member alone through the semi-Markov decoder with the judge's dead ball, as
P9ef); only then the six members are retrained with the head and judged on SoccerTrack (P9ei's pattern).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9eo")
SEEDS = (3, 4, 5)
OWNER_W = 0.5


def train():
    from mtah import grumodel as G
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.match import swap_teams
    from mtah.owner import owner_target
    from mtah.teammodel import features
    from mtah.unglue import unglue
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    ung = [(unglue(r, seed=11) if k == "metrica" else r, k) for r, k in raws]
    prod = lambda r, k, s: degrade(r, dict(rates_for(k), smooth_frames=25), seed=s)
    real = lambda r, k, s: realistic_dead(degrade(r, dict(rates_for(k), smooth_frames=25), seed=s), r, seed=1000 + s)
    copies = [([prod(r, k, 0), real(r, k, 1)], owner_target(r)) for r, k in ung]
    F = [[(features(m).astype(np.float32), features(swap_teams(m)).astype(np.float32), m) for m in ms] for ms, _ in copies]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    for t, (r, k) in enumerate(raws):
        judges = {"old": degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0),
                  "real": real(ung[t][0], k, 7)}
        name = judges["old"]["name"].split("+")[0]
        path = os.path.join(OUT, "oof_{}.npz".format(name))
        if os.path.exists(path):
            continue
        g = judges["old"]["half"] > 0
        data = {"C": [], "CO": []}
        for j in range(len(raws)):
            if j == t:
                continue
            yo, wo = copies[j][1]
            for Fm, Fs, m in F[j]:
                sh = G.short_mask(m["labels"])
                for mm, FF, oy in ((m, Fm, yo), (swap_teams(m), Fs, yo[:, ::-1])):
                    yt = (mm["labels"] == pm.H).astype(np.float32)
                    wt = aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32)
                    data["C"].append((FF, yt, wt, mm["half"] > 0))
                    data["CO"].append((FF, np.column_stack([yt, oy]).astype(np.float32),
                                       np.column_stack([wt, OWNER_W * wo, OWNER_W * wo]).astype(np.float32), mm["half"] > 0))
        mu, sd = G.feature_stats([d[0] for d in data["C"]])
        mod = G.GRUEnsemble([], mu, sd, "cuda")
        preds = {}
        for v, make in (("C", None), ("CO", lambda nf: G.widen(G._net, nf, heads=3))):
            nets = [G.train([(mod.norm(d[0]),) + tuple(d[1:]) for d in data[v]], s, "cuda", make_net=make) for s in SEEDS]
            for jn, m in judges.items():
                Fj = mod.norm(features(m).astype(np.float32))
                p = np.mean([G._predict_one(n, Fj, g, "cuda") for n in nets], axis=0)
                preds["{}_{}".format(v, jn)] = (p[:, 0] if p.ndim == 2 else p).astype(np.float32)
        np.savez_compressed(path, labels=judges["old"]["labels"], g=g, **preds)
        print(name, "saved", flush=True)


def score():
    from mtah import semimarkov as sm
    from p9de_team_fuse import C, theirs
    from p9l_rescore_calibrated import merged
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "oof_*.npz"))):
        n = os.path.basename(f)
        z, e, v16 = dict(np.load(f)), dict(np.load(C("p9ea", n))), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(v16["b"])
        md = theirs("mehmet_ali_topkara", n[4:-4], g)["p_dead"]
        dead = {"old": (e["p_old_prod"] + e["q_old"] + md) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + md) / 3}
        r = {}
        for jn in ("old", "real"):
            for v in ("C", "CO"):
                c = sm.decode_halves(np.nan_to_num(z["{}_{}".format(v, jn)].astype(np.float64), nan=0.5), dead[jn].astype(np.float64), g, b)
                r[(v, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[n[4:-4]] = r
        print(n[4:-4], {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real"):
        out = ["judge {} CO vs C".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            d = np.array([rows[m][("CO", jn)] for m in ms]) - np.array([rows[m][("C", jn)] for m in ms])
            out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
