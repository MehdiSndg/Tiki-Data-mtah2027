"""P9ef: the team GRU member (production features, P9a's recipe: other nine matches + team swap, short spells x2,
seeds 3-5) trained on
  A  the production copies (smooth 25, seed 0)
  B  A with Metrica's ball unglued before the degradation (mtah/unglue.py; the friend's v17)
  C  B plus a realistic-dead copy of every match (mtah/deadsim.py; P9dt)
Judges (held-out match): "old" = P9a's test copy (smooth 25, carry 0.25, seed 0); "real" = the realistic-dead copy
(P9ea's v6 judge: seed 7, realistic_dead seed 1007), Metrica unglued first. Predictions saved to cache/p9ef/ for
scoring through the chain with P9ea's dead-ball probabilities (analysis/p9ef_team_realistic.py score). GPU.

    .venv/bin/python analysis/p9ef_team_realistic.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9ef_team_realistic.py score

Registered before the run (13 Nov): on the "real" judge, B (and C) must be ahead of A on DFL at 1.6 s and not behind on
DFL 3 s or on Metrica at either referee (member alone through the semi-Markov decoder with the judge's dead ball);
only then are the six members retrained that way.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9ef")
SEEDS = (3, 4, 5)


def train():
    from mtah import grumodel as G
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.match import swap_teams
    from mtah.teammodel import features
    from mtah.unglue import unglue
    from p7d_player_set import fit_predict
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    ung = [(unglue(r, seed=11) if k == "metrica" else r, k) for r, k in raws]
    prod = lambda r, k, s: degrade(r, dict(rates_for(k), smooth_frames=25), seed=s)
    real = lambda r, k, s, carry=0.0: realistic_dead(degrade(r, dict(rates_for(k), smooth_frames=25), seed=s), r, seed=1000 + s)
    sets = {"A": [[prod(r, k, 0)] for r, k in raws],
            "B": [[prod(r, k, 0)] for r, k in ung]}
    sets["C"] = [b + [real(r, k, 1)] for b, (r, k) in zip(sets["B"], ung)]
    F = {v: [[(features(m).astype(np.float32), features(swap_teams(m)).astype(np.float32), m) for m in ms] for ms in sets[v]] for v in sets}
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
        preds = {}
        for v in sets:
            data = []
            for j in range(len(raws)):
                if j == t:
                    continue
                for Fm, Fs, m in F[v][j]:
                    sh = G.short_mask(m["labels"])
                    sw = swap_teams(m)
                    for mm, FF in ((m, Fm), (sw, Fs)):
                        data.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            allF = np.concatenate([d[0] for d in data])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            nets = [G.train([(mod.norm(d[0]),) + tuple(d[1:]) for d in data], s, "cuda") for s in SEEDS]
            for jn, m in judges.items():
                preds["{}_{}".format(v, jn)] = np.mean([G._predict_one(n, mod.norm(features(m).astype(np.float32)), g, "cuda") for n in nets], axis=0).astype(np.float32)
        np.savez_compressed(path, labels=judges["old"]["labels"], g=g, **preds)
        print(name, "saved", flush=True)


def score():
    from mtah import semimarkov as sm
    from p9de_team_fuse import C
    from p9l_rescore_calibrated import merged
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "oof_*.npz"))):
        n = os.path.basename(f)
        z, e, v16 = dict(np.load(f)), dict(np.load(C("p9ea", n))), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(v16["b"])
        from p9de_team_fuse import theirs
        md = theirs("mehmet_ali_topkara", n[4:-4], g)["p_dead"]
        dead = {"old": (e["p_old_prod"] + e["q_old"] + md) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + md) / 3}
        r = {}
        for jn in ("old", "real"):
            for v in ("A", "B", "C"):
                c = sm.decode_halves(np.nan_to_num(z["{}_{}".format(v, jn)].astype(np.float64), nan=0.5), dead[jn].astype(np.float64), g, b)
                r[(v, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[n[4:-4]] = r
        print(n[4:-4], {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real"):
        for v in ("B", "C"):
            out = ["judge {} {} vs A".format(jn, v)]
            for src in ("DFL", "metrica"):
                ms = [m for m in rows if src in m]
                d = np.array([rows[m][(v, jn)] for m in ms]) - np.array([rows[m][("A", jn)] for m in ms])
                out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
            print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
