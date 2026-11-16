"""P9en: the boundary GRU (the decoder's boundary probability b; production: external_gru(target="boundary", swap,
metrica), gru, short_k 1) re-trained on recipe C (mtah.members_c) against recipe A, per held-out match, seeds 3-5,
predicting P9ej's "old" and "real" copies. P9em: segmentation ("when") is the larger part of the team loss.
Writes cache/p9en/<match>.npz (A_old, A_real, C_old, C_real). GPU.

    FOLDS=... .venv/bin/python analysis/p9en_boundary_c.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9en_boundary_c.py score      # needs cache/p9ej, cache/p9ea

Registered before the run (14 Nov): with recipe C's six-member team (P9ej) and P9ea's dead ball, b from C against b
from A on the realistic judge: DFL ahead at 1.6 s, not behind at 3 s or on Metrica; then SoccerTrack as second judge.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9en")
SEEDS = (3, 4, 5)


def boundary_a(exclude, dev):
    from mtah import grumodel as G
    from mtah.match import swap_teams
    from mtah.teammodel import SMOOTH, external_training_matches
    ms = [m for m in external_training_matches(SMOOTH, ()) + G.metrica_training_matches(1) if exclude not in m["name"]]
    ms = ms + [swap_teams(m) for m in ms]
    Fs = [G.model_features("gru", m) for m in ms]
    mu, sd = G.feature_stats(Fs)
    model = G.GRUEnsemble([], mu, sd, dev, "gru", 1)
    alive = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    data = [(model.norm(F), G.boundary_target(m["labels"]), alive(m), m["half"] > 0) for F, m in zip(Fs, ms)]
    model.nets = [G.train(data, s, dev) for s in SEEDS]
    return model


def train():
    from mtah import grumodel as G
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.members_c import external_gru_c
    from mtah.unglue import unglue
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    folds = os.environ.get("FOLDS", "").split(",") if os.environ.get("FOLDS") else None
    os.makedirs(OUT, exist_ok=True)
    for r, k in raws:
        old = degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)
        name = old["name"].split("+")[0]
        path = os.path.join(OUT, "{}.npz".format(name))
        if (folds and name not in folds) or os.path.exists(path):
            continue
        ru = unglue(r, seed=11) if k == "metrica" else r
        real = realistic_dead(degrade(ru, dict(rates_for(k), smooth_frames=25), seed=7), ru, seed=1007)
        excl = name.replace("DFL-MAT-", "")
        out = {}
        for rec in ("A", "C"):
            model = boundary_a(excl, "cuda") if rec == "A" else external_gru_c("cuda", target="boundary", arch="gru", seeds=SEEDS, short_k=1.0, exclude=excl, cache=False)
            for jn, m in (("old", old), ("real", real)):
                out["{}_{}".format(rec, jn)] = G.predict(model, m).astype(np.float32)
        np.savez_compressed(path, **out)
        print(name, "saved", flush=True)


def score():
    from mtah import semimarkov as sm
    from p9de_team_fuse import C, lg, sg, theirs
    from p9ej_members_c_oof import MEMBERS
    from p9l_rescore_calibrated import merged
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        name = os.path.basename(f)[:-4]
        n = "oof_{}.npz".format(name)
        parts = [os.path.join(ROOT, "cache", "p9ej", "{}_{}.npz".format(name, mem)) for mem in MEMBERS]
        prod = os.environ.get("P9EN_TEAM") == "prod"        # 14 Nov ruling: P9ea's team (v26's six members, old copy)
        if not prod and not all(os.path.exists(p) for p in parts):
            continue
        B = dict(np.load(f))
        if prod:
            z, big, t1, d3 = (dict(np.load(C(x, n), allow_pickle=True)) for x in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
            p6 = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
            P = [{"C_old": p6, "C_real": p6}]
        else:
            P = [dict(np.load(p)) for p in parts]
        e, v16 = dict(np.load(C("p9ea", n))), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = v16["g"], v16["labels"]
        l3 = merged(lab, 3.0)
        M = theirs("mehmet_ali_topkara", name, g)
        dead = {"old": (e["p_old_prod"] + e["q_old"] + M["p_dead"]) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3}
        r = {}
        for jn in ("old", "real"):
            p = np.nan_to_num(np.mean([x["C_" + jn] for x in P], axis=0).astype(np.float64), nan=0.5)
            team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
            for rec in ("A", "C"):
                c = sm.decode_halves(team, dead[jn].astype(np.float64), g, np.nan_to_num(B["{}_{}".format(rec, jn)].astype(np.float64)))
                r[(rec, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[name] = r
        print(name, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real"):
        out = ["judge {} b from C vs A".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            if ms:
                d = np.array([rows[m][("C", jn)] for m in ms]) - np.array([rows[m][("A", jn)] for m in ms])
                out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
