"""P9ej: the six team members (v19c-v28: big / tcn128 / set, single and two-head, tcn128 single on 3 draws) trained per
held-out match on recipe A (production copies; mtah.grumodel.external_gru's recipe re-run here) and recipe C
(mtah.members_c: unglued Metrica + a realistic-dead copy), seeds 3-5, each predicting the held-out match's "old" copy
(P9a's test copy) and "real" copy (P9ef's realistic judge: Metrica unglued, realistic dead, seed 7).
Writes cache/p9ej/<match>_<member>.npz (A_old, A_real, C_old, C_real). GPU; split work with FOLDS / MEMBERS.

    FOLDS=DFL-MAT-J03WMX,metrica1 MEMBERS=big,big_mt .venv/bin/python analysis/p9ej_members_c_oof.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9ej_members_c_oof.py score

Registered before the run (13 Nov), three judges for replacing the six members by recipe C (the user's rule):
(1) realistic judge, full chain (six-member mean + Mali 0.25, v27a dead from P9ea, semi-Markov): C ahead of A on DFL
at 1.6 s, not behind on DFL 3 s or Metrica; (2) SoccerTrack full chain (P9ei: GRU member A vs C trained on all ten):
not behind; (3) our match: C's team call agrees with the video models' mean at least as often as A's where the video is
sure. The old judge is reported.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9ej")
SEEDS = (3, 4, 5)
MEMBERS = {"big": ("big", False, 1), "tcn128d3": ("tcn128", False, 3), "set": ("set", False, 1),
           "big_mt": ("big", True, 1), "tcn128_mt": ("tcn128", True, 1), "set_mt": ("set", True, 1)}


def recipe_a(arch, multitask, draws, exclude, dev):
    """external_gru(swap, metrica, short_k 2, arch, multitask, draws) without the held-out match, not cached."""
    import torch
    from mtah import grumodel as G
    from mtah.match import swap_teams
    from mtah.teammodel import SMOOTH, external_training_matches
    ms = [m for m in external_training_matches(SMOOTH, tuple((SMOOTH, d) for d in range(1, draws))) + G.metrica_training_matches(draws) if exclude not in m["name"]]
    ms = ms + [swap_teams(m) for m in ms]
    make = (lambda nf: G.widen(G.NETS[arch], nf)) if multitask else G.NETS[arch]
    Fs = [G.model_features(arch, m) for m in ms]
    mu, sd = G.feature_stats(Fs)
    model = G.GRUEnsemble([], mu, sd, dev, arch, 2 if multitask else 1)
    y1 = lambda m: (m["labels"] == pm.H).astype(np.float32)
    alive = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    w1 = lambda m: alive(m) * np.where(G.short_mask(m["labels"]), 2.0, 1.0).astype(np.float32)
    if multitask:
        y = lambda m: np.stack([y1(m), G.boundary_target(m["labels"])], 1)
        w = lambda m: np.stack([w1(m), alive(m)], 1).astype(np.float32)
    else:
        y, w = y1, w1
    data = [(model.norm(F), y(m), w(m), m["half"] > 0) for F, m in zip(Fs, ms)]
    del Fs
    model.nets = [G.train(data, s, dev, make_net=make) for s in SEEDS]
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
    members = os.environ.get("MEMBERS", "").split(",") if os.environ.get("MEMBERS") else list(MEMBERS)
    os.makedirs(OUT, exist_ok=True)
    for r, k in raws:
        old = degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)
        name = old["name"].split("+")[0]
        if folds and name not in folds:
            continue
        ru = unglue(r, seed=11) if k == "metrica" else r
        real = realistic_dead(degrade(ru, dict(rates_for(k), smooth_frames=25), seed=7), ru, seed=1007)
        excl = name.replace("DFL-MAT-", "")
        for mem in members:
            path = os.path.join(OUT, "{}_{}.npz".format(name, mem))
            if os.path.exists(path):
                continue
            arch, mt, draws = MEMBERS[mem]
            out = {}
            for rec in ("A", "C"):
                model = recipe_a(arch, mt, draws, excl, "cuda") if rec == "A" else external_gru_c("cuda", arch=arch, multitask=mt, seeds=SEEDS, draws=draws, exclude=excl, cache=False)
                for jn, m in (("old", old), ("real", real)):
                    out["{}_{}".format(rec, jn)] = G.predict(model, m).astype(np.float32)
            np.savez_compressed(path, **out)
            print(name, mem, "saved", flush=True)


def score():
    from mtah import semimarkov as sm
    from p9de_team_fuse import C, lg, sg, theirs
    from p9l_rescore_calibrated import merged
    rows = {}
    for f in sorted(glob.glob(C("p9ea", "oof_*.npz"))):
        n = os.path.basename(f)
        name = n[4:-4]
        parts = [os.path.join(OUT, "{}_{}.npz".format(name, mem)) for mem in MEMBERS]
        if not all(os.path.exists(p) for p in parts):
            print("incomplete", name)
            continue
        P = [dict(np.load(p)) for p in parts]
        e, v16 = dict(np.load(f)), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = v16["g"], v16["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(v16["b"])
        M = theirs("mehmet_ali_topkara", name, g)
        dead = {"old": (e["p_old_prod"] + e["q_old"] + M["p_dead"]) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3}
        r = {}
        for jn in ("old", "real"):
            for rec in ("A", "C"):
                p = np.nan_to_num(np.mean([x["{}_{}".format(rec, jn)] for x in P], axis=0).astype(np.float64), nan=0.5)
                team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
                c = sm.decode_halves(team, dead[jn].astype(np.float64), g, b)
                r[(rec, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[name] = r
        print(name, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real"):
        out = ["judge {} C vs A".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            if not ms:
                continue
            a = np.array([rows[m][("C", jn)] for m in ms])
            bb = np.array([rows[m][("A", jn)] for m in ms])
            out.append("{} A {:.4f} / {:.4f} C {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *bb.mean(0), *a.mean(0), *(a - bb).mean(0), int(((a - bb)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
