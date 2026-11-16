"""P9ek: the friend's short-possession expert (his v15 / v17: a sequence model around the ball pre-trained on PFF's
controlled possession, fine-tuned on the open matches, fused at 0.25-0.5). Ours: the player-set net (team features +
each team's players nearest the ball; mtah.setgru), pre-trained on 40 PFF matches with controlled-possession labels
(mtah.pff.pff_labels(controlled=True); raw broadcast tracking + team swap), then fine-tuned per held-out match on recipe C's
copies (mtah.members_c, swap, short spells x2), seeds 3-5; predictions on P9ej's "old" and "real" judge copies.
Writes cache/p9ek/<match>.npz (E_old, E_real). GPU.

    FOLDS=... .venv/bin/python analysis/p9ek_short_expert.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9ek_short_expert.py score        # needs cache/p9ej and cache/p9ea

Registered before the run (14 Nov): the expert at log-odds weight 0.25 on top of recipe C's six-member mean (+ Mali
0.25), realistic judge: DFL ahead at 1.6 s and not behind at 3 s or on Metrica; a second judge (SoccerTrack) follows
only if this holds.
"""
import copy
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9ek")
SEEDS = (3, 4, 5)
N_PFF = 40


def train():
    from mtah import grumodel as G
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.match import swap_teams
    from mtah.members_c import training_matches_c
    from mtah.pff import from_pff, pff_ids
    from mtah.unglue import unglue
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    folds = os.environ.get("FOLDS", "").split(",") if os.environ.get("FOLDS") else None
    pff = [from_pff(i, controlled=True) for i in pff_ids()[:N_PFF]]
    pff = pff + [swap_teams(m) for m in pff]
    print("PFF loaded", len(pff), flush=True)
    alive = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    y = lambda m: (m["labels"] == pm.H).astype(np.float32)
    w = lambda m: alive(m) * np.where(G.short_mask(m["labels"]), 2.0, 1.0).astype(np.float32)
    Fp = [G.model_features("set", m) for m in pff]
    os.makedirs(OUT, exist_ok=True)
    for r, k in raws:
        old = degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)
        name = old["name"].split("+")[0]
        path = os.path.join(OUT, "{}.npz".format(name))
        if (folds and name not in folds) or os.path.exists(path):
            continue
        ru = unglue(r, seed=11) if k == "metrica" else r
        real = realistic_dead(degrade(ru, dict(rates_for(k), smooth_frames=25), seed=7), ru, seed=1007)
        ms = training_matches_c(1, exclude=name.replace("DFL-MAT-", ""))
        ms = ms + [swap_teams(m) for m in ms]
        Fs = [G.model_features("set", m) for m in ms]
        mu, sd = G.feature_stats(Fs)
        model = G.GRUEnsemble([], mu, sd, "cuda", "set", 1)
        data = [(model.norm(F), y(m), w(m), m["half"] > 0) for F, m in zip(Fs, ms)]
        pre = [(model.norm(F), y(m), w(m), m["half"] > 0) for F, m in zip(Fp, pff)]
        model.nets = []
        for s in SEEDS:
            warm = G.train(pre, s, "cuda", make_net=G.NETS["set"]).train()
            model.nets.append(G.train(data, s, "cuda", make_net=lambda nf, wn=warm: copy.deepcopy(wn)))
        np.savez_compressed(path, E_old=G.predict(model, old).astype(np.float32), E_real=G.predict(model, real).astype(np.float32))
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
        prod = os.environ.get("P9EK_TEAM") == "prod"        # 14 Nov ruling: P9ea's team (v26's six members, old copy)
        if not prod and not all(os.path.exists(p) for p in parts):
            continue
        E = dict(np.load(f))
        if prod:
            z, big, t1, d3 = (dict(np.load(C(x, n), allow_pickle=True)) for x in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
            p6 = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
            P = [{"C_old": p6, "C_real": p6}]
        else:
            P = [dict(np.load(p)) for p in parts]
        e, v16 = dict(np.load(C("p9ea", n))), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = v16["g"], v16["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(v16["b"])
        M = theirs("mehmet_ali_topkara", name, g)
        dead = {"old": (e["p_old_prod"] + e["q_old"] + M["p_dead"]) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3}
        r = {}
        for jn in ("old", "real"):
            p = np.nan_to_num(np.mean([x["C_" + jn] for x in P], axis=0).astype(np.float64), nan=0.5)
            ex = np.nan_to_num(E["E_" + jn].astype(np.float64), nan=0.5)
            base = 0.75 * lg(p) + 0.25 * lg(M["team"])
            for tag, z in (("C", base), ("C+E", 0.75 * base + 0.25 * lg(ex))):
                c = sm.decode_halves(sg(z), dead[jn].astype(np.float64), g, b)
                r[(tag, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[name] = r
        print(name, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real"):
        out = ["judge {} C+E vs C".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            if ms:
                d = np.array([rows[m][("C+E", jn)] for m in ms]) - np.array([rows[m][("C", jn)] for m in ms])
                out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
