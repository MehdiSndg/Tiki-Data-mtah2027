"""P9et: recipe C's team member trained on copies whose DFL player tracks are de-jittered first (recipe E;
mtah.dejitter, centred moving average over 9 frames inside each track piece). DFL's raw players jitter (per-frame
acceleration median 6.3 m/s² vs our ATD's 0.85; Metrica 1.4); window 9 brings degraded DFL to 0.98 on J03WMX and J03WN1,
the closest to the ATD of 3/5/7/9/13, fixed before any model was trained (label-free, tracking statistics only).
Metrica is left as it is. Same member, seeds, copies and held-out folds as P9ef / P9es:
  C  P9ef's arm C
  E  C with the DFL raws de-jittered (k 9) before both copies
Judges per held-out match: "old" (P9a's test copy), "real" (P9ef's realistic judge) and "realdj" (the same realistic
judge built from the de-jittered raw: the ATD-like one). GPU; ARMS (C or E) and FOLDS split the work, each arm writes
cache/p9et/<arm>_<match>.npz.

    ARMS=E .venv/bin/python analysis/p9et_dejitter.py train
    /home/mehdi/venvs/mali/bin/python analysis/p9et_dejitter.py score

Registered before the run (14 Nov): E replaces C (six members retrained) only if, on realdj, E is ahead on DFL at 1.6 s
and not behind on DFL 3 s or on Metrica at either referee, AND on real E is not behind on DFL at 1.6 s; then
SoccerTrack (P9ei) as second judge.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9et")
DJ = 9
SEEDS = (3, 4, 5)
SMOOTH = {"C": 25, "E": 25}


def train():
    from mtah import grumodel as G
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.match import swap_teams
    from mtah.teammodel import features
    from mtah.unglue import unglue
    from mtah.dejitter import dejitter
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    ung = [(unglue(r, seed=11) if k == "metrica" else r, k) for r, k in raws]
    raw_of = {"C": lambda r, k: r, "E": lambda r, k: r if k == "metrica" else dejitter(r, DJ)}
    prod = lambda r, k, s, sm: degrade(r, dict(rates_for(k), smooth_frames=sm), seed=s)
    real = lambda r, k, s, sm: realistic_dead(degrade(r, dict(rates_for(k), smooth_frames=sm), seed=s), r, seed=1000 + s)
    folds = os.environ.get("FOLDS", "").split(",") if os.environ.get("FOLDS") else None
    arms = os.environ.get("ARMS", "C,D").split(",")
    F = {v: [[(features(m).astype(np.float32), features(swap_teams(m)).astype(np.float32), m) for m in (prod(raw_of[v](r, k), k, 0, sm), real(raw_of[v](r, k), k, 1, sm))]
             for r, k in ung] for v, sm in SMOOTH.items() if v in arms}
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    for t, (r, k) in enumerate(raws):
        judges = {"old": degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0),
                  "real": real(ung[t][0], k, 7, 25), "realdj": real(raw_of["E"](ung[t][0], k), k, 7, 25)}
        name = judges["old"]["name"].split("+")[0]
        g = judges["old"]["half"] > 0
        for v in arms:
            path = os.path.join(OUT, "{}_{}.npz".format(v, name))
            if (folds and name not in folds) or os.path.exists(path):
                continue
            preds = {}
            data = []
            for j in range(len(raws)):
                if j == t:
                    continue
                for Fm, Fs, m in F[v][j]:
                    sh = G.short_mask(m["labels"])
                    for mm, FF in ((m, Fm), (swap_teams(m), Fs)):
                        data.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            mu, sd = G.feature_stats([d[0] for d in data])
            mod = G.GRUEnsemble([], mu, sd, "cuda")
            nets = [G.train([(mod.norm(d[0]),) + tuple(d[1:]) for d in data], s, "cuda") for s in SEEDS]
            for jn, m in judges.items():
                preds["{}_{}".format(v, jn)] = np.mean([G._predict_one(n, mod.norm(features(m).astype(np.float32)), g, "cuda") for n in nets], axis=0).astype(np.float32)
            np.savez_compressed(path, labels=judges["old"]["labels"], g=g, **preds)
            print(v, name, "saved", flush=True)


def score():
    from mtah import semimarkov as sm
    from p9de_team_fuse import C, theirs
    from p9l_rescore_calibrated import merged
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "C_*.npz"))):
        n = "oof_" + os.path.basename(f)[2:]
        if not os.path.exists(f.replace("C_", "E_")):
            continue
        z = {**dict(np.load(f)), **dict(np.load(f.replace("C_", "E_")))}
        e, v16 = dict(np.load(C("p9ea", n))), dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(v16["b"])
        md = theirs("mehmet_ali_topkara", n[4:-4], g)["p_dead"]
        dead = {"old": (e["p_old_mix"] + e["q_old"] + md) / 3, "real": (e["p_v6_mix"] + e["q_v6"] + md) / 3}
        dead["realdj"] = dead["real"]
        r = {}
        for jn in ("old", "real", "realdj"):
            for v in SMOOTH:
                c = sm.decode_halves(np.nan_to_num(z["{}_{}".format(v, jn)].astype(np.float64), nan=0.5), dead[jn].astype(np.float64), g, b)
                r[(v, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[n[4:-4]] = r
        print(n[4:-4], {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for jn in ("old", "real", "realdj"):
        out = ["judge {} E vs C".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            if ms:
                d = np.array([rows[m][("E", jn)] for m in ms]) - np.array([rows[m][("C", jn)] for m in ms])
                out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
