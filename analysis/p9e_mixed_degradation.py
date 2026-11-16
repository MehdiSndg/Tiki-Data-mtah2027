"""Mixed-degradation training (the teammate's strongest new signal): the team members trained on each match degraded
twice, once as usual and once with a harsh degradation (ball losses, jitter, drift and track pieces x1.5, ball smoothed
over 50 frames), against the same members trained on two draws of the usual degradation (same data size; P7g found a
second usual draw alone gives nothing). Tested on the usual degradation, the harsh one and P6u's single shifts.
DFL leave one match out and Metrica 1-3 leave one game out, team call only (dead ball from the labels), flat switch,
fresh seeds (3, 4, 5). GPU.

    python analysis/p9e_mixed_degradation.py [members=gru,tcn,set] [only=<match>,...]      # -> eda_out/reports/P9e_mixed_degradation.md
"""

import glob
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

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
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402

SEEDS = (3, 4, 5)
TEST_SEED = 7
ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MEMBERS = ARGS.get("members", "gru,tcn,set").split(",")
ONLY = [x for x in ARGS.get("only", "").split(",") if x]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9e_mixed_degradation{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def usual(key):
    return dict(rates_for(key), smooth_frames=25)


def harsh(key):
    r = usual(key)
    return dict(r, loss_per_min=r["loss_per_min"] * 1.5, jitter_share=r["jitter_share"] * 1.5, drift_p=r["drift_p"] * 1.5,
                piece_drop=min(1.0, r["piece_drop"] * 1.5), smooth_frames=50)


def test_shifts(key):
    r = usual(key)
    return {"usual": r, "harsh": harsh(key), "loss x1.5": dict(r, loss_per_min=r["loss_per_min"] * 1.5),
            "jitter+drift x1.5": dict(r, jitter_share=r["jitter_share"] * 1.5, drift_p=r["drift_p"] * 1.5),
            "smooth 50": dict(r, smooth_frames=50), "pieces x1.5": dict(r, piece_drop=min(1.0, r["piece_drop"] * 1.5))}


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    copies = {"usual0": [degrade(r, usual(k), seed=0) for r, k in raws], "usual1": [degrade(r, usual(k), seed=1) for r, k in raws],
              "harsh1": [degrade(r, harsh(k), seed=1) for r, k in raws]}
    feats = {}                            # (copy, match) -> base features, swapped twin's, set features of both, matches
    for c, ms in copies.items():
        for k, m in enumerate(ms):
            sw = swap_teams(m)
            F, Fs = features(m), features(sw)
            S = (player_set(m), player_set(sw)) if "set" in MEMBERS else (None, None)
            feats[c, k] = (F, Fs) + S + (m, sw)
    n0 = feats["usual0", 0][0].shape[1]
    make = {"gru": None, "tcn": G._tcn, "set": partial(lambda nf, n: set_net(n), n=n0)}
    pick = lambda a, F, S: np.concatenate([F, S], 1) if a == "set" else F
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    for t, (r0, key) in enumerate(raws):
        name = copies["usual0"][t]["name"]
        if ONLY and not any(o in name for o in ONLY):
            continue
        ks = [k for k in range(len(raws)) if k != t]
        tests = {s: degrade(r0, rates, seed=TEST_SEED) for s, rates in test_shifts(key).items()}
        tf = {s: (features(mt), player_set(mt) if "set" in MEMBERS else None) for s, mt in tests.items()}
        preds = {}
        for variant, cs in (("usual x2", ("usual0", "usual1")), ("mixed", ("usual0", "harsh1"))):
            ps = {s: [] for s in tests}
            for a in MEMBERS:
                tr = []
                for c in cs:
                    for k in ks:
                        F, Fs, S, Ss, m, sw = feats[c, k]
                        sh = G.short_mask(m["labels"])
                        for mm, FF in ((m, pick(a, F, S)), (sw, pick(a, Fs, Ss))):
                            tr.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
                allF = np.concatenate([d[0] for d in tr])
                mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
                tr = [(mod.norm(d[0]),) + tuple(d[1:]) for d in tr]
                nets = [G.train(tr, sd, "cuda", make_net=make[a]) for sd in SEEDS]
                for s, mt in tests.items():
                    X, g = mod.norm(pick(a, *tf[s])), mt["half"] > 0
                    ps[s].append(np.mean([G._predict_one(n, X, g, "cuda") for n in nets], axis=0))
            preds[variant] = {s: np.mean(v, axis=0) for s, v in ps.items()}
        for s, mt in tests.items():
            g = mt["half"] > 0
            alive = np.isin(mt["labels"], (pm.H, pm.A))
            sc = lambda p: pm.spell_scores(mt["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(mt["labels"]))["mean_f1"]
            r = {"match": name, "test": s, "usual x2": sc(preds["usual x2"][s]), "mixed": sc(preds["mixed"][s])}
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9e: karışık bozmayla eğitim (üyeler {}, sadece takım, tohum 3–5)".format("+".join(MEMBERS)), "", "`analysis/p9e_mixed_degradation.py`.", "",
             "| test bozması | kaynak | normal ×2 | karışık | fark | önde |", "|---|---|---|---|---|---|"]
    for s in test_shifts("x"):
        for src in ("DFL", "metrica"):
            d = df[(df.test == s) & df.match.str.startswith(src)]
            if len(d):
                lines.append("| {} | {} | {:.3f} | {:.3f} | {:+.3f} | {} / {} |".format(s, src, d["usual x2"].mean(), d.mixed.mean(), (d.mixed - d["usual x2"]).mean(),
                                                                               int((d.mixed > d["usual x2"]).sum()), len(d)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    df.to_csv(OUT.replace(".md", ".csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
