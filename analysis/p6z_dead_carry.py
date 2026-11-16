"""Train the dead-ball classifier on a degradation where the ATD keeps a dead ball (dead_carry_p). Tests on
DFL LOMO and on our match (GT-free + public as a check).

    python analysis/p6z_dead_carry.py      # -> eda_out/reports/P6z_dead_carry.md

DFL: every match degraded twice, without (old) and with dead_carry_p = CARRY (new); test always on the
new one (closer to our ATD, which tracks a dead ball on ~13% of dead frames). For each test match the
dead-ball v2 HistGB is trained on the other six + Metrica 1-2 either on the old or on the new degradation;
the team call is the booster with production features trained on the old degradation (the same for
both, so the difference is the dead ball). Our match: the classifier trained on all DFL + Metrica, old vs
new: dead share, Smart Tagging set pieces with a dead ball in the 5 s before, and the public window's dead
recall / precision printed as a correctness check.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED as DSEED, to_blocks  # noqa: E402
from mtah.deadball import broadcast_still_dead, features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import SWITCH, decode, features, fit  # noqa: E402

CARRY = 0.25
OUT = os.path.join(ROOT, "eda_out", "reports", "P6z_dead_carry.md")


def hgb(Xs, ys):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))


def main():
    from p4d_still_and_kinks import runs_of
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    new = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=CARRY), seed=0) for r, k in raws]
    Fo, Fn = [features_v2(m) for m in old], [features_v2(m) for m in new]
    T = [features(m) for m in new]
    To = [features(m) for m in old]
    print("features done", flush=True)
    g_ = lambda m: m["half"] > 0
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(raws)) if k != t]
        m = new[t]
        g = g_(m)
        team = fit([To[k][np.isin(old[k]["labels"], (pm.H, pm.A))] for k in tr if k < len(ids)],
                   [(old[k]["labels"][np.isin(old[k]["labels"], (pm.H, pm.A))] == pm.H).astype(int) for k in tr if k < len(ids)])
        p = team.predict_proba(T[t])[:, 1]
        r = {"match": m["name"]}
        for tag, F, src in (("old", Fo, old), ("new", Fn, new)):
            clf = hgb([F[k][g_(src[k])] for k in tr], [src[k]["labels"][g_(src[k])] == pm.D for k in tr])
            q = np.zeros(len(g))
            q[g] = clf.predict_proba(Fn[t][g])[:, 1]
            dead = to_blocks(q) & g
            c = np.where(g, decode(p, g & ~dead, SWITCH), pm.X).astype(np.int8)
            r[tag] = pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]
            carried = (m["labels"] == pm.D) & ~np.isnan(m["bx"])
            r[tag + "_carried_recall"] = dead[carried].mean() if carried.any() else np.nan
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)

    from mtah.frames import load_frames
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    ours = from_tactical()
    fr = load_frames()
    g = ours["half"] > 0
    Fours = features_v2(ours)
    npl = np.isin(fr["period"].values, NOT_PLAYED)
    rule = broadcast_still_dead(fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    outside = np.ones(len(g), bool)
    outside[11185:18685] = False
    starts = [a for a, _ in runs_of(fr["setpiece"].values.astype(bool)) if outside[a] and g[a]]
    gt = ours["labels"]
    w = slice(11185, 18685)
    lines = ["# P6z: ölüde izlenen top bozmaya eklenince (DFL, maç dışarıda bırakarak + bizim maç)", "", "`analysis/p6z_dead_carry.py`, dead_carry_p = {}.".format(CARRY), "",
             "## DFL (test: taşınan ölü toplu bozma)", "", "| maç | eski eğitim | yeni eğitim | taşınan ölüyü yakalama eski / yeni |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.0%} / {:.0%} |".format(r["match"], r["old"], r["new"], r["old_carried_recall"], r["new_carried_recall"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | |".format(df.old.mean(), df.new.mean()), "",
              "## Bizim maç", "", "| sınıflandırıcı | + yayın kuralı | ölü payı | duran topların önünde ölü | public: GT ölüyü yakalama / isabet (kontrol) |", "|---|---|---|---|---|"]
    for tag, F, src in (("eski", Fo, old), ("yeni", Fn, new)):
        clf = hgb([F[k][g_(src[k])] for k in range(len(raws))], [src[k]["labels"][g_(src[k])] == pm.D for k in range(len(raws))])
        q = np.zeros(len(g))
        q[g] = clf.predict_proba(Fours[g])[:, 1]
        for with_rule in (False, True):
            dead = ((to_blocks(q) | npl) & g) | (rule if with_rule else False)
            cov = np.mean([dead[max(0, a - 125):a + 1].any() for a in starts])
            tt, o = gt[w] == pm.D, dead[w]
            lines.append("| {} | {} | {:.1%} | {:.1%} | {:.0%} / {:.0%} |".format(tag, "evet" if with_rule else "hayır", dead[g].mean(), cov, o[tt].mean(), tt[o].mean() if o.any() else 0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
