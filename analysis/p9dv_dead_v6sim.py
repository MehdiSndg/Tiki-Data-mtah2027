"""P9dv: the dead-ball classifier trained on copies with a realistic dead ball (mtah/deadsim.py, P9dt's finding)
instead of the production copies (CARRY_RATES: the ball reappears exactly at the restart).
Arms (HistGB on dead-ball v2 features, as v8's classifier, rows every 3rd frame): prod (P4.CARRY_RATES, draws 0-2) /
v6 (smooth 25, no carry, then realistic_dead; draws 0-2). Judges, never trained on:
  old    the held-out match's production copy (carries the artefact; reported only)
  v6     the held-out match's v6 copy (draw 7)
  PFF    real World Cup broadcast tracking, no degradation, 12 matches (never used in training here)
spellF1 with the labels' team and the predicted dead ball (1.0 = perfect dead ball). Our match, label-free: dead share
and frame agreement with the five video models' dead calls (an independent sensor).

    .venv/bin/python analysis/p9dv_dead_v6sim.py

Registered before the run (13 Nov): v6 replaces prod in the dead models only if it is ahead on the v6 judge (DFL and
Metrica), ahead on PFF, and the video agreement on our match does not fall.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p4_common as P4  # noqa: E402
from m1_dead_classifier import SEED, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.deadsim import realistic_dead  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.pff import from_pff, pff_ids  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9dc_dead_real import feats  # noqa: E402

V6 = dict(smooth_frames=25)


def v6copy(raw, rk, d):
    r = raw()
    return realistic_dead(degrade(r, dict(rates_for(rk), **V6), seed=d), r, seed=1000 + d)


def main(step=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    srcs = [(i, (lambda i=i: from_dfl(i)), i) for i in ids] + [("metrica{}".format(g), (lambda g=g: from_metrica_sample(g)), "metrica") for g in (1, 2, 3)]
    prod, v6, v6j = {}, {}, {}
    for name, raw, rk in srcs:
        for d in P4.DRAWS_V6:
            prod[(name, d)] = feats("sim_{}_{}".format(name, d), lambda raw=raw, rk=rk, d=d: degrade(raw(), dict(rates_for(rk), **P4.CARRY_RATES), seed=d))
            v6[(name, d)] = feats("simv6_{}_{}".format(name, d), lambda raw=raw, rk=rk, d=d: v6copy(raw, rk, d))
        v6j[name] = feats("simv6_{}_7".format(name), lambda raw=raw, rk=rk: v6copy(raw, rk, 7))
        print("features", name, flush=True)
    pff = {i: feats("pff_{}".format(i), lambda i=i: from_pff(i)) for i in pff_ids()[-12:]}
    print("PFF features", len(pff), flush=True)
    rows = lambda F, lab: (F[np.flatnonzero(lab != pm.X)[::step]], lab[np.flatnonzero(lab != pm.X)[::step]] == pm.D)
    fit = lambda parts: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    score = lambda clf, F, lab: (lambda g: (lambda dead: pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"])(
        to_blocks(np.where(g, clf.predict_proba(np.where(g[:, None], F, 0))[:, 1], 0)) & g))(lab != pm.X)
    R = {}
    for t, _, _ in srcs:
        tr = [n for n, _, _ in srcs if n != t]
        arms = {"prod": fit([rows(*prod[(n, d)][:2]) for n in tr for d in P4.DRAWS_V6]),
                "v6": fit([rows(*v6[(n, d)][:2]) for n in tr for d in P4.DRAWS_V6])}
        for judge, (F, lab, _) in (("old", prod[(t, 0)]), ("v6", v6j[t])):
            for a, clf in arms.items():
                R.setdefault(("DFL" if "metrica" not in t else "Metrica", judge, a), []).append(score(clf, F, lab))
        print(t, {k[1:]: round(v[-1], 4) for k, v in R.items() if k[0] == ("DFL" if "metrica" not in t else "Metrica")}, flush=True)
    full = {"prod": fit([rows(*prod[(n, d)][:2]) for n, _, _ in srcs for d in P4.DRAWS_V6]),
            "v6": fit([rows(*v6[(n, d)][:2]) for n, _, _ in srcs for d in P4.DRAWS_V6])}
    for i, (F, lab, _) in pff.items():
        for a, clf in full.items():
            R.setdefault(("PFF", "real", a), []).append(score(clf, F, lab))
    for k, v in sorted(R.items()):
        print("MEAN", " | ".join(k), "{:.4f} (n {})".format(np.mean(v), len(v)))
    from mtah.match import from_tactical
    m = from_tactical()
    g = m["half"] > 0
    F = features_v2(m).values.astype(np.float32)
    vd = np.load(os.path.join(ROOT, "cache", "barath_dense_avg5.npz"))["q_dead"].astype(np.float64)[g] > 0.5
    for a, clf in full.items():
        p = np.zeros(len(g))
        p[g] = clf.predict_proba(F[g])[:, 1]
        dead = to_blocks(p) & g
        print("OUR MATCH", a, "dead share {:.3f}; agreement with the video dead calls {:.4f}; video-dead called dead {:.3f}, video-live called live {:.3f}".format(
            dead[g].mean(), np.mean(dead[g] == vd), dead[g][vd].mean(), 1 - dead[g][~vd].mean()), flush=True)
    np.save(os.path.join(ROOT, "cache", "p9dv_ours_v6_p.npy"), np.where(g, full["v6"].predict_proba(np.where(g[:, None], F, 0))[:, 1], 0))


if __name__ == "__main__":
    main()
