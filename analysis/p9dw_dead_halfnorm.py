"""P9dw: per-half scaled inputs for the dead-ball classifier (the friend's v20 "a model whose input is scaled to each
half's own tempo"; Mali found it neutral with his model). Every dead-ball v2 feature is centred on its half's median and
divided by its half's IQR (robust instance normalisation, label-free; on our match the same rule per half).
Arms (HistGB as v8's classifier, rows every 3rd frame), trained on P9dv's cached copies: prod, prod + norm, v6, v6 + norm.
Judges as P9dv (old, v6 held-out copies; 12 PFF matches; our match label-free).

    .venv/bin/python analysis/p9dw_dead_halfnorm.py

Also P9dv's follow-up arm "mix" (prod draws + v6 draws), registered after P9dv's result and before this run, under
P9dv's rule: ahead of prod on the v6 judge (DFL, Metrica) and on PFF, video agreement on our match not lower.

Registered before the run (13 Nov): norm is kept only if, on top of the better of prod / v6, it is ahead on the v6 judge
(DFL and Metrica) and on PFF, and the video agreement on our match does not fall.
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
from mtah.external import DFL  # noqa: E402
from mtah.pff import pff_ids  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9dc_dead_real import feats  # noqa: E402


def halves(lab):
    ok = lab != pm.X
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a > 25 * 60 * 5]


def norm(F, runs):
    G = F.copy()
    for a, b in runs:
        x = F[a:b]
        med = np.nanmedian(x, 0)
        iqr = np.nanpercentile(x, 75, 0) - np.nanpercentile(x, 25, 0)
        G[a:b] = (x - med) / np.where(iqr > 1e-6, iqr, 1.0)
    return G


def main(step=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    names = ids + ["metrica{}".format(g) for g in (1, 2, 3)]
    nope = lambda: (_ for _ in ()).throw(RuntimeError("P9dv's feature cache is missing"))
    C = {}
    for n in names:
        for d in P4.DRAWS_V6:
            C[("prod", n, d)] = feats("sim_{}_{}".format(n, d), nope)
            C[("v6", n, d)] = feats("simv6_{}_{}".format(n, d), nope)
        C[("v6j", n, 7)] = feats("simv6_{}_7".format(n), nope)
    pff = {i: feats("pff_{}".format(i), nope) for i in pff_ids()[-12:]}
    tf = lambda F, lab, nm: norm(F, halves(lab)) if nm else F
    rows = lambda F, lab, nm: (lambda ix: (tf(F, lab, nm)[ix], lab[ix] == pm.D))(np.flatnonzero(lab != pm.X)[::step])
    fit = lambda parts: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    def score(clf, F, lab, nm):
        g = lab != pm.X
        X = np.where(g[:, None], tf(F, lab, nm), 0)
        dead = to_blocks(np.where(g, clf.predict_proba(X)[:, 1], 0)) & g
        return pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"]

    ARMS = (("prod", False), ("prod+norm", True), ("v6", False), ("v6+norm", True), ("mix", False), ("mix+norm", True))
    srcs_of = lambda src: ("prod", "v6") if src == "mix" else (src,)
    R = {}
    for t in names:
        tr = [n for n in names if n != t]
        for arm, nm in ARMS:
            src = arm.split("+")[0]
            clf = fit([rows(*C[(s, n, d)][:2], nm) for s in srcs_of(src) for n in tr for d in P4.DRAWS_V6])
            for judge, key in (("old", ("prod", t, 0)), ("v6", ("v6j", t, 7))):
                R.setdefault(("DFL" if "metrica" not in t else "Metrica", judge, arm), []).append(score(clf, *C[key][:2], nm))
        print(t, "done", flush=True)
    from mtah.match import from_tactical
    m = from_tactical()
    g = m["half"] > 0
    Fo = features_v2(m).values.astype(np.float32)
    runs = [(int(np.flatnonzero(m["half"] == h)[0]), int(np.flatnonzero(m["half"] == h)[-1]) + 1) for h in (1, 2)]
    vd = np.load(os.path.join(ROOT, "cache", "barath_dense_avg5.npz"))["q_dead"].astype(np.float64)[g] > 0.5
    for arm, nm in ARMS:
        src = arm.split("+")[0]
        clf = fit([rows(*C[(s, n, d)][:2], nm) for s in srcs_of(src) for n in names for d in P4.DRAWS_V6])
        for i, (F, lab, _) in pff.items():
            R.setdefault(("PFF", "real", arm), []).append(score(clf, F, lab, nm))
        X = norm(Fo, runs) if nm else Fo
        p = np.zeros(len(g))
        p[g] = clf.predict_proba(X[g])[:, 1]
        dead = to_blocks(p) & g
        print("OUR MATCH", arm, "dead share {:.3f}; agreement with the video dead calls {:.4f}".format(dead[g].mean(), np.mean(dead[g] == vd)), flush=True)
    for k, v in sorted(R.items()):
        print("MEAN", " | ".join(k), "{:.4f} (n {})".format(np.mean(v), len(v)))


if __name__ == "__main__":
    main()
