"""P9dx: the dead-ball GRU (P4.dead_gru's net and recipe: features_v2, Metrica 1-3, X frames out, 3 seeds) trained on
prod copies (CARRY_RATES, draws 0-2) vs mix copies (the same + P9dv's realistic-dead copies, draws 0-2), the P9dw
"mix" arm for the second dead-ball model. Features from P9dv's cache (cache/p9dc/).
Judges: held-out folds J03WMX, J03WN1, J03WOH, metrica1 (old and v6 copies), 12 PFF matches (models on all ten), our
match label-free (agreement with the video dead calls). GPU.

    .venv/bin/python analysis/p9dx_deadgru_mix.py

Registered before the run (13 Nov), P9dv's rule: mix ahead of prod on the v6 judge and on PFF, video agreement on our
match not lower. If it holds (and P9dw's mix for the classifier held), v27 trains both dead models on mix copies.
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
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.external import DFL  # noqa: E402
from mtah.pff import pff_ids  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9dc_dead_real import feats  # noqa: E402

FOLDS = ("DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "metrica1")


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    names = ids + ["metrica{}".format(g) for g in (1, 2, 3)]
    nope = lambda: (_ for _ in ()).throw(RuntimeError("P9dv's feature cache is missing"))
    C = {}
    for n in names:
        for d in P4.DRAWS_V6:
            C[("prod", n, d)] = feats("sim_{}_{}".format(n, d), nope)[:2]
            C[("v6", n, d)] = feats("simv6_{}_{}".format(n, d), nope)[:2]
        C[("v6j", n, 7)] = feats("simv6_{}_7".format(n), nope)[:2]
    pff = {i: feats("pff_{}".format(i), nope)[:2] for i in pff_ids()[-12:]}
    src = {"prod": ("prod",), "mix": ("prod", "v6")}

    def fit(arm, tr):
        parts = [C[(s, n, d)] for s in src[arm] for n in tr for d in P4.DRAWS_V6]
        allF = np.concatenate([F for F, _ in parts])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        data = [(mod.norm(F), (lab == pm.D).astype(np.float32), (lab != pm.X).astype(np.float32), lab != pm.X) for F, lab in parts]
        mod.nets = [G.train(data, s, "cuda") for s in G.SEEDS]
        return mod

    def prob(mod, F, g):
        return np.where(g, np.mean([G._predict_one(n, mod.norm(F.astype(np.float32)), g, "cuda") for n in mod.nets], axis=0), 0)

    def score(mod, F, lab):
        g = lab != pm.X
        dead = to_blocks(prob(mod, F, g)) & g
        return pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"]

    R = {}
    for t in FOLDS:
        tr = [n for n in names if n != t]
        for arm in ("prod", "mix"):
            mod = fit(arm, tr)
            for judge, key in (("old", ("prod", t, 0)), ("v6", ("v6j", t, 7))):
                R.setdefault((judge, arm), []).append(score(mod, *C[key]))
        print(t, {k: round(v[-1], 4) for k, v in R.items()}, flush=True)
    from mtah.match import from_tactical
    m = from_tactical()
    g = m["half"] > 0
    Fo = features_v2(m).values.astype(np.float32)
    vd = np.load(os.path.join(ROOT, "cache", "barath_dense_avg5.npz"))["q_dead"].astype(np.float64)[g] > 0.5
    for arm in ("prod", "mix"):
        mod = fit(arm, names)
        for F, lab in pff.values():
            R.setdefault(("PFF", arm), []).append(score(mod, F, lab))
        q = prob(mod, Fo, g)
        np.save(os.path.join(ROOT, "cache", "p9dx_ours_{}_q.npy".format(arm)), q.astype(np.float32))
        dead = to_blocks(q) & g
        print("OUR MATCH", arm, "dead share {:.3f}; agreement with the video dead calls {:.4f}".format(dead[g].mean(), np.mean(dead[g] == vd)), flush=True)
    for k, v in sorted(R.items()):
        print("MEAN", " | ".join(k), "{:.4f} (n {})".format(np.mean(v), len(v)))


if __name__ == "__main__":
    main()
