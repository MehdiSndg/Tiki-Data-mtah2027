"""P9ea: v27a against v26 in the full chain (semi-Markov decoder, our labels at 1.6 s / 3 s), DFL 7 + Metrica 3.
Team: v26's fused out-of-fold probability (six members + Mali's team at 0.25) and boundary, unchanged (v27a does not
touch them). Dead ball: mean of the classifier (prod or mix, held-out fold, HistGB as production: all rows), the
dead-ball GRU (prod, held-out fold, 3 seeds) and Mali's P(dead), each dead-ball model reading the judge's copy:
  old  the production test copy (= the out-of-fold copy, carries the restart artefact)
  v6   the realistic-dead copy (draw 7)
(the team probabilities were computed on the old copy in both cases). GPU for the GRU folds.

    .venv/bin/python analysis/p9ea_chain_deadmix.py

Registered before the run (13 Nov): v27a replaces v26 as the submission if on the v6 judge it is ahead on DFL at 1.6 s
and not behind on DFL at 3 s or on Metrica at either referee. The old judge is reported only.
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
from m1_dead_classifier import SEED  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from mtah.external import DFL  # noqa: E402
from p9dc_dead_real import feats  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    names = ids + ["metrica{}".format(g) for g in (1, 2, 3)]
    nope = lambda: (_ for _ in ()).throw(RuntimeError("P9dv's feature cache is missing"))
    Fc = {(s, n, d): feats("{}_{}_{}".format(k, n, d), nope)[:2] for s, k in (("prod", "sim"), ("v6", "simv6")) for n in names for d in P4.DRAWS_V6}
    J = {n: {"old": Fc[("prod", n, 0)], "v6": feats("simv6_{}_7".format(n), nope)[:2]} for n in names}
    rows = lambda F, lab: (F[lab != pm.X], lab[lab != pm.X] == pm.D)
    fit = lambda parts: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
        np.concatenate([p[0] for p in parts]).astype(np.float32), np.concatenate([p[1] for p in parts]))
    res = {}
    for t in names:
        if os.path.exists(C("p9ea", "oof_{}.npz".format(t))) and os.environ.get("P9EA_SKIP"):
            continue
        tr = [n for n in names if n != t]
        clf = {"prod": fit([rows(*Fc[("prod", n, d)]) for n in tr for d in P4.DRAWS_V6]),
               "mix": fit([rows(*Fc[(s, n, d)]) for s in ("prod", "v6") for n in tr for d in P4.DRAWS_V6])}
        parts = [Fc[("prod", n, d)] for n in tr for d in P4.DRAWS_V6]
        allF = np.concatenate([F for F, _ in parts])
        gm = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        gm.nets = [G.train([(gm.norm(F), (lab == pm.D).astype(np.float32), (lab != pm.X).astype(np.float32), lab != pm.X) for F, lab in parts], s, "cuda") for s in G.SEEDS]
        oname = t
        n = "oof_{}.npz".format(oname)
        z, big, t1, d3 = (dict(np.load(C(x, n), allow_pickle=True)) for x in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        M = theirs("mehmet_ali_topkara", t, g)
        team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
        r = {}
        os.makedirs(C("p9ea"), exist_ok=True)
        dump = {}
        for judge, (F, jl) in J[t].items():
            assert np.array_equal(jl, lab), (t, judge)
            X = np.where(g[:, None], F, 0).astype(np.float32)
            q = np.where(g, np.mean([G._predict_one(nn, gm.norm(F.astype(np.float32)), g, "cuda") for nn in gm.nets], axis=0), 0)
            dump["q_{}".format(judge)] = q.astype(np.float32)
            for arm, c in clf.items():
                pdd = np.where(g, c.predict_proba(X)[:, 1], 0)
                dump["p_{}_{}".format(judge, arm)] = pdd.astype(np.float32)
                dead = (pdd + q + M["p_dead"]) / 3
                codes = np.where(g, sm.decode_halves(team, dead, g, b), pm.X).astype(np.int8)
                r[(judge, arm)] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        np.savez_compressed(C("p9ea", n), **dump)
        res[t] = r
        print(t, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    for judge in ("old", "v6"):
        out = ["judge " + judge]
        for src in ("DFL", "metrica"):
            ms = [m for m in res if src in m]
            a = np.array([res[m][(judge, "mix")] for m in ms])
            bb = np.array([res[m][(judge, "prod")] for m in ms])
            out.append("{} prod {:.4f} / {:.4f} mix {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *bb.mean(0), *a.mean(0), *(a - bb).mean(0), int(((a - bb)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
