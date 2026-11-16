"""P9du: player-track realism (P9dp: our degradation cuts player tracks ~10x more often than our ATD and shows 3x its
same-side pairs < 1 m; Enes's delivery calibrated player gaps to our ATD). Variant PL = DEFAULT rates with
dup_share 0.06, piece_drop 0.06, view_drops_players False (label-free match to our ATD: 10.35 vs 10.24 visible per
side, 7.5 vs 11.5 slot pieces per min, 12-15% vs 12% frames with a same-side pair < 1 m).
Team GRU member (production features, P9a's recipe: other nine matches + swap, short x2, seeds 3-5):
  A trained on the usual copies, B trained on PL copies; each tested on the usual judge (P9a's test copy) and on the
  PL judge (the same with PL). Scored alone through the semi-Markov decoder with our dead mean and boundary from
  cache/oof_v16. GPU.

    .venv/bin/python analysis/p9du_player_realism.py

Registered before the run (13 Nov): on the PL judge B must be ahead of A on DFL at 1.6 s and not behind on DFL 3 s or
Metrica; only then are the six members retrained on PL copies (the larger job).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

PL = dict(dup_share=0.06, piece_drop=0.06, view_drops_players=False)
OUT = os.path.join(ROOT, "cache", "oof_p9du")
SEEDS = (3, 4, 5)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    tr = {v: [degrade(r, dict(rates_for(k), smooth_frames=25, **over), seed=0) for r, k in raws] for v, over in (("A", {}), ("B", PL))}
    F = {v: [(features(m).astype(np.float32), features(swap_teams(m)).astype(np.float32)) for m in tr[v]] for v in tr}
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    rows = {}
    for t, (r, k) in enumerate(raws):
        judges = {j: degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25, **over), seed=0) for j, over in (("usual", {}), ("PL", PL))}
        name = judges["usual"]["name"].split("+")[0]
        z = dict(np.load(os.path.join(ROOT, "cache", "oof_v16", "oof_{}.npz".format(name)), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        dead, b = (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"])) / 2, np.nan_to_num(z["b"])
        ks = [j for j in range(len(raws)) if j != t]
        res, preds = {}, {}
        for v in ("A", "B"):
            data = []
            for j in ks:
                m, sw = tr[v][j], swap_teams(tr[v][j])
                sh = G.short_mask(m["labels"])
                for mm, FF in ((m, F[v][j][0]), (sw, F[v][j][1])):
                    data.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            allF = np.concatenate([d[0] for d in data])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            nets = [G.train([(mod.norm(d[0]),) + tuple(d[1:]) for d in data], s, "cuda") for s in SEEDS]
            for jn, m in judges.items():
                assert np.array_equal(m["labels"], lab)
                p = np.mean([G._predict_one(n, mod.norm(features(m).astype(np.float32)), g, "cuda") for n in nets], axis=0)
                preds["{}_{}".format(v, jn)] = p
                c = sm.decode_halves(np.nan_to_num(p, nan=0.5), dead, g, b)
                res[(v, jn)] = [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        np.savez_compressed(os.path.join(OUT, "oof_{}.npz".format(name)), **preds)
        rows[name] = res
        print(name, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in res.items()}, flush=True)
    for jn in ("usual", "PL"):
        for v in ("A", "B"):
            out = ["judge {} train {}".format(jn, v)]
            for src in ("DFL", "metrica"):
                ms = [m for m in rows if src in m]
                a = np.array([rows[m][(v, jn)] for m in ms])
                d = a - np.array([rows[m][("A", jn)] for m in ms])
                out.append("{} {:.4f} / {:.4f}  vs A {:+.4f} / {:+.4f} ({}/{})".format(src, *a.mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
            print(" | ".join(out))


if __name__ == "__main__":
    main()
