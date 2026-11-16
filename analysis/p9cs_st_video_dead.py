"""P9cs: the dense video head's dead-ball probability as a third dead-ball model (rule registered in notes/BULGULAR.md
on 11 Nov before the run). SoccerTrack v2, the degraded copies of P9bw; v8's dead-ball classifier and GRU (trained on
DFL + Metrica); the cross-fold video predictions of P9ci. Arms, full chain (dead ball predicted too):
  chain            dead to_blocks((p + q) / 2), team v19c
  + dense          the same dead ball, team with the dense evidence (w 0.5)
  + dense + dead   dead to_blocks((p + q + v) / 3), team with the dense evidence

    [SMOOTH=151] python analysis/p9cs_st_video_dead.py <fold A model's predictions> <fold B model's predictions>
"""
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
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9ck_st_dense_fusion import evidence  # noqa: E402
from p9cv_evidence_shape import box  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

TEST = {"A": ("118577", "118578", "128057", "128058"), "B": ("117092", "117093", "118575", "118576")}


def video_dead(d, mid, half):
    n1 = int(np.sum(half == 1))
    v = np.full(len(half), np.nan)
    for h, hname in ((1, "1st"), (2, "2nd")):
        q = np.repeat(np.load(os.path.join(d, "pred_{}_{}.npz".format(mid, hname)))["Q"][:, 2].astype(np.float64), 2)
        off = 0 if h == 1 else n1 + 1
        n = min(len(q), int(np.sum(half == h)))
        v[off:off + n] = q[:n]
    return v


def main(dA, dB):
    clf = P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True)
    gm = P4.dead_gru(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, nseeds=6)
    R = []
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            m0 = from_soccertrack(mid, merge_s=1.6)
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            pt, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
            F = features_v2(m)
            p = np.zeros(len(g))
            p[g] = clf.predict_proba(F[g])[:, 1]
            q = np.where(g, np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0), 0)
            v = video_dead(d, mid, m0["half"])
            v = np.where(np.isnan(v), (p + q) / 2, v)
            lo = np.log(np.clip(pt, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - pt, 1e-4, 1 - 1e-4))
            pv = 1 / (1 + np.exp(-(lo + 0.5 * box(evidence(d, mid, m0["half"]), int(os.environ.get("SMOOTH", 1))))))
            d0, d1 = to_blocks((p + q) / 2) & g, to_blocks((p + q + v) / 3) & g
            l3 = merged(lab, 3.0)
            sc = lambda team, dead: np.array([pm.spell_scores(L, np.where(g, decode(team, g & ~dead, sw), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
            a, b, c = sc(pt, d0), sc(pv, d0), sc(pv, d1)
            truth = lab[g] == pm.D
            R.append((a, b, c))
            print(mid, "chain {:.4f} {:.4f} | + dense {:+.4f} {:+.4f} | + video dead {:+.4f} {:+.4f} | dead frame accuracy chain {:.3f} video alone {:.3f} three {:.3f}".format(
                *a, *(b - a), *(c - b), np.mean(d0[g] == truth), np.mean((v[g] > 0.5) == truth), np.mean(d1[g] == truth)), flush=True)
    R = np.array(R)
    gd, gv = R[:, 1] - R[:, 0], R[:, 2] - R[:, 1]
    ok = gv.mean(0).min() >= 0.003 and (gv[:, 0] > 0).sum() >= 6 and gv.min() >= -0.01
    print("full chain mean {:.4f} {:.4f}; dense evidence {:+.4f} {:+.4f} ({} / 8); video dead on top {:+.4f} {:+.4f} ({} / 8, worst {:+.4f}) -> {}".format(
        *R[:, 0].mean(0), *gd.mean(0), int((gd[:, 0] > 0).sum()), *gv.mean(0), int((gv[:, 0] > 0).sum()), gv.min(), "HOLDS" if ok else "does not hold"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
