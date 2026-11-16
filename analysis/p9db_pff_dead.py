"""P9db: v8's dead-ball model (classifier + GRU, trained on degraded DFL + Metrica) on real broadcast tracking it never
saw: PFF World Cup matches (mtah.external.from_pff), no degradation, no training. Per match: dead frame accuracy,
dead-ball precision / recall, predicted and true dead share, and spellF1 with the TEAM FROM THE LABELS and the dead
ball predicted (1.0 = a perfect dead ball), so the number is the cost of the dead ball alone. The same for SoccerTrack
matches (degraded as P9bw) for scale.

    python analysis/p9db_pff_dead.py <pff ids,...> [<soccertrack ids,...>]
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
from mtah.pff import from_pff  # noqa: E402


def oracle_team(lab, dead, g):
    """The labels' team on every frame (carried over the labels' dead frames), dead where predicted."""
    live = np.isin(lab, (pm.H, pm.A))
    if not live.any():
        return np.where(g, pm.D, pm.X).astype(np.int8)
    idx = np.maximum.accumulate(np.where(live, np.arange(len(lab)), -1))
    idx = np.where(idx < 0, np.flatnonzero(live)[0], idx)
    team = lab[idx]
    return np.where(g, np.where(dead, pm.D, team), pm.X).astype(np.int8)


def main(pff, st=""):
    clf = P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True)
    gm = P4.dead_gru(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, nseeds=3)
    rows = []
    for kind, ids in (("pff", pff), ("st", st)):
        for mid in [x for x in ids.split(",") if x]:
            if kind == "pff":
                m = from_pff(mid)
            else:
                m0 = from_soccertrack(mid, merge_s=1.6)
                m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            lab = m["labels"]
            g = lab != pm.X
            F = features_v2(m)
            p = np.zeros(len(g))
            p[g] = clf.predict_proba(F[g])[:, 1]
            q = np.where(g, np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0), 0)
            dead = to_blocks((p + q) / 2) & g
            truth = lab == pm.D
            tp = np.sum(dead & truth)
            f1 = pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"]
            ball = ~np.isnan(m["bx"])
            rows.append((np.mean(dead[g] == truth[g]), f1))
            print(kind, mid, "dead frame accuracy {:.3f} precision {:.3f} recall {:.3f} share true {:.3f} predicted {:.3f} | spellF1 with the labels' team {:.4f} | "
                  "ball present live {:.3f} dead {:.3f}".format(np.mean(dead[g] == truth[g]), tp / max(dead.sum(), 1), tp / max(truth.sum(), 1), truth[g].mean(), dead[g].mean(), f1,
                                                               ball[g & ~truth].mean(), ball[truth].mean()), flush=True)
    print("mean dead frame accuracy {:.3f}, spellF1 with the labels' team {:.4f}".format(*np.mean(rows, 0)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
