"""P9eg: v27a's dead-ball classifier (production + realistic-dead copies) against v26's on SoccerTrack's full chain, a
judge independent of our simulation: real tracking (degraded as P9bw; its dead ball moves and is placed as in a real
match), the video channels as the submission uses them (P9di's inputs: team = chain + 0.5 x smoothed dense evidence,
boundary from the switch cost), P(dead) = mean of the classifier, the GRU (6 seeds, as P9di) and the video head;
semi-Markov decoder. 8 matches, our labels at 1.6 s / 3 s; dead-ball frame accuracy and the dead-only spellF1 (labels'
team) too. Neither classifier saw SoccerTrack.

    .venv/bin/python analysis/p9eg_st_deadmix.py

Second independent judge for v27a (13 Nov, registered before the run with P9ea): v27a must not be behind v26 here at
either referee (mean over 8 matches).
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
from mtah import semimarkov as sm  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from p9cs_st_video_dead import TEST  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    clf = {"v26": P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True),
           "v27a": P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, mix=True)}
    print("classifiers trained", flush=True)
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, b = z["g"], z["lab"], z["b"].astype(np.float64)
        assert len(g) == len(m["bx"])
        l3 = merged(lab, 3.0)
        F = features_v2(m)
        p_old = np.zeros(len(g))
        p_old[g] = clf["v26"].predict_proba(F[g])[:, 1]
        q_chk = z["dead2"].astype(np.float64) * 2 - p_old
        print(mid, "reconstruction check: q in [{:.3f}, {:.3f}] (must lie in [0, 1])".format(q_chk[g].min(), q_chk[g].max()), flush=True)
        qv3 = z["dead3"].astype(np.float64) * 3 - p_old      # q + v (P9di stored (p + q + v) / 3 with this same v26 classifier p)
        r = {}
        for k, c in clf.items():
            p = np.zeros(len(g))
            p[g] = c.predict_proba(F[g].astype(np.float32) if k == "v27a" else F[g])[:, 1]
            dead = (p + qv3) / 3
            codes = np.where(g, sm.decode_halves(z["team"].astype(np.float64), dead, g, b), pm.X).astype(np.int8)
            truth = lab == pm.D
            blocks = to_blocks(dead) & g
            r[k] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)] + [
                np.mean(blocks[g] == truth[g]), pm.spell_scores(lab, oracle_team(lab, blocks, g), pm.valid_runs(lab))["mean_f1"]]
        rows[mid] = r
        print(mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    a, bb = np.array([rows[m]["v27a"] for m in rows]), np.array([rows[m]["v26"] for m in rows])
    print("SoccerTrack 8 | chain 1.6 s / 3 s | dead frame accuracy | dead-only spellF1")
    print("v26  {:.4f} / {:.4f} | {:.4f} | {:.4f}".format(*bb.mean(0)))
    print("v27a {:.4f} / {:.4f} | {:.4f} | {:.4f}".format(*a.mean(0)))
    print("diff {:+.4f} / {:+.4f} | {:+.4f} | {:+.4f}; ahead at 1.6 s in {}/8".format(*(a - bb).mean(0), int(((a - bb)[:, 0] > 0).sum())))


if __name__ == "__main__":
    main()
