"""P9ez: the dead-ball GRU with 6 seeds instead of 3. The submission builds it with 3 (kept on 11 Nov only so v17's file
reproduced byte for byte; "6 seeds stays the default for the next build" was never acted on), while the SoccerTrack
judge chains (P9cs, P9di -> P9eg, P9ei, P9ev) already use 6. Both models are cached (models/dead_gru_d012_m123x*_d728…);
inference only. Three judges:
  st     SoccerTrack full chain, team as P9di stored (fixed), P(dead) = P9di's dead3 (classifier + 6-seed GRU + video)
         against the same with the 3-seed GRU (dead3 + (q3 - q6) / 3)
  pff    PFF World Cup (12 matches, ball only where seen): dead-ball frame accuracy of the GRU alone, 3 vs 6 seeds
  video  our match, label-free: v29's dead-ball blocks (classifier + GRU + video, its dump) with q from 3 vs 6 seeds,
         agreement with the seven video models' dead calls (P9dw's measure)

    .venv/bin/python analysis/p9ez_deadgru_seeds.py

Registered before the run (15 Nov): 6 seeds go into the build only if SoccerTrack is not behind at either referee,
PFF accuracy not lower, and video agreement not lower.
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


def q_of(gm, m):
    g = m["half"] > 0
    F = gm.norm(features_v2(m).values.astype(np.float32))
    return np.where(g, np.mean([G._predict_one(n, F, g, gm.dev) for n in gm.nets], axis=0), 0)


def main():
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from mtah.match import from_tactical
    from mtah.pff import from_pff, pff_ids
    from p9cs_st_video_dead import TEST
    from p9l_rescore_calibrated import merged
    gms = {k: P4.dead_gru(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, nseeds=k) for k in (3, 6)}
    verdict = []
    # 1. SoccerTrack chain
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, b = z["g"], z["lab"], z["b"].astype(np.float64)
        q = {k: q_of(gm, m) for k, gm in gms.items()}
        d6 = z["dead3"].astype(np.float64)
        dead = {6: d6, 3: d6 + (q[3] - q[6]) / 3}
        l3 = merged(lab, 3.0)
        r = {}
        for k, d in dead.items():
            c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), d, g, b), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[mid] = r
        print("st", mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    d = np.array([rows[m][6] for m in rows]) - np.array([rows[m][3] for m in rows])
    verdict.append((d.mean(0) >= 0).all())
    print("SoccerTrack 6 vs 3 seeds: {:+.4f} / {:+.4f} ({}/8 ahead at 1.6 s)".format(*d.mean(0), int((d[:, 0] > 0).sum())), flush=True)
    # 2. PFF dead-ball accuracy
    acc = {3: [], 6: []}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, ball="visible")
        g = m["half"] > 0
        truth = m["labels"] == pm.D
        for k, gm in gms.items():
            acc[k].append(np.mean((to_blocks(q_of(gm, m)) & g)[g] == truth[g]))
        print("pff", gid, "3: {:.4f} 6: {:.4f}".format(acc[3][-1], acc[6][-1]), flush=True)
    verdict.append(np.mean(acc[6]) >= np.mean(acc[3]))
    print("PFF dead accuracy 3 seeds {:.4f}, 6 seeds {:.4f}".format(np.mean(acc[3]), np.mean(acc[6])), flush=True)
    # 3. our match, label-free
    m = from_tactical()
    m["labels"] = np.zeros(len(m["half"]), np.int8)          # never read
    g = m["half"] > 0
    z = np.load(os.path.join(ROOT, "cache", "v29_hsmm.npz"))
    vd = np.load(os.path.join(ROOT, "cache", "barath_dense_avg7.npz"))["q_dead"].astype(np.float64)[g] > 0.5
    q6 = q_of(gms[6], m)
    ag = {}
    for k, q in ((3, z["dq"].astype(np.float64)), (6, q6)):
        ag[k] = np.mean((to_blocks((z["dp"] + q + z["dv"]) / 3) & g)[g] == vd)
    verdict.append(ag[6] >= ag[3])
    print("our match: agreement with the video dead calls 3 seeds {:.4f}, 6 seeds {:.4f}; q 3 vs dump max diff {:.2e}".format(
        ag[3], ag[6], float(np.abs(q_of(gms[3], m) - z["dq"])[g].max())), flush=True)
    print("rule (all three):", "holds" if all(verdict) else "FAILS", verdict)


if __name__ == "__main__":
    main()
