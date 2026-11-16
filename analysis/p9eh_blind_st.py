"""P9eh: the ball-blind channel (P9ds) judged on SoccerTrack's full chain, because P9dy's Metrica judge carries the glued
Metrica ball (13 Nov scan: within 0.3 m of a player on ~27% of degraded Metrica live frames vs 5% on our ATD), which
favours the ball-seeing members there. Blind GRU + TCN trained on all ten external matches (P9ds's recipe, seeds 3-5),
predicted on the 8 SoccerTrack matches (P9di's copies), fused into P9di's team probability (chain + smoothed video
evidence) at log-odds weight w, P9di's dead ball (classifier + GRU + video), semi-Markov decoder. GPU.

    .venv/bin/python analysis/p9eh_blind_st.py

Registered before the run (13 Nov): w in {0.1, 0.2} picked per match on the other seven; the blind channel is adopted
only if SoccerTrack nested is ahead at 1.6 s and not behind at 3 s (DFL already ahead in P9dy: +0.0017 / +0.0032).
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
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9cs_st_video_dead import TEST  # noqa: E402
from p9de_team_fuse import lg, sg  # noqa: E402
from p9ds_ballblind import blind  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

WS = (0.1, 0.2)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    To, Tso = [features(blind(m)) for m in old], [features(blind(m)) for m in so]
    keep = np.flatnonzero(np.nanstd(np.concatenate(To), axis=0) > 1e-9)
    To, Tso = [t[:, keep].astype(np.float32) for t in To], [t[:, keep].astype(np.float32) for t in Tso]
    Sh = [G.short_mask(m["labels"]) for m in old]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    data = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(Sh[j], 2.0, 1.0).astype(np.float32), mm["half"] > 0)
            for j in range(len(old)) for mm, F in ((old[j], To[j]), (so[j], Tso[j]))]
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, b = z["g"], z["lab"], z["b"].astype(np.float64)
        Tt = features(blind(m))[:, keep].astype(np.float32)
        if not rows:
            print("training on all ten", flush=True)
        pb = (fit_predict(data, (Tt, g), seeds=(3, 4, 5)) + fit_predict(data, (Tt, g), make_net=G._tcn, seeds=(3, 4, 5))) / 2
        pb = np.nan_to_num(pb, nan=0.5)
        l3 = merged(lab, 3.0)
        team, dead = z["team"].astype(np.float64), z["dead3"].astype(np.float64)
        sc = lambda t: [pm.spell_scores(L, np.where(g, sm.decode_halves(t, dead, g, b), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        r = {"base": sc(team)}
        for w in WS:
            r[w] = sc(sg((1 - w) * lg(team) + w * lg(pb)))
        alive = np.isin(lab, (pm.H, pm.A)) & g
        rows[mid] = r
        print(mid, "blind live accuracy {:.3f} team {:.3f}".format(np.mean((pb[alive] > 0.5) == (lab[alive] == pm.H)), np.mean((team[alive] > 0.5) == (lab[alive] == pm.H))),
              {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    names = sorted(rows)
    base = np.array([rows[m]["base"] for m in names])
    for k in WS + ("nested",):
        a = np.array([rows[m][k] if k != "nested" else rows[m][max(WS, key=lambda w: np.mean([np.mean(rows[o][w]) - np.mean(rows[o]["base"]) for o in names if o != m]))] for m in names])
        d = a - base
        print(k, "SoccerTrack {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/8)".format(*a.mean(0), *d.mean(0), int((d[:, 0] > 0).sum())))


if __name__ == "__main__":
    main()
