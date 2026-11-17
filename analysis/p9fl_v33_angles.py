"""P9fl: v33's dead-ball change (P9fh video weight 1/2 + P9fi D weight 1.0) checked from other angles than spellF1.
  st     SoccerTrack (P9di inputs): frame accuracy of D, dead-block onset / offset error, missed / invented dead blocks,
         per video fold (A models scored on fold A's matches, B on B's)
  ours   our match, label-free (v29's dump: p, dp, dq, dv, b, rule + Mali's p_dead): D share, dead blocks, and how many
         Smart Tagging set pieces (restarts) have a dead block ending within their window (restart recall)
No setting is chosen here; v30 (1/3, 1.15) vs v33 (1/2, 1.0).
"""
import copy
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402


def cfg_for(ew):
    c = copy.deepcopy(sm.CONFIG)
    c["decoder"]["emission_weights"][2] = ew
    return c


def blocks(codes, runs):
    return [(a, b) for a, b, k in pm.segments(codes, runs) if k == pm.D]


def st():
    from p9cs_st_video_dead import TEST
    fold = {m: k for k in TEST for m in TEST[k]}
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "p9di", "in_*.npz"))):
        z = np.load(f)
        mid = os.path.basename(f)[3:-4]
        g, lab = z["g"], z["lab"]
        d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
        v = np.clip(3 * d3 - 2 * d2, 0, 1)
        runs = pm.valid_runs(lab)
        G = blocks(lab, runs)
        r = {}
        for k, (w, ew) in (("v30", (1 / 3, 1.15)), ("v33", (0.5, 1.0))):
            c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), w * v + (1 - w) * d2, g, z["b"].astype(np.float64), cfg=cfg_for(ew)), pm.X).astype(np.int8)
            P = blocks(c, runs)
            acc = np.mean((c[g] == pm.D) == (lab[g] == pm.D))
            on, off, hit = [], [], 0
            for a, b in G:
                ov = [(min(b, d) - max(a, s), s, d) for s, d in P if d > a and s < b]
                if ov:
                    hit += 1
                    _, s, d = max(ov)
                    on.append(abs(s - a) / 25)
                    off.append(abs(d - b) / 25)
            inv = sum(1 for s, d in P if not any(d > a and s < b for a, b in G))
            r[k] = (acc, hit / len(G), inv, np.median(on), np.median(off))
        out[mid] = r
        print(mid, fold[mid], {k: tuple(round(x, 3) for x in v_) for k, v_ in r.items()}, flush=True)
    for k in ("v30", "v33"):
        a = np.array([out[m][k] for m in out])
        print(k, "D accuracy {:.4f}  dead blocks found {:.3f}  invented {:.1f}/match  onset |err| {:.2f} s  offset |err| {:.2f} s".format(*a.mean(0)))
    for fd in ("A", "B"):
        ms = [m for m in out if fold[m] == fd]
        print("fold", fd, {k: round(float(np.mean([out[m][k][0] for m in ms])), 4) for k in ("v30", "v33")})


def ours():
    import load_smart_tagging
    z = np.load(os.path.join(ROOT, "cache", "v29_hsmm.npz"))
    mali = np.load(os.path.join(ROOT, "cache", "team_in", "mehmet_ali_topkara", "barath_rebuilt", "components.npz"))
    g, p, b, rule = z["g"], z["p"], z["b"], z["rule"]
    dp, dq, dv, md, ok = z["dp"], z["dq"], z["dv"], mali["p_dead"].astype(np.float64), mali["valid"]
    ev = [e for e in load_smart_tagging.load() if e["code"] == "SET PIECES"]
    runs = [(a, c) for a, c in zip(np.flatnonzero(np.diff(np.r_[0, g.astype(np.int8)]) == 1), np.flatnonzero(np.diff(np.r_[g.astype(np.int8), 0]) == -1) + 1)]
    for k, (w, ew) in (("v30", (None, 1.15)), ("v33", (0.5, 1.0))):
        pdead = np.where(ok, (dp + dq + dv + md) / 4, (dp + dq + dv) / 3) if w is None else 0.5 * dv + 0.5 * np.where(ok, (dp + dq + md) / 3, (dp + dq) / 2)
        c = np.where(g, sm.decode_halves(p, pdead, g, b, forced_dead=rule, cfg=cfg_for(ew)), pm.X).astype(np.int8)
        P = blocks(c, runs)
        ends = np.array([d for s, d in P])
        rec = np.mean([np.any((ends >= e["start_frame"] - 50) & (ends <= e["end_frame"] + 50)) for e in ev])
        S = pm.possession_spells(c, runs)
        print(k, "D share {:.3f}, dead blocks {}, possessions {}, set pieces with a dead block ending in their window {:.3f} ({} set pieces)".format(
            np.mean(c[g] == pm.D), len(P), len(S), rec, len(ev)))


if __name__ == "__main__":
    {"st": st, "ours": ours}[sys.argv[1]]()
