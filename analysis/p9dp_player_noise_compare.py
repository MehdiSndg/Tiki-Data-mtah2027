"""P9dp: label-free comparison of player-track defects, our ATD vs degraded DFL / Metrica (Enes's player-noise idea).

    .venv/bin/python analysis/p9dp_player_noise_compare.py

Per side: visible-slot count per frame (in-half frames), slot-gap lengths (invisible stretch between two
visible pieces of one slot), slot-piece lengths, near-duplicate pairs (< 1 m, same side), jitter
(deviation from a 1 s centred mean). No labels are used.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

FPS = 25


def runs(v):
    v = v.astype(np.int8); d = np.flatnonzero(np.diff(v)) + 1
    s = np.r_[0, d]; e = np.r_[d, len(v)]
    return s, e, v[s].astype(bool)


def stats(m):
    g = m["half"] > 0
    px, py = m["px"], m["py"]
    vis = ~np.isnan(px)
    out = {}
    cnt = [vis[g][:, m["slot_team"] == t].sum(1) for t in (0, 1)]
    c = np.concatenate(cnt)
    out["vis/side mean"] = c.mean()
    for k in (9, 10, 11, 12):
        out[f"P(vis{'<=' if k < 11 else ('==' if k == 11 else '>=')}{k})"] = (np.mean(c <= k) if k < 11 else (np.mean(c == k) if k == 11 else np.mean(c >= k)))
    gaps, pieces = [], []
    for h in (1, 2):
        idx = np.flatnonzero(m["half"] == h)
        for j in range(vis.shape[1]):
            s, e, v = runs(vis[idx, j])
            gaps += [(e[i] - s[i]) / FPS for i in range(1, len(v) - 1) if not v[i]]
            pieces += [(e[i] - s[i]) / FPS for i in range(len(v)) if v[i]]
    gaps, pieces = np.array(gaps), np.array(pieces)
    tot_min = g.sum() / FPS / 60
    out["gaps/min"] = len(gaps) / tot_min
    for q in (25, 50, 90):
        out[f"gap p{q} s"] = np.percentile(gaps, q) if len(gaps) else np.nan
    out["pieces/min"] = len(pieces) / tot_min
    out["piece p50 s"] = np.median(pieces)
    sub = np.flatnonzero(g)[::25]
    dup = 0
    for t in (0, 1):
        L, W = m["pitch"]
        X = np.stack([px[sub][:, m["slot_team"] == t] * L, py[sub][:, m["slot_team"] == t] * W], -1)
        dd = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)
        iu = np.triu_indices(X.shape[1], 1)
        dup = dup + np.sum(dd[:, iu[0], iu[1]] < 1.0, axis=1)
    out["frames w/ dup<1m"] = np.mean(dup > 0)
    return out


def main():
    rows = {"ATD (ours)": stats(from_tactical())}
    rows["DFL J03WMX clean"] = stats(from_dfl("J03WMX"))
    for i in ("J03WMX", "J03WN1", "J03WOH"):
        for s in (0, 1):
            rows[f"DFL {i} s{s}"] = stats(degrade(from_dfl(i), rates_for(i), seed=s))
    for gm in (1, 2):
        rows[f"Metrica {gm}"] = stats(degrade(from_metrica_sample(gm), rates_for("metrica"), seed=0))
    keys = list(rows["ATD (ours)"])
    print("source".ljust(18) + "".join(k[:14].rjust(15) for k in keys))
    for n, r in rows.items():
        print(n.ljust(18) + "".join(f"{r[k]:15.3f}" for k in keys))


if __name__ == "__main__":
    main()
