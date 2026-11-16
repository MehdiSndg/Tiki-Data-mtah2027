"""P9dt: ball visibility around restarts, our ATD (Smart Tagging set pieces, label-free) against degraded DFL / Metrica
(the labels' dead -> live transitions) -- the friend's V6 point: in the old simulation the ball's disappearing and
reappearing gave the stoppage away. Profiles of P(ball seen) from -12 s to +6 s around the restart.

    .venv/bin/python analysis/p9dt_restart_visibility.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

FPS = 25
OFFS = np.arange(-12 * FPS, 6 * FPS + 1, FPS)


def profile(vis, anchors, g):
    rows = []
    for a in anchors:
        idx = a + OFFS
        if idx[0] < 0 or idx[-1] >= len(vis) or not g[idx].all():
            continue
        rows.append(vis[idx])
    return np.mean(rows, 0), len(rows)


def main():
    m = from_tactical()
    g = m["half"] > 0
    vis = ~np.isnan(m["bx"])
    ev = load_smart_tagging.load()
    kinds = {}
    for e in ev:
        if e["code"] == "SET PIECES":
            k = str(e["tags"].get("Set Piece") or e["tags"])
            kinds.setdefault(k, []).append((e["start_frame"], e["end_frame"]))
    print("set piece tags:", {k: len(v) for k, v in kinds.items()})
    allsp = [x for v in kinds.values() for x in v]
    for nm, anchors in (("ATD tag start", [a for a, b in allsp]), ("ATD tag end", [b for a, b in allsp])):
        p, n = profile(vis, anchors, g)
        print(nm.ljust(28), n, " ".join("{:.2f}".format(x) for x in p))
    for name, raw, key in (("DFL J03WMX", from_dfl("J03WMX"), "J03WMX"), ("DFL J03WN1", from_dfl("J03WN1"), "J03WN1"), ("Metrica 1", from_metrica_sample(1), "metrica")):
        for tag, rates in (("old", dict(rates_for(key), smooth_frames=25)), ("new carry .25", dict(rates_for(key), smooth_frames=25, dead_carry_p=0.25))):
            d = degrade(raw, rates, seed=0)
            lab, gg = d["labels"], d["half"] > 0
            dead = lab == pm.D
            rs = np.flatnonzero(dead[:-1] & np.isin(lab[1:], (pm.H, pm.A)) & gg[1:]) + 1
            p, n = profile(~np.isnan(d["bx"]), rs, gg)
            print("{} {} restart".format(name, tag).ljust(28), n, " ".join("{:.2f}".format(x) for x in p))
    print("offsets s:", (OFFS / FPS).astype(int).tolist())


if __name__ == "__main__":
    main()
