"""EDA section G: what do possession and dead ball look like in the external datasets?

    uv run python analysis/g_external.py

DFL: per-frame BallPossession / BallStatus straight from the files (floodlight), turned into
H/A/D labels and summarised with the same spell logic as the official scorer. The per-match
label arrays are cached to cache/dfl_labels.npz for later sections.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
DFL = os.path.join(ROOT, "data", "external", "dfl")
CACHE = os.path.join(ROOT, "cache", "dfl_labels.npz")
FPS = 25


def dfl_labels():
    if os.path.exists(CACHE):
        with np.load(CACHE) as d:
            return {k: d[k] for k in d.files}
    from floodlight.io.dfl import read_position_data_xml
    out = {}
    for pos in sorted(glob.glob(os.path.join(DFL, "*positions*.xml"))):
        mid = pos.rsplit("_", 1)[-1][:-4]
        info = glob.glob(os.path.join(DFL, "*matchinformation*{}.xml".format(mid)))[0]
        _, poss, status, _, _ = read_position_data_xml(pos, info)
        for half in poss:
            p, s = poss[half].code, status[half].code
            out["{}_{}".format(mid, half)] = np.where(s == 0, pm.D, np.where(p == 1, pm.H, pm.A)).astype(np.int8)
        print("  loaded", mid)
    np.savez_compressed(CACHE, **out)
    return out


def summarise(codes):
    runs = [(0, len(codes))]
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / FPS
    dead = [b - a for a, b, c in pm.segments(codes, runs) if c == pm.D]
    return lengths, np.array(dead) / FPS


def main():
    labels = dfl_labels()
    print("{:<20} {:>6} {:>6} {:>7} {:>7} {:>8} {:>8} {:>9}".format(
        "DFL match/half", "min", "D%", "spells", "/5min", "mean s", "med s", "<2s share"))
    allp = []
    for key in sorted(labels):
        c = labels[key]
        lengths, dead = summarise(c)
        allp.append(lengths)
        print("{:<20} {:6.1f} {:6.1%} {:7d} {:7.1f} {:8.1f} {:8.1f} {:9.1%}".format(
            key, len(c) / FPS / 60, (c == pm.D).mean(), len(lengths), len(lengths) / (len(c) / FPS / 300),
            lengths.mean(), np.median(lengths), (lengths < 2).mean()))
    allp = np.concatenate(allp)
    print("\nall DFL spells: n {}  duration s p10 {:.1f}  p25 {:.1f}  p50 {:.1f}  p75 {:.1f}  p90 {:.1f}".format(
        len(allp), *np.percentile(allp, [10, 25, 50, 75, 90])))

    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    w = gt[11185:18685]
    lengths, dead = summarise(w)
    print("public GT (5 min): D {:.1%}  spells {}  mean {:.1f} s  median {:.1f} s  <2s share {:.1%}  durations {}".format(
        (w == pm.D).mean(), len(lengths), lengths.mean(), np.median(lengths), (lengths < 2).mean(),
        np.round(np.sort(lengths), 1).tolist()))


if __name__ == "__main__":
    main()
