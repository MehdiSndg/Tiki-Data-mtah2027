"""Section H: line the broadcast up with the tactical video, then ask what the broadcast's cutaways say.

    uv run python analysis/h_broadcast_sync.py

Coarse offset from the burned-in match clock (read by eye on four frames: broadcast 20000 = 08:32,
60000 = 35:12, 100000 = 53:46, 140000 = 80:25), refined per window by matching the two ATD ball
tracks (both on normalised pitch coordinates). Then, on the synchronised timeline: is a broadcast
frame without a usable homography (close-up, replay, crowd) a sign of a dead ball?
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from mtah.data import load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402

# Params
COARSE = {1: 3985, 2: 26867}        # tactical = broadcast + offset, from the clock overlay
WINDOW = 1500                       # broadcast frames per refinement window (60 s)
SEARCH = 75                         # +/- frames searched around the coarse offset
OUT = os.path.join(ROOT, "cache", "broadcast_sync.npz")


def broadcast_ball():
    raw = asset_paths(source="broadcast")["atd_raw"]
    n = 164488
    bx = np.full(n, np.nan, np.float32)
    by = bx.copy()
    with open(raw) as f:
        for line in f:
            a, _, b = line.rstrip("\n").split(":")
            if b[0] != "N":
                x, y, _ = b.split(",")
                bx[int(a) - 1], by[int(a) - 1] = float(x), float(y)
    return bx, by


def main():
    m = load_tactical()
    tbx, tby = m["bx"], m["by"]
    bbx, bby = broadcast_ball()
    hom = pd.read_csv(asset_paths(source="broadcast")["homography"], usecols=["frame", "valid"])["valid"].values.astype(bool)
    nb = len(bbx)
    # broadcast frame -> half, from the coarse offsets and the tactical half blocks
    t_half = m["half"]
    offset = np.full(nb, np.nan)
    rows = []
    for a in range(0, nb - WINDOW, WINDOW):
        best = None
        for half, co in COARSE.items():
            for d in range(-SEARCH, SEARCH + 1):
                t = np.arange(a, a + WINDOW) + co + d
                if t[0] < 0 or t[-1] >= len(tbx) or not np.all(t_half[t] == half):
                    continue
                ok = ~np.isnan(bbx[a:a + WINDOW]) & ~np.isnan(tbx[t])
                if ok.sum() < 200:
                    continue
                err = np.median(np.hypot(bbx[a:a + WINDOW][ok] - tbx[t][ok], bby[a:a + WINDOW][ok] - tby[t][ok]))
                if best is None or err < best[0]:
                    best = (err, co + d, int(ok.sum()))
        if best is not None and best[0] < 0.02:              # within ~2 m on the pitch
            offset[a:a + WINDOW] = best[1]
            rows.append((a, best[1], best[0], best[2]))
    r = pd.DataFrame(rows, columns=["b_start", "offset", "median_err", "n"])
    print("windows synchronised: {} of {}; offset per half (median): {}".format(
        len(r), nb // WINDOW, r.groupby(r["offset"] > 15000)["offset"].median().to_dict()))
    print("offset spread inside a half (p5..p95): {}".format(
        r.groupby(r["offset"] > 15000)["offset"].quantile([0.05, 0.95]).round().to_dict()))
    print("median ball distance at the best offset: {:.4f} (normalised)".format(r["median_err"].median()))
    np.savez_compressed(OUT, offset=offset, valid=hom)

    # map broadcast cutaways onto the tactical timeline
    df = load_frames()
    ok = ~np.isnan(offset)
    t = (np.arange(nb) + np.nan_to_num(offset)).astype(int)[ok]
    cut = ~hom[ok]
    g = m["graded"][t]
    gt = m["gt_public"][t]
    pub = (t >= 11185) & (t < 18685)
    print("\nbroadcast frames on graded tactical frames: {}; of those without a usable homography: {:.1%}".format(int(g.sum()), cut[g].mean()))
    for name, sel in (("public GT alive", pub & (gt < 2)), ("public GT dead", pub & (gt == 2))):
        print("   {} (understanding only): broadcast cut away on {:.1%} of {} frames".format(name, cut[sel].mean(), int(sel.sum())))
    st = df["ball_status"].values[t]
    ms = pd.Series(df["med_speed"].values).rolling(50, center=True, min_periods=10).median().values[t]
    for name, sel in (("tactical ball MISSING", g & (st == 3)), ("tactical ball seen", g & (st <= 1)),
                      ("players slow (< 1.3 m/s, 2 s median)", g & (ms < 1.3)), ("players fast", g & (ms >= 1.3))):
        print("   {} (GT-free): broadcast cut away on {:.1%} of {} frames".format(name, cut[sel].mean(), int(sel.sum())))


if __name__ == "__main__":
    main()
