"""When is a change of possession marked, relative to the new team's first contact with the ball?

    uv run python analysis/t_transition_timing.py

1. Public GT, every H<->A transition (UNDERSTANDING ONLY -- the number must not become a parameter):
   first frame, within 2 s before the change, on which a player of the new team is within
   CONTACT m of the ATD ball; lag = GT change - that frame.
2. Metrica sample games 1-2 (external, may inform parameters): for every RECOVERY event, the frame
   on which the recovering player first comes within CONTACT m of the ball, against the event's
   annotated start frame.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.data import FPS, PITCH, load_tactical  # noqa: E402

# Params
CONTACT = 1.5                # m (Borghesi's re-tuned T_P for Link & Hoernig)
LOOKBACK = 2 * FPS
METRICA = os.path.join(ROOT, "data", "external", "metrica-sample-data", "data")


def public_gt_lags():
    m = load_tactical()
    gt = m["gt_public"]
    out = []
    for f, a, b in pm.transitions(gt, [(11185, 18685)]):
        if {a, b} != {pm.H, pm.A}:
            continue
        team = 0 if b == pm.H else 1
        cols = np.flatnonzero(m["slot_team"] == team)
        lo = f - LOOKBACK
        d = np.hypot((m["px"][lo:f + FPS, cols] - m["bx"][lo:f + FPS, None]) * PITCH[0],
                     (m["py"][lo:f + FPS, cols] - m["by"][lo:f + FPS, None]) * PITCH[1])
        with np.errstate(invalid="ignore"):
            close = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) <= CONTACT
        hit = np.flatnonzero(close)
        out.append((f, "HA"[a] + "->" + "HA"[b], (f - (lo + hit[0])) / FPS if len(hit) else np.nan))
    return out


def metrica_lags():
    rows = []
    for g in (1, 2):
        ev = pd.read_csv(os.path.join(METRICA, "Sample_Game_{0}/Sample_Game_{0}_RawEventsData.csv".format(g)))
        tracks = {}
        for side in ("Home", "Away"):
            t = pd.read_csv(os.path.join(METRICA, "Sample_Game_{0}/Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(g, side)),
                            skiprows=2)
            cols = list(t.columns)
            # columns come as PlayerN, Unnamed (x, y) pairs and Ball, Unnamed
            for i, c in enumerate(cols):
                if c.startswith("Player"):
                    tracks[c] = (t[c].values, t[cols[i + 1]].values)
                if c == "Ball":
                    ball = (t[c].values, t[cols[i + 1]].values)
            frame0 = int(t["Frame"].values[0])
        for _, r in ev[ev["Type"] == "RECOVERY"].iterrows():
            p = str(r["From"])
            if p not in tracks:
                continue
            f = int(r["Start Frame"]) - frame0
            lo, hi = max(0, f - LOOKBACK), f + LOOKBACK
            px, py = tracks[p]
            d = np.hypot((px[lo:hi] - ball[0][lo:hi]) * PITCH[0], (py[lo:hi] - ball[1][lo:hi]) * PITCH[1])
            with np.errstate(invalid="ignore"):
                hit = np.flatnonzero(d <= CONTACT)
            rows.append((g, r["Subtype"], (f - (lo + hit[0])) / FPS if len(hit) else np.nan,
                         float(np.nanmin(d)) if np.any(~np.isnan(d)) else np.nan))
    return pd.DataFrame(rows, columns=["game", "subtype", "lag_s", "min_dist"])


def main():
    lags = public_gt_lags()
    v = np.array([x[2] for x in lags], float)
    print("public GT H<->A transitions: {}; new team within {} m of the ATD ball before the change: {}".format(
        len(lags), CONTACT, int((~np.isnan(v)).sum())))
    for f, kind, lag in lags:
        print("   frame {:6d} {}  lag {}".format(f, kind, "--" if np.isnan(lag) else "{:+.2f} s".format(lag)))
    print("   lag s: median {:.2f}  p25 {:.2f}  p75 {:.2f}".format(*np.nanpercentile(v, [50, 25, 75])))
    r = metrica_lags()
    ok = r["lag_s"].notna()
    print("\nMetrica sample RECOVERY events: {}; player within {} m of the ball within +/-2 s: {}".format(len(r), CONTACT, int(ok.sum())))
    print("   annotated frame minus first contact, s: median {:.2f}  p25 {:.2f}  p75 {:.2f}".format(
        *np.percentile(r.loc[ok, "lag_s"], [50, 25, 75])))
    print(r[ok].groupby(r["subtype"].fillna("-"))["lag_s"].agg(["median", "size"]).round(2).to_string())


if __name__ == "__main__":
    main()
