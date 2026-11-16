"""The common Match format every evaluation source is converted to, and our own match in it.

    from mtah.match import from_tactical, spell_stats
    m = from_tactical()

A Match is a dict, one entry per frame along the first axis of every array:

    name       str
    px, py     (n, k) float32   track slot x, y normalised 0-1, (0, 0) top-left, x along the pitch;
                                NaN when absent
    slot_team  (k,)   int8      0 home, 1 away
    gk         {0: slot, 1: slot}  each team's goalkeeper slot, -1 when unknown
    bx, by     (n,)   float32   ball, normalised like the players; NaN when absent
    half       (n,)   int8      1, 2, or 0 outside both halves
    labels     (n,)   int8      possession_metrics codes (H A D X); X outside the halves
    sign       {1: +1/-1, 2: +1/-1}  the direction along x in which the HOME side attacks
    pitch      (length, width)  metres, for every distance
    public_gt  True only on our match, whose labels are the public ground truth

For our match the labels are the public ground truth: a correctness check, never a target to tune on.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.periods import attack_sign  # noqa: E402

FPS = 25


def spell_stats(codes):
    """Possessions per 5 graded minutes and the D share, as analysis/g2_metrica_events.stats counts them."""
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    graded = codes != pm.X
    return {"spells_per_5min": len(spells) / (graded.sum() / FPS / 300),
            "d_share": float(np.mean(codes[graded] == pm.D))}


def keepers_and_sign(px, half, slot_team):
    """Each team's keeper -- its slot with the most extreme first-half mean x, on the side the team sits
    on -- and the home attack sign. The side is the lower of the two team means, not halfway: a team
    can sit past halfway for a whole half (DFL J03WR9)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # slots never seen in the first half
        mean_x = np.nanmean(px[half == 1], axis=0)
        team_mean = {t: np.nanmean(mean_x[slot_team == t]) for t in (0, 1)}
    home_low = team_mean[0] < team_mean[1] if not np.isnan(team_mean[0] + team_mean[1]) else team_mean[0] < 0.5
    gk = {}
    for t in (0, 1):
        slots = np.flatnonzero((slot_team == t) & ~np.isnan(mean_x))
        if not len(slots):
            gk[t] = -1
            continue
        low = home_low if t == 0 else not home_low
        gk[t] = int(slots[np.argmin(mean_x[slots])] if low else slots[np.argmax(mean_x[slots])])
    first = 1 if home_low else -1
    return gk, {1: first, 2: -first}


def swap_teams(m):
    """A copy of match m with home and away swapped: labels H <-> A, slot teams, keepers, attack sign.
    Nothing about possession depends on which team is home, so it is a second, equally real match for
    training the team models (P7e). Arrays not named here are shared with m."""
    s = dict(m)
    lab = np.asarray(m["labels"]).copy()
    lab[m["labels"] == pm.H], lab[m["labels"] == pm.A] = pm.A, pm.H
    s["labels"] = lab
    s["slot_team"] = (1 - np.asarray(m["slot_team"])).astype(np.int8)
    s["gk"] = {0: m["gk"][1], 1: m["gk"][0]}
    s["sign"] = {h: -v for h, v in m["sign"].items()}
    s["name"] = m["name"] + "+swap"
    return s


def from_tactical():
    m = load_tactical()
    return {"name": "tactical", "px": m["px"], "py": m["py"], "slot_team": m["slot_team"],
            "gk": dict(GK_SLOTS), "bx": m["bx"], "by": m["by"], "half": m["half"],
            "labels": m["gt_public"], "sign": attack_sign(asset_paths()["atd_meta"]),
            "pitch": tuple(float(v) for v in PITCH), "public_gt": True}
