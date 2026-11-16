"""Load the tactical match bundle once, as numpy arrays indexed by video frame, and cache it.

    from mtah.data import load_tactical
    m = load_tactical()            # ~2 s from cache/; ~1 min the first time
    m["px"][frame, slot]           # normalised pitch x of a track slot, NaN when absent

Everything is indexed by **video frame** (0 .. 199318), the submission row, so no caller ever
touches the ATD's own counter. Arrays:

    px, py, ps       (n, 46) float32   track slot x, y (0-1) and speed (m/s); NaN when absent
    slot_team        (46,)   int8      0 home, 1 away -- from the ATD metadata, first team = home
    bx, by, bs       (n,)    float32   ball x, y (0-1) and speed; NaN when absent
    hom              (n, 3, 3) float64 pitch (0-1) -> pixel homography
    hom_err          (n,)    float32   reprojection error
    graded           (n,)    bool      inside one of the two graded halves
    half             (n,)    int8      1, 2, or 0 outside both
    gt_public        (n,)    int8      public ground truth as possession_metrics codes (H A D X)

The cache lives in cache/ (gitignored): it is derived match data and must not be published.
Delete the file to rebuild it; it is also rebuilt when a source file is newer than the cache.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
PITCH = GAMES[GAME]["pitch"]
FPS = GAMES[GAME]["fps"]


def _parse_atd(raw_path, n_frames, n_slots):
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = px.copy()
    ps = px.copy()
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball_part = line.rstrip("\n").split(":")
            row = int(frame_id) - 1                     # ATD counts from 1, video from 0
            for slot, group in enumerate(players.split(";")):
                if group[0] == "N":
                    continue
                x, y, s = group.split(",")
                px[row, slot] = float(x)
                py[row, slot] = float(y)
                if s[0] != "N":
                    ps[row, slot] = float(s)
            if ball_part[0] != "N":
                ball[row] = [np.nan if v[0] == "N" else float(v) for v in ball_part.split(",")]
    return px, py, ps, ball


def _slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def goalkeeper_slots(meta_path):
    """{0 home, 1 away: 0-based slot} of the players the ATD metadata marks position_type Goalkeeper."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    out = {}
    for i, p in enumerate(root.iter("Player")):
        kind = {q.findtext("Name"): q.findtext("Value") for q in p.iter("ProviderParameter")}.get("position_type")
        if kind == "Goalkeeper":
            out.setdefault(0 if p.get("teamId") == home_id else 1, i)
    return out


GK_SLOTS = goalkeeper_slots(asset_paths()["atd_meta"]) if os.path.exists(asset_paths()["atd_meta"]) else {}


def _build(path):
    meta, raw, hom_path = require(["atd_meta", "atd_raw", "homography"])
    n = source_frames("tactical")
    slot_team = _slot_teams(meta)
    px, py, ps, ball = _parse_atd(raw, n, len(slot_team))

    import pandas as pd
    import possession_metrics as pm
    table = pd.read_csv(hom_path)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    hom = table[cols].values.astype(np.float64).reshape(-1, 3, 3)

    half = np.zeros(n, dtype=np.int8)
    for h, first, last in half_video_blocks(GAME):
        half[first:last + 1] = h

    gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    np.savez(path, px=px, py=py, ps=ps, slot_team=slot_team,
             bx=ball[:, 0], by=ball[:, 1], bs=ball[:, 2],
             hom=hom, hom_err=table["reprojection_error"].values.astype(np.float32),
             half=half, graded=half > 0, gt_public=gt.astype(np.int8))


def load_tactical():
    """The cached tactical arrays as a dict (see module docstring)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "tactical_v1.npz")
    sources = require(["atd_meta", "atd_raw", "homography"]) + [GT_PUBLIC]
    if not os.path.exists(path) or os.path.getmtime(path) < max(map(os.path.getmtime, sources)):
        _build(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def to_metres(x, y):
    """Normalised (0-1) pitch coordinates -> metres, using the measured pitch size."""
    return x * PITCH[0], y * PITCH[1]


def project(hom, x, y):
    """Normalised pitch -> pixels for one 3x3 matrix; arrays broadcast."""
    d = hom[2, 0] * x + hom[2, 1] * y + hom[2, 2]
    return (hom[0, 0] * x + hom[0, 1] * y + hom[0, 2]) / d, (hom[1, 0] * x + hom[1, 1] * y + hom[1, 2]) / d
