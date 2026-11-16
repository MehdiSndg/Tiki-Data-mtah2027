"""SkillCorner open data (A-League 2024/25, 20 games, MIT licence) -> TrackingData at 25 fps.

Source: github.com/SkillCorner/opendata (credit SkillCorner). Files per game under ROOT/<id>/:
_match.json (rosters, pitch size), _tracking_extrapolated.jsonl (broadcast tracking, 10 fps, metres,
origin at the centre spot), _phases_of_play.csv. Downloaded by tools/fetch_skillcorner.sh; a tracking
file is used only when its size equals the Git LFS pointer size (expected_sizes.json).

Tracking: off-camera players are extrapolated (kept: our corruption copies thin them like the target);
the ball is kept only where `is_detected` is true (extrapolated ball positions are dropped, which gives
ATD-like ball gaps). 10 -> 25 fps by linear interpolation between neighbouring source frames; a value
is NaN unless both neighbours exist.

Labels (README: "Phases of play are only defined when the ball is in play. When the ball is out of
play there is no phase of play"):  D outside every phase within a period; inside a phase the team in
possession of that phase (H home / A away). Phase-level possession is coarser than the target's
(no short touches), see docs/research/skillcorner_plan_2026-11-04.md.
"""

import json

import numpy as np
import pandas as pd

from ..constants import A, D, H, X
from ..data import TrackingData
from ..paths import CACHE, EXTERNAL

SRC_FPS, FPS = 10, 25


def root():
    ext = EXTERNAL / "skillcorner"
    return ext if (ext / "expected_sizes.json").is_file() else CACHE / "skillcorner"


def games(base=None):
    """Ids (str) whose tracking file is completely downloaded, in matches.json order."""
    base = base or root()
    sizes = json.loads((base / "expected_sizes.json").read_text(encoding="utf-8"))
    out = []
    for g, size in sizes.items():
        f = base / g / f"{g}_tracking_extrapolated.jsonl"
        if f.is_file() and f.stat().st_size == size and (base / g / f"{g}_phases_of_play.csv").is_file() \
                and (base / g / f"{g}_match.json").is_file():
            out.append(g)
    return out


def read_source(game, base=None):
    """Source-rate arrays: frame (S,), period (S,), players (S, N, 2) centre-origin metres,
    ball (S, 2), team (N,), meta."""
    base = (base or root()) / str(game)
    m = json.loads((base / f"{game}_match.json").read_text(encoding="utf-8"))
    home = m["home_team"]["id"]
    pids = [p["id"] for p in m["players"]]
    index = {p: i for i, p in enumerate(pids)}
    team = np.array([0 if p["team_id"] == home else 1 for p in m["players"]], np.int8)
    frames, periods, balls, rows = [], [], [], []
    with open(base / f"{game}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            frames.append(fr["frame"])
            periods.append(fr["period"] or 0)
            b = fr.get("ball_data") or {}
            balls.append((b["x"], b["y"]) if b.get("is_detected") and b.get("x") is not None else (np.nan, np.nan))
            rows.append([(index[p["player_id"]], p["x"], p["y"]) for p in fr.get("player_data") or []
                         if p.get("player_id") in index])
    frame = np.asarray(frames, np.int64)
    if np.any(np.diff(frame) != 1):
        raise ValueError(f"skillcorner {game}: source frames are not contiguous")
    players = np.full((len(frame), len(pids), 2), np.nan, np.float32)
    for t, r in enumerate(rows):
        if r:
            a = np.asarray(r, np.float64)
            players[t, a[:, 0].astype(int)] = a[:, 1:]
    meta = dict(home_id=home, pitch=(float(m["pitch_length"]), float(m["pitch_width"])))
    return frame, np.asarray(periods, np.int8), players, np.asarray(balls, np.float32), team, meta


def phase_labels(phases, frame, period, home_id):
    """Source-rate H/A/D labels: D in a period outside every phase, the phase's team inside it."""
    lab = np.where(period > 0, D, X).astype(np.int8)
    for r in phases.itertuples():
        a = int(np.clip(r.frame_start - frame[0], 0, len(frame)))
        b = int(np.clip(r.frame_end - frame[0] + 1, 0, len(frame)))
        seg = slice(a, b)
        lab[seg] = np.where(period[seg] > 0, H if r.team_in_possession_id == home_id else A, X)
    return lab


def upsample(frame, period, players, ball, labels):
    """10 -> 25 fps. Positions: linear interpolation where both neighbours exist; period and labels:
    the nearer source frame (ties to the earlier one)."""
    n = (len(frame) - 1) * FPS // SRC_FPS + 1
    s = np.arange(n) * (SRC_FPS / FPS)
    lo = np.floor(s).astype(int)
    hi = np.minimum(lo + 1, len(frame) - 1)
    w = (s - lo).astype(np.float32)
    near = np.where(w > 0.5, hi, lo)

    def lerp(x):
        ww = w.reshape((-1,) + (1,) * (x.ndim - 1))
        return (x[lo] * (1 - ww) + x[hi] * ww).astype(np.float32)   # NaN if either neighbour is NaN

    return (np.arange(n, dtype=np.int64), period[near].copy(), lerp(players), lerp(ball), labels[near].copy())


def load_game(game, base=None):
    frame, period, players, ball, team, meta = read_source(game, base)
    phases = pd.read_csv((base or root()) / str(game) / f"{game}_phases_of_play.csv")
    labels = phase_labels(phases, frame, period, meta["home_id"])
    frame, period, players, ball, labels = upsample(frame, period, players, ball, labels)
    L, W = meta["pitch"]
    shift = np.array([L / 2, W / 2], np.float32)                     # centre origin -> corner origin
    return TrackingData(name=f"skillcorner_{game}", fps=FPS, pitch=(L, W), frame=frame, period=period,
                        players=players + shift, ball=ball + shift, team=team, labels=labels,
                        meta={"source": "skillcorner_open", "game": str(game), "n_phases": int(len(phases))})


def cached_game(game, rebuild=False):
    name = f"skillcorner_{game}"
    if not rebuild and TrackingData.exists(name):
        return TrackingData.load(name)
    data = load_game(game)
    data.save()
    return data
