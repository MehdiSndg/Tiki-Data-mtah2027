"""PFF FC FIFA World Cup 2022 open data (64 games) -> TrackingData at 25 fps (11 Nov).

Source: the request form of PFF FC (the user obtained the shared folder link on 11 Nov 2026; credit PFF FC).
Files under ROOT: tracking/<id>.jsonl.bz2 (broadcast tracking, 29.97 fps, metres, origin at the centre
spot), events/<id>.json, metadata/<id>.json, rosters/<id>.json. Fetched by outputs/pff_download.sh.

Tracking: the smoothed player tracks (off-camera players are estimated by the provider; kept, our
corruption copies thin them like the target); the ball only where the provider marks it VISIBLE
(ATD-like gaps). 29.97 -> 25 fps by linear interpolation between neighbouring source frames; a value is
NaN unless both neighbours exist.

Labels from the game events (events/<id>.json, `gameEvents`):
  OTB / *KICKOFF  a player is on the ball: his team has the ball from the start of the event
  OUT             ball out of play / whistle: dead from the start of the event until the next OTB
  END             end of the period: unknown afterwards
  SUB / others    ignored
Before the first event of a period the label is X. Like DFL, possession changes on single touches; the
training target suppresses short returns (pipeline/skillcorner_src.py).
"""

import bz2
import json

import numpy as np

from ..constants import A, D, H, X
from ..data import TrackingData
from ..paths import EXTERNAL

ROOT = EXTERNAL / "pff"
FPS = 25


def games():
    """Ids (str) with all four files, sorted."""
    out = []
    for f in sorted((ROOT / "tracking").glob("*.jsonl.bz2")):
        g = f.name.split(".")[0]
        if all((ROOT / k / f"{g}.json").is_file() for k in ("events", "metadata", "rosters")):
            out.append(g)
    return out


def _meta(game):
    m = json.loads((ROOT / "metadata" / f"{game}.json").read_text(encoding="utf-8"))
    return m[0] if isinstance(m, list) else m


def read_source(game):
    """Source-rate arrays: time (S,) video seconds, period (S,), players (S, N, 2) centre-origin metres,
    ball (S, 2), team (N,), meta. Frames missing in the file are NaN / period 0."""
    rows = []
    jerseys = ({}, {})
    with bz2.open(ROOT / "tracking" / f"{game}.jsonl.bz2", "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            b = fr.get("ballsSmoothed") or {}
            ball = (b["x"], b["y"]) if b.get("visibility") == "VISIBLE" and b.get("x") is not None else (np.nan, np.nan)
            pl = []
            for side, key in enumerate(("homePlayersSmoothed", "awayPlayersSmoothed")):
                for p in fr.get(key) or []:
                    if p.get("x") is None or p.get("jerseyNum") is None:
                        continue
                    j = jerseys[side].setdefault(str(p["jerseyNum"]), len(jerseys[side]))
                    pl.append((side, j, p["x"], p["y"]))
            rows.append((int(fr["frameNum"]), float(fr["videoTimeMs"]) / 1000.0, int(fr.get("period") or 0), ball, pl))
    num = np.array([r[0] for r in rows], np.int64)
    S = int(num.max() - num.min() + 1)
    pos = num - num.min()
    n_home, n_away = len(jerseys[0]), len(jerseys[1])
    m = _meta(game)
    fps = float(m["fps"])
    time = (np.arange(S) - pos[0]) / fps + rows[0][1]
    period = np.zeros(S, np.int8)
    ball = np.full((S, 2), np.nan, np.float32)
    players = np.full((S, n_home + n_away, 2), np.nan, np.float32)
    for (_, t, per, b, pl), i in zip(rows, pos):
        time[i], period[i], ball[i] = t, per, b
        for side, j, x, y in pl:
            players[i, j + side * n_home] = (x, y)
    team = np.r_[np.zeros(n_home, np.int8), np.ones(n_away, np.int8)]
    pitch = (m.get("stadium") or {}).get("pitches") or [{}]
    meta = dict(fps=fps, pitch=(float(pitch[0].get("length") or 105.0), float(pitch[0].get("width") or 68.0)))
    return time, period, players, ball, team, meta


def game_events(game):
    """[(period, start_time_s, type, home_team or None)] sorted by time."""
    ev = json.loads((ROOT / "events" / f"{game}.json").read_text(encoding="utf-8"))
    out = []
    for e in ev:
        g = e.get("gameEvents") or {}
        if g.get("gameEventType") is None or e.get("startTime") is None:
            continue
        out.append((int(g.get("period") or 0), float(e["startTime"]), str(g["gameEventType"]), g.get("homeTeam")))
    return sorted(out, key=lambda r: r[1])


def event_labels(events, time, period):
    """Source-rate H/A/D/X labels from the game events (module docstring)."""
    lab = np.full(len(time), X, np.int8)
    for per in sorted(set(period.tolist()) - {0}):
        rows = np.flatnonzero(period == per)
        a, b = rows[0], rows[-1] + 1
        state, start = X, a
        for p, t, typ, home in events:
            if p != per:
                continue
            i = int(np.clip(np.searchsorted(time, t), a, b))
            if typ == "OTB" or typ.endswith("KICKOFF"):
                if home is None:
                    continue
                new = H if home else A
            elif typ == "OUT":
                new = D
            elif typ == "END":
                new = X
            else:
                continue
            lab[start:i] = state
            state, start = new, i
        lab[start:b] = state
    lab[period == 0] = X
    return lab


def resample(time, period, players, ball, labels, src_fps):
    """src_fps -> 25 fps (same rules as io.skillcorner.upsample)."""
    n = int((len(time) - 1) * FPS / src_fps) + 1
    s = np.arange(n) * (src_fps / FPS)
    lo = np.minimum(np.floor(s).astype(int), len(time) - 1)
    hi = np.minimum(lo + 1, len(time) - 1)
    w = (s - lo).astype(np.float32)
    near = np.where(w > 0.5, hi, lo)

    def lerp(x):
        ww = w.reshape((-1,) + (1,) * (x.ndim - 1))
        return (x[lo] * (1 - ww) + x[hi] * ww).astype(np.float32)

    return np.arange(n, dtype=np.int64), period[near].copy(), lerp(players), lerp(ball), labels[near].copy()


def load_game(game):
    time, period, players, ball, team, meta = read_source(game)
    labels = event_labels(game_events(game), time, period)
    frame, period, players, ball, labels = resample(time, period, players, ball, labels, meta["fps"])
    L, W = meta["pitch"]
    shift = np.array([L / 2, W / 2], np.float32)                     # centre origin -> corner origin
    return TrackingData(name=f"pff_{game}", fps=FPS, pitch=(L, W), frame=frame, period=period,
                        players=players + shift, ball=ball + shift, team=team, labels=labels,
                        meta={"source": "pff_fc_wc2022", "game": str(game)})


def cached_game(game, rebuild=False):
    name = f"pff_{game}"
    if not rebuild and TrackingData.exists(name):
        return TrackingData.load(name)
    data = load_game(game)
    data.save()
    return data
