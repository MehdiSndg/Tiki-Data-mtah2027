"""Metrica Sports open sample data (3 anonymised games) -> TrackingData with event-derived labels.

Why: our ground truth comes from Metrica's own corrected tracking (ball status / owning team).
DFL labels follow DFL's touch-level possession; Metrica's own event tagging is the closest open
proxy for Metrica's possession semantics, and the sample tracking is Metrica's own tracking.
Source: github.com/metrica-sports/sample-data ("please be responsible ... acknowledge the source");
stored under more-than-a-hack-2027/data/external/metrica (never committed).

Labels from events (frames are synchronised with tracking, 25 fps):
  D   from a dead-ball trigger -- BALL OUT (start), FAULT RECEIVED (start), a SHOT whose subtype
      contains GOAL or OUT (end) -- until the start frame of the next SET PIECE; a SET PIECE with
      no tagged trigger is dead from the end of the previous event; and every played frame
      before the first event of a period (pre kick-off).
  H/A possession changes at the start frame of the other team's first CONTROL event (PASS,
      RECOVERY, SET PIECE, SHOT, BALL LOST start) -- Metrica's "first controlled touch".
  X   loose ball: from the end of a BALL LOST until the next control event (by either team), and
      anything after the last event of a period. X frames are masked in training/evaluation.

Ball-status labels (11 Nov, `metrica_labels: ball`, ball_status_labels): the dead flag is taken from the
ball channel instead of the events. In this tracking the ball is empty while it is out of play: 97-100 %
of the SET PIECE events are preceded by a ball gap, a FAULT RECEIVED is followed by one within a frame,
and a BALL OUT event spans kick -> out (median 33-50 frames) with the gap starting one frame after its
END. The event rule above starts the dead ball at the kick and, after a trigger whose restart is not
tagged SET PIECE, keeps live play dead until the next SET PIECE (12-24 thousand "dead" frames per game
with the ball tracked, 59-84 % of them faster than 2 m/s). The competition's ground truth takes its dead
flag from the provider's ball status, so the ball channel is the closer analogue.
  D   ball gaps of at least `min_dead` frames (real dead balls: p10 102-108 frames; tracking drop-outs
      in live play: median 2-8 frames)
  H/A the event rule, applied at all times (no dead-ball state machine); shorter ball gaps keep the team
  X   loose ball as above; frames between the end of a dead ball and the next control event
"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..constants import A, D, H, X
from ..data import TrackingData
from ..paths import EXTERNAL

ROOT = EXTERNAL / "metrica" / "data"
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
CONTROL = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def _read_team_csv(path):
    """(period, frame, positions (T, n, 2) normalised, ball (T, 2) or None, player names)."""
    raw = pd.read_csv(path, header=2)
    cols = list(raw.columns)
    period = raw["Period"].to_numpy(np.int8)
    frame = raw["Frame"].to_numpy(np.int64)
    names, xy = [], []
    ball = None
    for i, c in enumerate(cols):
        if c.startswith("Player") or c == "Ball":
            pair = raw.iloc[:, [i, i + 1]].to_numpy(np.float32)
            if c == "Ball":
                ball = pair
            else:
                names.append(c)
                xy.append(pair)
    return period, frame, np.stack(xy, axis=1), ball, names


def _events_csv(path):
    e = pd.read_csv(path)
    return pd.DataFrame({"team": e["Team"].str.strip(), "type": e["Type"].str.strip().str.upper(),
                         "subtype": e["Subtype"].fillna("").astype(str).str.strip().str.upper(),
                         "period": e["Period"].astype(int), "start": e["Start Frame"].astype(int),
                         "end": e["End Frame"].astype(int)})


def _events_json(path, home_team_id):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ev in d["data"]:
        subs = ev.get("subtypes") or []
        if isinstance(subs, dict):
            subs = [subs]
        rows.append({"team": "Home" if ev["team"]["id"] == home_team_id else "Away",
                     "type": str(ev["type"]["name"]).strip().upper(),
                     "subtype": "-".join(str(s.get("name", "")).strip().upper() for s in subs),
                     "period": int(ev["period"]), "start": int(ev["start"]["frame"] or 0),
                     "end": int((ev.get("end") or {}).get("frame") or ev["start"]["frame"] or 0)})
    return pd.DataFrame(rows)


def derive_labels(events, frame, period):
    """Per-frame H/A/D/X labels from the event table (see module docstring)."""
    n = len(frame)
    pos = {f: i for i, f in enumerate(frame)}
    idx = lambda f: pos.get(int(f), int(np.searchsorted(frame, f)))  # noqa: E731
    lab = np.full(n, X, np.int8)
    ev = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    for per in sorted(set(period.tolist()) - {0}):
        rows = np.flatnonzero(period == per)
        e = ev[ev["period"] == per]
        if len(e) == 0:
            continue
        a, b = rows[0], rows[-1] + 1
        first = max(a, min(b, idx(e["start"].iloc[0])))
        lab[a:first] = D                                           # before the kick-off
        cur, cur_from, dead_from, loose_from, prev_end = None, first, None, None, first
        for _, r in e.iterrows():
            s = max(a, min(b, idx(r["start"])))
            en = max(a, min(b, idx(r["end"]))) if r["end"] > 0 else s
            team = H if r["team"] == "Home" else A
            if dead_from is None and r["type"] == "SET PIECE" and s > first:
                # a restart without a tagged trigger (throw-in / free kick with no BALL OUT or
                # FAULT RECEIVED): the ball was dead from the end of the previous event
                stop = min(prev_end, s)
                if cur is not None:
                    lab[cur_from:stop] = cur
                cur, loose_from, dead_from = None, None, stop
            prev_end = max(prev_end, en, s)
            if dead_from is not None:
                if r["type"] == "SET PIECE":
                    lab[dead_from:s] = D
                    dead_from = None
                    cur, cur_from, loose_from = team, s, None
                continue
            if r["type"] in CONTROL:
                if loose_from is not None:                         # loose ball ends here
                    if cur is not None:
                        lab[cur_from:loose_from] = cur
                    cur, cur_from, loose_from = team, s, None
                elif team != cur:
                    if cur is not None:
                        lab[cur_from:s] = cur
                    cur, cur_from = team, s
                if r["type"] == "BALL LOST":
                    loose_from = max(s, en)
            trig = (r["type"] in ("BALL OUT", "FAULT RECEIVED")
                    or (r["type"] == "SHOT" and ("GOAL" in r["subtype"] or "OUT" in r["subtype"])))
            if trig:
                stop = en if r["type"] == "SHOT" else s
                if cur is not None:
                    lab[cur_from:stop] = cur
                cur, loose_from = None, None
                dead_from = stop
        if cur is not None and dead_from is None:
            last = max(a, min(b, idx(e["end"].max())))
            lab[cur_from:(loose_from if loose_from is not None else last)] = cur
    return lab


def _events(game):
    if game in (1, 2):
        return _events_csv(ROOT / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv")
    g = ROOT / "Sample_Game_3"
    home_id = next(ET.parse(g / "Sample_Game_3_metadata.xml").getroot().iter("Team")).get("id")
    return _events_json(g / "Sample_Game_3_events.json", home_id)


def _true_runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def ball_status_labels(events, frame, period, ball_missing, min_dead=50):
    """Per-frame H/A/D/X labels with the dead flag from the ball channel (module docstring)."""
    n = len(frame)
    pos = {f: i for i, f in enumerate(frame)}
    idx = lambda f: pos.get(int(f), int(np.searchsorted(frame, f)))  # noqa: E731
    lab = np.full(n, X, np.int8)
    dead = np.zeros(n, bool)
    ev = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    for per in sorted(set(period.tolist()) - {0}):
        rows = np.flatnonzero(period == per)
        a, b = rows[0], rows[-1] + 1
        for c, d in _true_runs(ball_missing[a:b]):
            if d - c >= min_dead:
                dead[a + c:a + d] = True
        e = ev[(ev["period"] == per) & ev["type"].isin(CONTROL)]
        if len(e) == 0:
            continue
        cur, cur_from, loose_from = None, a, None
        starts = []
        for _, r in e.iterrows():
            s = max(a, min(b, idx(r["start"])))
            en = max(a, min(b, idx(r["end"]))) if r["end"] > 0 else s
            team = H if r["team"] == "Home" else A
            starts.append(s)
            if loose_from is not None:                             # loose ball ends here
                if cur is not None:
                    lab[cur_from:loose_from] = cur
                cur, cur_from, loose_from = team, s, None
            elif team != cur:
                if cur is not None:
                    lab[cur_from:s] = cur
                cur, cur_from = team, s
            if r["type"] == "BALL LOST":
                loose_from = max(s, en)
        if cur is not None:
            last = max(a, min(b, idx(ev.loc[ev["period"] == per, "end"].max())))
            lab[cur_from:(loose_from if loose_from is not None else last)] = cur
        starts = np.array(sorted(starts))
        for c, d in _true_runs(dead[a:b]):                         # restart later than the gap's end: unknown
            j = np.searchsorted(starts, a + d - 2)
            nxt = starts[j] if j < len(starts) else b
            if nxt > a + d:
                lab[a + d:nxt] = X
    lab[dead] = D
    lab[period == 0] = X
    return lab


def ball_labels(game, min_dead=50, rebuild=False):
    """Cached ball-status labels of a game (cache/metrica_<g>_ball<min_dead>.npy)."""
    from ..paths import CACHE
    path = CACHE / f"metrica_{game}_ball{int(min_dead)}.npy"
    if path.is_file() and not rebuild:
        return np.load(path)
    data = cached_game(game)
    lab = ball_status_labels(_events(game), np.asarray(data.frame), np.asarray(data.period),
                             np.isnan(data.ball[:, 0]), int(min_dead))
    np.save(path, lab)
    return lab


def load_game(game):
    if game in (1, 2):
        g = ROOT / f"Sample_Game_{game}"
        ph, fh, xyh, ball, _ = _read_team_csv(g / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
        pa, fa, xya, _, _ = _read_team_csv(g / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
        assert np.array_equal(fh, fa), "home/away frames differ"
        frame, period = fh, ph
        players = np.concatenate([xyh, xya], axis=1)
        team = np.r_[np.zeros(xyh.shape[1], np.int8), np.ones(xya.shape[1], np.int8)]
        events = _events_csv(g / f"Sample_Game_{game}_RawEventsData.csv")
    else:
        from kloppy import metrica
        g = ROOT / "Sample_Game_3"
        ds = metrica.load_tracking_epts(meta_data=str(g / "Sample_Game_3_metadata.xml"),
                                        raw_data=str(g / "Sample_Game_3_tracking.txt"), coordinates="metrica")
        home, away = ds.metadata.teams
        plist = list(home.players) + list(away.players)
        index = {p.player_id: i for i, p in enumerate(plist)}
        team = np.array([0] * len(home.players) + [1] * len(away.players), np.int8)
        n = len(ds.records)
        players = np.full((n, len(plist), 2), np.nan, np.float32)
        ball = np.full((n, 2), np.nan, np.float32)
        frame = np.zeros(n, np.int64)
        period = np.zeros(n, np.int8)
        for t, fr in enumerate(ds.records):
            frame[t], period[t] = fr.frame_id, fr.period.id
            if fr.ball_coordinates is not None and fr.ball_coordinates.x is not None:
                ball[t] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
            for pl, pdata in fr.players_data.items():
                c = pdata.coordinates
                if c is not None and c.x is not None:
                    players[t, index[pl.player_id]] = (c.x, c.y)
        root = ET.parse(g / "Sample_Game_3_metadata.xml").getroot()
        home_id = next(root.iter("Team")).get("id")
        events = _events_json(g / "Sample_Game_3_events.json", home_id)
    scale = np.array(PITCH, np.float32)
    labels = derive_labels(events, frame, period)
    return TrackingData(name=f"metrica_{game}", fps=25, pitch=PITCH, frame=frame, period=period,
                        players=players * scale, ball=ball * scale, team=team, labels=labels,
                        meta={"source": "metrica_open", "game": game, "n_events": int(len(events))})


def cached_game(game, rebuild=False):
    name = f"metrica_{game}"
    if not rebuild and TrackingData.exists(name):
        return TrackingData.load(name)
    data = load_game(game)
    data.save()
    return data
