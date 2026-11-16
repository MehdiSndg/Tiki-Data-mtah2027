"""External tracking converted to the Match format (mtah/match.py), with labels derived from it.

    from mtah.external import from_metrica_sample
    m = from_metrica_sample(1)

    m = from_dfl("DFL-MAT-J03WOH")

Metrica sample games 1-2: tracking from the public sample repository, labels derived from Metrica's
own manually annotated events following Metrica's definitions (literature/README.md). The labels are
OUR derivation, a proxy for the challenge's definition, not ground truth. The data is never
committed; the licence of the sample data is unclear (see notes/BULGULAR.md).

DFL open data (7 Bundesliga matches, CC-BY 4.0; Bassek et al.): positions via floodlight, labels
from DFL's BallStatus / BallPossession, brought towards Metrica's definition by merge_short (DFL
flips possession on every touch; analysis/g3_dfl_conversion.py chose 1.6 s, external to external).
Each squad member is one slot (substitutes are NaN until they come on). The halves are joined with
one X frame between them. y keeps DFL's orientation; nothing here depends on its sign.

SkillCorner open data (10 A-League 2024-25 matches, MIT; github.com/SkillCorner/opendata, git LFS):
broadcast tracking at 10 fps, brought to 25 fps (to_25fps). Labels from SkillCorner's automatic phases of
play (ball in play): outside every phase is a stoppage, except a gap shorter than GAP_S inside one period,
which keeps the previous team. The team is the per-frame possession group (the team of the player in
possession), carried forward while nobody has the ball, the phase's team until somebody does (fill_group):
329 spells a match (2 matches) against the phases' 249, near DFL's 308-388. Automatic, not ground truth. detected_only drops
SkillCorner's extrapolated positions (players and ball off camera), the way our ATD loses them.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.match import keepers_and_sign  # noqa: E402

# Params
METRICA = os.path.join(ROOT, "data", "external", "metrica-sample-data", "data")
METRICA_PITCH = (105.0, 68.0)
DFL = os.path.join(ROOT, "data", "external", "dfl")
SKILLCORNER = os.path.join(ROOT, "data", "external", "skillcorner", "data", "matches")
SOCCERTRACK = os.environ.get("MTAH_SOCCERTRACK", os.path.join(ROOT, "data", "external", "soccertrack"))
ST_ACTIONS = {"PASS", "DRIVE", "HIGH PASS", "CROSS"}
SC_FPS = 10
GAP_S = 2.0
CACHE = os.path.join(ROOT, "cache")
FPS = 25
TEAM = {"Home": pm.H, "Away": pm.A}
OWNING = {"PASS", "RECOVERY", "SET PIECE", "SHOT"}
RESTART = {"SET PIECE"}


def metrica_event_labels(events, n_frames):
    """Per-frame H/A/D/X codes (index = Metrica frame number, 0..n_frames) from Metrica's events.

      possession  team of the last PASS, RECOVERY, SET PIECE or SHOT (a BALL LOST hands possession
                  to the opponent only through the opponent's RECOVERY that follows it)
      dead starts BALL OUT (at its end frame: the ball crosses the line); a SHOT whose subtype
                  contains OUT or GOAL (end frame); FAULT RECEIVED (start); a BALL LOST or BALL OUT
                  with a FORCED / VOLUNTARY intervention (start); CARD (start)
      dead ends   the next SET PIECE, a PASS with subtype GOAL KICK, or a RECOVERY (Metrica: "brings
                  the ball back into play via a dead-ball situation that is not a SET PIECE")
    """
    ev = events.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    codes = np.full(n_frames + 1, pm.X, np.int8)
    for period in sorted(ev["Period"].unique()):
        e = ev[ev["Period"] == period]
        s0, s1 = int(e["Start Frame"].min()), int(e["End Frame"].max())
        seq = np.full(s1 - s0 + 1, pm.D, np.int8)     # before the first owning event: kick-off pending
        dead_from = None
        marks = []                                     # (frame, kind, value)
        for _, r in e.iterrows():
            typ, sub = r["Type"], str(r["Subtype"]) if pd.notna(r["Subtype"]) else ""
            start, end = int(r["Start Frame"]), int(r["End Frame"]) if r["End Frame"] > 0 else int(r["Start Frame"])
            restart = typ in RESTART or (typ == "PASS" and "GOAL KICK" in sub) or typ == "RECOVERY"
            if restart and dead_from is not None:
                marks.append((dead_from, "dead_until", start))
                dead_from = None
            if typ in OWNING:
                marks.append((start, "own", TEAM[r["Team"]]))
            if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
                dead_from = end if dead_from is None else dead_from
            if typ in ("FAULT RECEIVED", "CARD") or ("FORCED" in sub or "VOLUNTARY" in sub):
                dead_from = start if dead_from is None else dead_from
        own = [(f, v) for f, k, v in marks if k == "own"]
        k, cur = 0, pm.D
        for i, f in enumerate(range(s0, s1 + 1)):
            while k < len(own) and own[k][0] <= f:
                cur = own[k][1]
                k += 1
            seq[i] = cur
        for f, kind, until in marks:
            if kind == "dead_until":
                seq[f - s0:until - s0] = pm.D
        codes[s0:s1 + 1] = seq
    return codes


def _metrica_tracking(path):
    """(frame, period, {player: (x, y)}, (ball x, ball y)) from one RawTrackingData CSV."""
    tr = pd.read_csv(path, skiprows=2)
    cols = list(tr.columns)
    players = {c: (tr[c].values, tr[cols[i + 1]].values) for i, c in enumerate(cols) if c.startswith("Player")}
    i = cols.index("Ball")
    return tr["Frame"].values, tr["Period"].values, players, (tr["Ball"].values, tr[cols[i + 1]].values)


def _metrica_game3():
    """Sample game 3 is a FIFA/EPTS export (the format of our ATD) with JSON events, and its player
    channels change with substitutions, so kloppy reads it. Coordinates come back untouched (0-1,
    top-left). Returns the same pieces as games 1-2: frame, period, tracks, ball, events table."""
    import json
    from kloppy import metrica
    folder = os.path.join(METRICA, "Sample_Game_3")
    ds = metrica.load_tracking_epts(meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
                                    raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    df = ds.to_df()
    tracks = {}
    for t, side in zip(ds.metadata.teams, ("home", "away")):
        tracks[side] = [(df[p.player_id + "_x"].values, df[p.player_id + "_y"].values) for p in t.players
                        if p.player_id + "_x" in df.columns]
    rows = []
    for e in json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]:
        sub = e["subtypes"]
        sub = "-".join(x["name"] for x in sub) if isinstance(sub, list) else (sub["name"] if sub else np.nan)
        rows.append({"Team": "Home" if e["team"]["name"] == ds.metadata.teams[0].name else "Away",
                     "Type": e["type"]["name"], "Subtype": sub, "Period": e["period"],
                     "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"]})
    return (df["frame_id"].values, df["period_id"].values, tracks["home"], tracks["away"],
            (df["ball_x"].values, df["ball_y"].values), pd.DataFrame(rows))


def from_metrica_sample(game):
    if game == 3:
        frame, period, home, away, ball, events = _metrica_game3()
        return _metrica_match(game, frame, period, home, away, ball, events)
    folder = os.path.join(METRICA, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{{}}_Team.csv".format(game))
    frame, period, home, ball = _metrica_tracking(stem.format("Home"))
    frame_a, _, away, _ = _metrica_tracking(stem.format("Away"))
    assert np.array_equal(frame, frame_a)
    events = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
    return _metrica_match(game, frame, period, list(home.values()), list(away.values()), ball, events)


def _metrica_match(game, frame, period, home, away, ball, events):
    n = int(frame.max()) + 1                            # index = Metrica frame number; frame 0 is empty
    tracks = list(home) + list(away)
    px = np.full((n, len(tracks)), np.nan, np.float32)
    py = px.copy()
    for k, (x, y) in enumerate(tracks):
        px[frame, k], py[frame, k] = x, y
    slot_team = np.r_[np.zeros(len(home)), np.ones(len(away))].astype(np.int8)
    bx = np.full(n, np.nan, np.float32)
    by = bx.copy()
    bx[frame], by[frame] = ball
    half = np.zeros(n, np.int8)
    half[frame] = np.where(period <= 2, period, 0)
    labels = metrica_event_labels(events, n - 1)
    gk, sign = keepers_and_sign(px, half, slot_team)
    return {"name": "metrica{}".format(game), "px": px, "py": py, "slot_team": slot_team, "gk": gk,
            "bx": bx, "by": by, "half": half, "labels": labels, "sign": sign, "pitch": METRICA_PITCH}


def merge_short(codes, max_s):
    """A copy with every possession spell shorter than max_s seconds, with the OTHER team on both
    sides of it, given back to that team (Metrica ignores a touch with no significant impact)."""
    out = codes.copy()
    if max_s <= 0:
        return out
    segs = pm.segments(out, [(0, len(out))])
    for i in range(1, len(segs) - 1):
        a, b, c = segs[i]
        prev_c, next_c = segs[i - 1][2], segs[i + 1][2]
        if c in (pm.H, pm.A) and prev_c == next_c and prev_c in (pm.H, pm.A) and prev_c != c \
                and (b - a) / FPS < max_s:
            out[a:b] = prev_c
    return out


def _dfl_raw(match_id):
    """Positions and raw labels of one DFL match, halves joined by one X frame; cached."""
    path = os.path.join(CACHE, "dfl_{}.npz".format(match_id))
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    from floodlight.io.dfl import read_position_data_xml
    pos = glob.glob(os.path.join(DFL, "*positions*{}.xml".format(match_id)))[0]
    info = glob.glob(os.path.join(DFL, "*matchinformation*{}.xml".format(match_id)))[0]
    xy, poss, status, _, pitch = read_position_data_xml(pos, info)
    (x0, x1), (y0, y1) = pitch.xlim, pitch.ylim
    parts = []
    for h, sec in ((1, "firstHalf"), (2, "secondHalf")):
        H, A, B = xy[sec]["Home"].xy, xy[sec]["Away"].xy, xy[sec]["Ball"].xy
        P = np.concatenate([H, A], axis=1)
        s, p = status[sec].code, poss[sec].code
        parts.append({"px": (P[:, 0::2] - x0) / (x1 - x0), "py": (P[:, 1::2] - y0) / (y1 - y0),
                      "bx": (B[:, 0] - x0) / (x1 - x0), "by": (B[:, 1] - y0) / (y1 - y0),
                      "half": np.full(len(s), h, np.int8),
                      "raw": np.where(s == 0, pm.D, np.where(p == 1, pm.H, pm.A)).astype(np.int8)})
        n_home = H.shape[1] // 2
    sep = {"px": np.full((1, parts[0]["px"].shape[1]), np.nan), "py": np.full((1, parts[0]["px"].shape[1]), np.nan),
           "bx": [np.nan], "by": [np.nan], "half": [0], "raw": [pm.X]}
    out = {k: np.concatenate([parts[0][k], sep[k], parts[1][k]]) for k in parts[0]}
    for k in ("px", "py", "bx", "by"):
        out[k] = out[k].astype(np.float32)
    out["half"], out["raw"] = out["half"].astype(np.int8), out["raw"].astype(np.int8)
    out["slot_team"] = np.r_[np.zeros(n_home), np.ones(out["px"].shape[1] - n_home)].astype(np.int8)
    out["pitch"] = np.array([x1 - x0, y1 - y0])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def from_dfl(match_id, merge_s=1.6):
    d = _dfl_raw(match_id)
    labels = d["raw"].copy()
    for h in (1, 2):
        idx = np.flatnonzero(d["half"] == h)
        labels[idx] = merge_short(d["raw"][idx], merge_s)
    gk, sign = keepers_and_sign(d["px"], d["half"], d["slot_team"])
    return {"name": match_id, "px": d["px"], "py": d["py"], "slot_team": d["slot_team"], "gk": gk,
            "bx": d["bx"], "by": d["by"], "half": d["half"], "labels": labels, "sign": sign,
            "pitch": tuple(float(v) for v in d["pitch"])}


def to_25fps(a, n):
    """n frames at FPS from a series sampled at SC_FPS: integer codes take the sample at or before each
    frame; floats are interpolated linearly, NaN when a neighbouring sample is NaN."""
    src = np.arange(n) * (SC_FPS / FPS)
    i0 = np.minimum(np.floor(src).astype(int), len(a) - 1)
    if np.issubdtype(a.dtype, np.integer):
        return a[i0]
    i1 = np.minimum(i0 + 1, len(a) - 1)
    w = (src - i0)[(slice(None),) + (None,) * (a.ndim - 1)]
    out = a[i0] * (1 - w) + a[i1] * w
    exact = w == 0
    return np.where(exact, a[i0], out).astype(a.dtype)


def skillcorner_labels(phases, period, home_id):
    """Codes at SC_FPS from the phases of play (half-open [frame_start, frame_end)): the phase's team,
    D between phases, X outside the periods; a gap shorter than GAP_S within one period keeps the team
    of the phase before it."""
    codes = np.where(period > 0, pm.D, pm.X).astype(np.int8)
    ph = phases.sort_values("frame_start")
    prev = None
    for a, b, team in ph[["frame_start", "frame_end", "team_in_possession_id"]].itertuples(index=False):
        a, b = int(a), int(b)
        c = pm.H if team == home_id else pm.A
        if prev is not None and 0 < a - prev[0] < GAP_S * SC_FPS and period[prev[0] - 1] == period[min(a, len(period) - 1)]:
            codes[prev[0]:a] = prev[1]
        codes[a:b] = np.where(period[a:b] > 0, c, pm.X)
        prev = (b, c)
    return codes


def fill_group(phase_codes, group):
    """The alive frames of phase_codes given to the last team with a player in possession (group: H, A or
    -1 for nobody), carried forward and reset at every stoppage; the phase's team until the first one."""
    out = phase_codes.copy()
    last = -1
    for i, c in enumerate(phase_codes):
        if c not in (pm.H, pm.A):
            last = -1
            continue
        if group[i] >= 0:
            last = group[i]
        if last >= 0:
            out[i] = last
    return out


def _skillcorner_raw(match_id, detected_only):
    path = os.path.join(CACHE, "skillcorner_{}{}.npz".format(match_id, "_det" if detected_only else ""))
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    import json
    d = os.path.join(SKILLCORNER, match_id)
    meta = json.load(open(os.path.join(d, "{}_match.json".format(match_id))))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    players = sorted(meta["players"], key=lambda p: (p["team_id"] != home, p["id"]))
    slot = {p["id"]: k for k, p in enumerate(players)}
    frames = [json.loads(line) for line in open(os.path.join(d, "{}_tracking_extrapolated.jsonl".format(match_id)))]
    n10 = frames[-1]["frame"] + 1
    px = np.full((n10, len(players)), np.nan, np.float32)
    py = px.copy()
    bx, by = np.full(n10, np.nan, np.float32), np.full(n10, np.nan, np.float32)
    period = np.zeros(n10, np.int8)
    group = np.full(n10, -1, np.int8)
    for f in frames:
        i = f["frame"]
        if f["period"] not in (1, 2):
            continue
        period[i] = f["period"]
        group[i] = {"home team": pm.H, "away team": pm.A}.get(f["possession"]["group"], -1)
        b = f["ball_data"]
        if b["x"] is not None and (b["is_detected"] or not detected_only):
            bx[i], by[i] = b["x"] / L + 0.5, b["y"] / W + 0.5
        for q in f["player_data"]:
            if q["player_id"] in slot and (q["is_detected"] or not detected_only):
                px[i, slot[q["player_id"]]], py[i, slot[q["player_id"]]] = q["x"] / L + 0.5, q["y"] / W + 0.5
    labels = fill_group(skillcorner_labels(pd.read_csv(os.path.join(d, "{}_phases_of_play.csv".format(match_id))), period, home), group)
    n = int(n10 * FPS / SC_FPS)
    out = {"px": to_25fps(px, n), "py": to_25fps(py, n), "bx": to_25fps(bx, n), "by": to_25fps(by, n),
           "half": to_25fps(period, n), "labels": to_25fps(labels, n),
           "slot_team": np.array([p["team_id"] != home for p in players], np.int8), "pitch": np.array([L, W])}
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def skillcorner_ids():
    """Match ids from the download, or from the caches where only those were copied (Colab)."""
    if os.path.isdir(SKILLCORNER):
        return sorted(os.listdir(SKILLCORNER))
    return sorted({os.path.basename(q).split("_")[1].split(".")[0] for q in glob.glob(os.path.join(CACHE, "skillcorner_*.npz"))})


def from_skillcorner(match_id, detected_only=True):
    d = _skillcorner_raw(match_id, detected_only)
    gk, sign = keepers_and_sign(d["px"], d["half"], d["slot_team"])
    return {"name": "sc" + match_id, "px": d["px"], "py": d["py"], "slot_team": d["slot_team"], "gk": gk,
            "bx": d["bx"], "by": d["by"], "half": d["half"], "labels": d["labels"], "sign": sign,
            "pitch": tuple(float(v) for v in d["pitch"])}


# SoccerTrack v2 (10 university matches, CC BY 4.0; huggingface.co/datasets/atomscott/soccertrack-v2, gated):
# full-pitch panoramic tracking at 25 fps, every player every frame (gsr/), and a ball track per frame
# (ball/) that is reconstructed between annotated ball actions, with a per-frame status: 0 dead, 1 / 2 one
# team in possession, 3 loose (checked on two matches, 26 Oct: 98-100% of the 3 s after OUT events are 0;
# 92% of each side's passes and drives fall on one code). Which team is 1 differs per match and the sides
# swap at half time, so each half's side -> code map is the majority code at that side's passes and drives.


def soccertrack_labels(status, home_code):
    """Codes from the ball track's status: 0 -> D, home_code -> H, the other team code -> A; a loose ball
    (3) or an unknown status keeps the last team (the challenge's rule), or the next one at the start."""
    team = np.where(status == home_code, pm.H, np.where(np.isin(status, (1, 2)), pm.A, np.nan))
    filled = pd.Series(team).ffill().bfill().fillna(pm.D).values
    return np.where(status == 0, pm.D, filled).astype(np.int8)


def soccertrack_side_codes(events, status):
    """{"left": code, "right": code}: the majority status at each side's passes / drives in one half."""
    out = {}
    for side in ("left", "right"):
        codes = [int(status[min(e["frame"], len(status) - 1)]) for e in events if e["team"] == side and e["label"] in ST_ACTIONS]
        codes = [c for c in codes if c in (1, 2)]
        out[side] = max((1, 2), key=codes.count) if codes else None
    return out


def _soccertrack_half(mid, half, root):
    """px, py (n, slots), slot keys and sides, ball, status of one half; frame k of the half is index k - 1."""
    import json
    z = np.load(os.path.join(root, "ball", "{}_{}_ball.npz".format(mid, half)))
    n = len(z["status"])
    g = json.load(open(os.path.join(root, "gsr", mid, "{}_{}.json".format(mid, half))))
    cols, side, rows = {}, {}, []
    for a in g["annotations"]:
        if a.get("supercategory") != "object":
            continue
        at = a.get("attributes", {})
        if at.get("role") not in ("player", "goalkeeper") or at.get("team") not in ("left", "right"):
            continue
        key = at.get("player_id") or "{}-t{}".format(half, a.get("track_id"))
        k = cols.setdefault(key, len(cols))
        side.setdefault(key, at["team"])
        b = a["bbox_pitch"]
        rows.append((int(a["image_id"][-6:]) - 1, k, b["x_bottom_middle"], b["y_bottom_middle"]))
    del g
    r = np.array(rows, np.float64)
    px = np.full((n, len(cols)), np.nan, np.float32)
    py = px.copy()
    ok = r[:, 0] < n
    px[r[ok, 0].astype(int), r[ok, 1].astype(int)] = (r[ok, 2] + 52.5) / 105.0
    py[r[ok, 0].astype(int), r[ok, 1].astype(int)] = (r[ok, 3] + 34.0) / 68.0
    return {"px": px, "py": py, "keys": list(cols), "side": [side[k] for k in cols],
            "bx": (z["x"] + 52.5) / 105.0, "by": (z["y"] + 34.0) / 68.0, "status": z["status"].astype(np.int8)}


def _soccertrack_raw(mid, root=None):
    path = os.path.join(CACHE, "soccertrack_{}.npz".format(mid))
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    import json
    root = root or SOCCERTRACK
    events = json.load(open(os.path.join(root, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    halves = []
    for h, half in ((1, "1st"), (2, "2nd")):
        d = _soccertrack_half(mid, half, root)
        d["codes"] = soccertrack_side_codes([e for e in events if e["gameTime"].startswith("{} ".format(h))], d["status"])
        halves.append(d)
    keys = list(dict.fromkeys(halves[0]["keys"] + halves[1]["keys"]))
    team_of = {}
    for d in halves:
        for k, sd in zip(d["keys"], d["side"]):
            team_of.setdefault(k, 0 if d["codes"][sd] == 1 else 1)      # code 1 is "home"
    parts = []
    for h, d in enumerate(halves, 1):
        n = len(d["status"])
        px = np.full((n, len(keys)), np.nan, np.float32)
        py = px.copy()
        idx = [keys.index(k) for k in d["keys"]]
        px[:, idx], py[:, idx] = d["px"], d["py"]
        parts.append({"px": px, "py": py, "bx": d["bx"].astype(np.float32), "by": d["by"].astype(np.float32),
                      "half": np.full(n, h, np.int8), "labels": soccertrack_labels(d["status"], 1)})
    sep = {"px": np.full((1, len(keys)), np.nan, np.float32), "py": np.full((1, len(keys)), np.nan, np.float32),
           "bx": np.array([np.nan], np.float32), "by": np.array([np.nan], np.float32), "half": np.array([0], np.int8),
           "labels": np.array([pm.X], np.int8)}
    out = {k: np.concatenate([parts[0][k], sep[k], parts[1][k]]) for k in parts[0]}
    out["slot_team"] = np.array([team_of[k] for k in keys], np.int8)
    out["pitch"] = np.array([105.0, 68.0])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def soccertrack_ids(root=None):
    root = root or SOCCERTRACK
    if os.path.isdir(os.path.join(root, "bas")):
        return sorted(os.listdir(os.path.join(root, "bas")))
    return sorted({os.path.basename(q).split("_")[1].split(".")[0] for q in glob.glob(os.path.join(CACHE, "soccertrack_*.npz"))})


def soccertrack_usable(m):
    """False for a match whose ball track carries no possession status (132831, 132877: always 4) or whose side
    map left a team without players."""
    g = m["half"] > 0
    return bool(np.mean(np.isin(m["labels"][g], (pm.H, pm.A))) >= 0.3
                and all(np.isfinite(m["px"][g][:, m["slot_team"] == t]).any() for t in (0, 1)))


def from_soccertrack(mid, root=None, merge_s=0.0):
    d = _soccertrack_raw(mid, root)
    labels = d["labels"].copy()
    if merge_s > 0:
        for h in (1, 2):
            idx = np.flatnonzero(d["half"] == h)
            labels[idx] = merge_short(d["labels"][idx], merge_s)
    gk, sign = keepers_and_sign(d["px"], d["half"], d["slot_team"])
    return {"name": "st" + mid, "px": d["px"], "py": d["py"], "slot_team": d["slot_team"], "gk": gk,
            "bx": d["bx"], "by": d["by"], "half": d["half"], "labels": labels, "sign": sign,
            "pitch": tuple(float(v) for v in d["pitch"])}
