"""The PFF FC (Gradient Sports) 2022 World Cup open data as match dicts (P9da's compact npz per match). Kept out of
mtah/external.py on purpose: that file is part of the model cache key (mtah.grumodel.HASH_FILES)."""
import glob
import os

import numpy as np

import possession_metrics as pm
from mtah.external import CACHE, FPS, merge_short
from mtah.match import keepers_and_sign

PFF = os.environ.get("MTAH_PFF", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "pff_npz"))
PFF_RESTART = ("T", "F", "G", "C", "K", "P", "D")          # setpieceType of a restart (everything but open play "O")


def pff_ids(root=None):
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root or PFF, "*.npz")) if ".tmp" not in p)


CONTROLLED = ("PA", "IT", "BC", "CR", "SH")      # pass, initial touch (reception), ball carry, cross, shot


def pff_labels(ev, frame, period, controlled=False):
    """Per tracking frame H / A / D / X from PFF's game events (P9da's slim rows). An on-the-ball event gives its
    frames to its team; the gap up to the next one stays with that team (the ball travelling) unless the ball went
    out (an OUT event in between) or the next event is a restart (set piece, kick-off): then the gap is dead, from the
    OUT event's frame (or the end of the last touch) to the restart. X outside the periods."""
    lab = np.full(len(frame), pm.X, np.int8)
    idx = lambda f: int(np.searchsorted(frame, f))
    rows = sorted((e for e in ev if e.get("start_frame") is not None and e["gameEventType"] in ("OTB", "OUT", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "END")),
                  key=lambda e: (e["start_frame"], e["end_frame"] or e["start_frame"]))
    last_end, last_team, out_at = None, None, None
    for e in rows:
        a, b = idx(e["start_frame"]), idx(e["end_frame"] if e["end_frame"] is not None else e["start_frame"]) + 1
        kind = e["gameEventType"]
        if kind == "OUT":
            out_at = a if out_at is None else out_at
            continue
        if kind == "END":
            if last_end is not None and last_team is not None:
                lab[last_end:a] = last_team if out_at is None else pm.D
                if out_at is not None:
                    lab[last_end:out_at] = last_team
            last_end, last_team, out_at = None, None, None
            continue
        if e.get("home_ball") is None:
            continue
        team = pm.H if e["home_ball"] else pm.A
        restart = kind != "OTB" or e.get("setpieceType") in PFF_RESTART
        if controlled and not restart and last_team is not None and team != last_team and e.get("possessionEventType") not in CONTROLLED:
            team = last_team
        if last_end is not None and a > last_end:
            if out_at is not None or restart:
                cut = min(max(out_at if out_at is not None else last_end, last_end), a)
                lab[last_end:cut] = last_team
                lab[cut:a] = pm.D
            else:
                lab[last_end:a] = last_team
        lab[a:max(b, a + 1)] = team
        last_end, last_team, out_at = max(b, a + 1), team, None
    lab[period == 0] = pm.X
    return lab


def _cache_key(root):
    """What a cached match depends on beyond the export file: the labelling code and sets, and which export it is."""
    import hashlib
    import inspect
    src = inspect.getsource(pff_labels) + repr((CONTROLLED, PFF_RESTART, FPS)) + os.path.abspath(root or PFF)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _pff_raw(gid, root=None):
    path = os.path.join(CACHE, "pff_{}.npz".format(gid))
    key = _cache_key(root)
    if os.path.exists(path):
        with np.load(path) as d:
            if "key" in d.files and str(d["key"]) == key:
                return {k: d[k] for k in d.files if k != "key"}
    import json
    with np.load(os.path.join(root or PFF, gid + ".npz")) as z:
        z = {k: z[k] for k in z.files}
    lab = pff_labels(json.loads(str(z["events"])), z["frame"], z["period"])
    lab_c = pff_labels(json.loads(str(z["events"])), z["frame"], z["period"], controlled=True)
    cols = [(s, j) for s in (0, 1) for j in range(z["xy"].shape[2]) if (z["vis"][:, s, j] > 0).any()]
    parts = []
    for h in (1, 2):
        k = np.flatnonzero(z["period"] == h)
        if not len(k):
            continue
        t = z["t_ms"][k]
        grid = np.arange(t[0], t[-1], 1000.0 / FPS)
        j = k[np.clip(np.searchsorted(t, grid), 0, len(k) - 1)]         # the tracking frame at or after each 25 fps tick
        xy = z["xy"][j].astype(np.float32)
        xy[z["xy"][j] == -32768] = np.nan
        px = np.stack([xy[:, s, c, 0] for s, c in cols], 1) / 100 / 105 + 0.5
        py = np.stack([xy[:, s, c, 1] for s, c in cols], 1) / 100 / 68 + 0.5
        ball = z["ball"][j].astype(np.float32)
        ball[z["ball"][j] == -32768] = np.nan
        parts.append({"px": px, "py": py, "bx": ball[:, 0] / 100 / 105 + 0.5, "by": ball[:, 1] / 100 / 68 + 0.5, "half": np.full(len(j), h, np.int8),
                      "labels": lab[j], "labels_c": lab_c[j], "ball_vis": z["ball_vis"][j], "vis": np.stack([z["vis"][j][:, s, c] for s, c in cols], 1)})
    n = len(cols)
    sep = {"px": np.full((1, n), np.nan, np.float32), "py": np.full((1, n), np.nan, np.float32), "bx": np.array([np.nan], np.float32),
           "by": np.array([np.nan], np.float32), "half": np.array([0], np.int8), "labels": np.array([pm.X], np.int8), "labels_c": np.array([pm.X], np.int8),
           "ball_vis": np.array([0], np.uint8), "vis": np.zeros((1, n), np.uint8)}
    out = {k: np.concatenate([parts[0][k], sep[k], parts[1][k]]) for k in parts[0]}
    out["slot_team"] = np.array([s for s, _ in cols], np.int8)
    out["pitch"] = np.array([105.0, 68.0])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, key=np.array(key), **out)
    return out


def from_pff(gid, root=None, merge_s=0.0, ball="all", controlled=False):
    """A PFF FC 2022 World Cup match (broadcast tracking, 29.97 fps -> 25 fps) as a match dict. ball: "all" keeps the
    ball wherever PFF gives one (seen or estimated), "visible" only where it was seen. controlled: the labels of
    pff_labels(controlled=True) (a change needs a controlled action)."""
    d = _pff_raw(gid, root)
    labels = (d["labels_c"] if controlled else d["labels"]).copy()
    if merge_s > 0:
        for h in (1, 2):
            idx = np.flatnonzero(d["half"] == h)
            labels[idx] = merge_short((d["labels_c"] if controlled else d["labels"])[idx], merge_s)
    bx, by = d["bx"].copy(), d["by"].copy()
    if ball == "visible":
        bx[d["ball_vis"] < 2] = np.nan
        by[d["ball_vis"] < 2] = np.nan
    gk, sign = keepers_and_sign(d["px"], d["half"], d["slot_team"])
    return {"name": "pff" + gid, "px": d["px"], "py": d["py"], "slot_team": d["slot_team"], "gk": gk, "bx": bx, "by": by, "half": d["half"],
            "labels": labels, "sign": sign, "pitch": tuple(float(v) for v in d["pitch"]), "ball_vis": d["ball_vis"], "vis": d["vis"]}
