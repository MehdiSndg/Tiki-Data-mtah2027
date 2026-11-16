"""P9da: the PFF FC (Gradient Sports) 2022 World Cup open data -> one compact npz per match (Colab; external data).
Tracking <id>.jsonl.bz2 (29.97 fps broadcast tracking, raw and smoothed) and Event Data/<id>.json.

    python p9da_pff_convert_colab.py <pff dir> <out dir> [n processes]
    # -> <out>/<id>.npz: frame (frameNum), t_ms (videoTimeMs), period (0 outside),
    #    jersey (2, K) the jersey numbers of the K slots per team (home, away; -1 unused),
    #    xy (n, 2, K, 2) int16 cm raw players (-32768 missing), vis (n, 2, K) uint8: 0 missing, 1 estimated, 2 visible,
    #    xys the same smoothed, ball (n, 3) int16 cm raw (first ball; -32768 missing), ball_vis (n,) uint8,
    #    balls smoothed ball, balls_vis; ev_frame / ev_end_frame (tracking's start_frame / end_frame per game event id),
    #    events: JSON string, the slim rows of Event Data
"""
import bz2
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

K, MISS = 20, -32768
VIS = {"ESTIMATED": 1, "VISIBLE": 2}
GE = ("gameEventType", "period", "teamId", "homeTeam", "playerId", "setpieceType", "endType", "outType", "subType", "startGameClock", "videoMissing", "touches")
PE = ("possessionEventType", "passOutcomeType", "passType", "ballHeightType", "receiverPlayerId", "passerPlayerId", "carrierPlayerId", "shotOutcomeType",
      "clearanceOutcomeType", "challengeOutcomeType", "challengeType", "touchOutcomeType", "touchType", "reboundOutcomeType", "ballCarrierPlayerId", "challengerPlayerId")


def cm(v):
    return MISS if v is None else int(np.clip(round(float(v) * 100), -32767, 32767))


def convert(args):
    src, out, gid = args
    dst = os.path.join(out, gid + ".npz")
    if os.path.exists(dst):
        return gid, "exists"
    slots = [{}, {}]
    F, T, P, XY, XS, VI, B, BV, BS, BSV, EF = [], [], [], [], [], [], [], [], [], [], {}
    with bz2.open(os.path.join(src, "Tracking Data", gid + ".jsonl.bz2"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            xy, xs, vi = np.full((2, K, 2), MISS, np.int16), np.full((2, K, 2), MISS, np.int16), np.zeros((2, K), np.uint8)
            for s, (ka, kb) in enumerate((("homePlayers", "homePlayersSmoothed"), ("awayPlayers", "awayPlayersSmoothed"))):
                for key, arr in ((ka, xy), (kb, xs)):
                    for p in r.get(key) or []:
                        j = slots[s].setdefault(str(p.get("jerseyNum")), len(slots[s]))
                        if j >= K or p.get("x") is None:
                            continue
                        arr[s, j] = cm(p["x"]), cm(p["y"])
                        if key == ka:
                            vi[s, j] = VIS.get(p.get("visibility"), 1)
            b = (r.get("balls") or [None])[0]
            bs = r.get("ballsSmoothed")
            bs = bs[0] if isinstance(bs, list) and bs else bs
            F.append(r.get("frameNum") or 0); T.append(r.get("videoTimeMs") or 0.0); P.append(r.get("period") or 0)
            XY.append(xy); XS.append(xs); VI.append(vi)
            B.append([cm(b.get(k)) for k in "xyz"] if b and b.get("x") is not None else [MISS] * 3); BV.append(VIS.get((b or {}).get("visibility"), 0) if b else 0)
            B_s = [cm(bs.get(k)) for k in "xyz"] if isinstance(bs, dict) and bs.get("x") is not None else [MISS] * 3
            BS.append(B_s); BSV.append(VIS.get(bs.get("visibility"), 1) if isinstance(bs, dict) and bs.get("x") is not None else 0)
            g = r.get("game_event")
            if g and r.get("game_event_id") is not None:
                EF[int(r["game_event_id"])] = (g.get("start_frame"), g.get("end_frame"), g.get("home_ball"))
    ev = []
    for e in json.load(open(os.path.join(src, "Event Data", gid + ".json"))):
        g, p = e.get("gameEvents") or {}, e.get("possessionEvents") or {}
        row = {"id": e.get("gameEventId"), "pid": e.get("possessionEventId"), "start": e.get("startTime"), "end": e.get("endTime"), "time": e.get("eventTime")}
        row.update({k: g.get(k) for k in GE})
        row.update({k: p.get(k) for k in PE if p.get(k) is not None})
        f = EF.get(int(e["gameEventId"])) if e.get("gameEventId") is not None else None
        row["start_frame"], row["end_frame"], row["home_ball"] = f if f else (None, None, None)
        ev.append(row)
    jersey = np.full((2, K), -1, np.int16)
    for s in (0, 1):
        for name, j in slots[s].items():
            if j < K:
                jersey[s, j] = int(name) if name.isdigit() else -1
    np.savez_compressed(dst + ".tmp.npz", frame=np.array(F, np.int32), t_ms=np.array(T, np.float64), period=np.array(P, np.int8), jersey=jersey,
                        xy=np.array(XY), xys=np.array(XS), vis=np.array(VI), ball=np.array(B, np.int16), ball_vis=np.array(BV, np.uint8),
                        balls=np.array(BS, np.int16), balls_vis=np.array(BSV, np.uint8), events=json.dumps(ev), overflow=max(len(slots[0]), len(slots[1])))
    os.replace(dst + ".tmp.npz", dst)
    return gid, "frames {} events {} slots {} {}".format(len(F), len(ev), len(slots[0]), len(slots[1]))


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    ids = sorted(os.path.basename(p)[:-10] for p in glob.glob(os.path.join(src, "Tracking Data", "*.jsonl.bz2")))
    ids = [g for g in ids if os.path.exists(os.path.join(src, "Event Data", g + ".json"))]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for gid, msg in pool.imap_unordered(convert, [(src, out, g) for g in ids]):
            print(gid, msg, flush=True)
    print("CONVERT_DONE", len(ids), flush=True)
