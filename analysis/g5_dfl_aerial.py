"""Does a ball in the air change who controls it? Ball height from DFL (external), GT-free for us.

    uv run python analysis/g5_dfl_aerial.py

floodlight drops the ball's Z, so the DFL position XML is parsed directly. Check of the parser: with
no height gate the control rule must reproduce analysis/g4_dfl_rules.py on the same 3 matches
(control R=1.5 T=0.4 vs DFL merged < 1.6 s: spellF1 0.608).

Gate: a player can only count towards taking control on frames where the ball is at most H m high
(a ball in the air is "in contest", not controlled: literature/README.md). The current team keeps it.
"""

import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402
from g3_dfl_conversion import merge_short  # noqa: E402
from g4_dfl_rules import DFL, MATCHES, rule_control, score  # noqa: E402

# Params
R, T = 1.5, 0.4
HEIGHTS = (np.inf, 3.0, 2.0, 1.5, 1.0)


def home_team(info):
    for t in ET.parse(info).getroot().iter("Team"):
        if t.get("Role") == "home":
            return t.get("TeamId")


def parse(pos):
    """{section: {"ball": (N, X, Y, Z, status, poss), "players": [(team, N, X, Y), ...]}}"""
    out, cur, rows = {}, None, []
    for ev, el in ET.iterparse(pos, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            cur, rows = (el.get("GameSection"), el.get("TeamId")), []
        elif ev == "end" and el.tag == "Frame":
            a = el.attrib
            if cur[1] == "BALL":
                rows.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a["Z"]),
                             int(a["BallStatus"]), int(a["BallPossession"])))
            else:
                rows.append((int(a["N"]), float(a["X"]), float(a["Y"])))
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            sec = out.setdefault(cur[0], {"players": []})
            arr = np.array(rows)
            if cur[1] == "BALL":
                sec["ball"] = arr
            else:
                sec["players"].append((cur[1], arr))
            el.clear()
    return out


def halves(mid):
    pos = glob.glob(os.path.join(DFL, "*positions*{}.xml".format(mid)))[0]
    info = glob.glob(os.path.join(DFL, "*matchinformation*{}.xml".format(mid)))[0]
    home = home_team(info)
    data = parse(pos)
    res = []
    for sec in ("firstHalf", "secondHalf"):
        d = data[sec]
        b = d["ball"]
        N0, n = int(b[0, 0]), len(b)
        assert np.all(np.diff(b[:, 0]) == 1)
        bx, by, bz = b[:, 1], b[:, 2], b[:, 3]
        dH, dA = np.full(n, np.inf), np.full(n, np.inf)
        for team, p in d["players"]:
            i = p[:, 0].astype(int) - N0
            ok = (i >= 0) & (i < n)
            dist = np.hypot(p[ok, 1] - bx[i[ok]], p[ok, 2] - by[i[ok]])
            tgt = dH if team == home else dA
            np.minimum.at(tgt, i[ok], dist)
        status, poss = b[:, 4], b[:, 5]
        truth = np.where(status == 0, pm.D, np.where(poss == 1, pm.H, pm.A)).astype(np.int8)
        res.append((dH, dA, status == 1, bz, truth))
    return res


def main():
    mids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))[:MATCHES]
    hs = [h for mid in mids for h in halves(mid)]
    z = np.concatenate([h[3][h[2]] for h in hs])
    print("ball height on alive frames: > 1 m {:.1%}, > 2 m {:.1%}, > 3 m {:.1%}".format(np.mean(z > 1), np.mean(z > 2), np.mean(z > 3)))
    print("control R={} T={} vs DFL merged < 1.6 s, {} halves of {} matches (parser check: H=inf must give 0.608)".format(R, T, len(hs), MATCHES))
    print("   {:<10} {:>8} {:>8} {:>12}".format("gate H", "spellF1", "macroF1", "spells p/gt"))
    for H in HEIGHTS:
        res = np.array([score(merge_short(t, 1.6), rule_control(np.where(bz > H, np.inf, dH), np.where(bz > H, np.inf, dA), al, R, T))
                        for dH, dA, al, bz, t in hs])
        print("   {:<10} {:>8.3f} {:>8.3f} {:>6.0f}/{:<5.0f}".format("none" if np.isinf(H) else "{} m".format(H), res[:, 0].mean(), res[:, 1].mean(), res[:, 2].sum(), res[:, 3].sum()))


if __name__ == "__main__":
    main()
