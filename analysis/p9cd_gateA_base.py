"""P9cd: the base of the video evidence's gate A (notes/BULGULAR.md, 9 Nov): at every SoccerTrack v2 ball action with an
actor, the team of the player nearest the ball, with the ball degraded at our match's rates (rates_for, smoothed over 25
frames, dead_carry_p 0.25, seed 0, as P9bw; a missing ball is held at its last position) and, for information, with the
undegraded (event-anchored) ball. The 8 matches with a possession status (side -> home / away from it, as
mtah.external). Prints the accuracy per match and per fold.

    python analysis/p9cd_gateA_base.py <soccertrack root>
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_soccertrack, soccertrack_side_codes, soccertrack_usable  # noqa: E402

FOLDS = {"A": ("117092", "117093", "118575", "118576", "132831"), "B": ("118577", "118578", "128057", "128058", "132877")}


def held(x):
    ok = ~np.isnan(x)
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(x)), -1))
    out = np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)
    return out


def nearest_team(m, k, bx, by):
    L, W = m["pitch"]
    d = np.hypot((m["px"][k] - bx[k]) * L, (m["py"][k] - by[k]) * W)
    if np.all(np.isnan(d)):
        return -1
    return int(m["slot_team"][np.nanargmin(d)])


def main(root):
    res = {}
    for fold, mids in FOLDS.items():
        for mid in mids:
            m0 = from_soccertrack(mid)
            if not soccertrack_usable(m0):
                continue
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            acts = json.load(open(os.path.join(root, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
            n1 = int(np.sum(m0["half"] == 1))
            bxd, byd = held(m["bx"]), held(m["by"])
            ok_d = ok_t = tot = 0
            for h, hname in ((1, "1st"), (2, "2nd")):
                st = np.load(os.path.join(root, "ball", "{}_{}_ball.npz".format(mid, hname)))["status"]
                ev = [a for a in acts if a["gameTime"].strip().startswith("{} ".format(h))]
                codes = soccertrack_side_codes(ev, st)
                for a in ev:
                    if a.get("player_id") in (None, "") or codes.get(a["team"]) is None:
                        continue
                    k = int(a["frame"]) + (0 if h == 1 else n1 + 1)
                    if k >= len(m["bx"]) or m["half"][k] != h:
                        continue
                    team = 0 if codes[a["team"]] == 1 else 1
                    tot += 1
                    ok_d += nearest_team(m, k, bxd, byd) == team
                    ok_t += nearest_team(m0, k, m0["bx"], m0["by"]) == team
            res[mid] = (fold, tot, ok_d / max(tot, 1), ok_t / max(tot, 1))
            print(fold, mid, "actions", tot, "nearest to degraded ball {:.3f}".format(ok_d / max(tot, 1)),
                  "to undegraded ball {:.3f}".format(ok_t / max(tot, 1)), flush=True)
    for fold in FOLDS:
        r = [v for v in res.values() if v[0] == fold]
        n = sum(v[1] for v in r)
        print("fold", fold, "actions", n, "degraded {:.3f}".format(sum(v[1] * v[2] for v in r) / max(n, 1)),
              "undegraded {:.3f}".format(sum(v[1] * v[3] for v in r) / max(n, 1)))


if __name__ == "__main__":
    main(sys.argv[1])
