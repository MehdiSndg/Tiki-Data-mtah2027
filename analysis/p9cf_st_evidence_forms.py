"""P9cf: fusion forms for action-frame team evidence, on fold A of SoccerTrack v2 only (fold B is kept unseen for the
registered test). P9ce: adding +-W around every action loses 0.02-0.13 even with perfect evidence (a touch is not a
possession: 9 % of the actions' teams differ from the label, and each forces a short spell). Forms, with simulated
evidence (recall, team accuracy, +-3 frames jitter):
  after   +-W on f..f+HW only, possession classes only (no block / tackle / out / header)
  pair    two consecutive detected actions of one team at most GAP s apart: +w per frame over the interval between them
  cost    the change cost multiplied by C within +-HW of an action whose team differs from the previous action's
  pair+cost

    python analysis/p9cf_st_evidence_forms.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import SOCCERTRACK, from_soccertrack, soccertrack_side_codes  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

MIDS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("117092", "117093", "118575", "118576")
POSS = ("PASS", "DRIVE", "HIGH PASS", "CROSS", "SHOT", "THROW IN", "FREE KICK")
HW, FPS = 6, 25
SIMS = [(1.0, 1.0, 0), (0.9, 0.7, 3), (0.8, 0.6, 3)]
FORMS = [("after", 2.0), ("after", 4.0), ("pair", 0.25), ("pair", 0.5), ("pair", 1.0), ("pair", 2.0), ("pair", 4.0), ("pair", 8.0),
         ("cost", 0.5), ("cost", 0.25), ("pair+cost", 0.5), ("pair+cost", 1.0)]
GAP = 4.0


def actions(mid, half):
    acts = json.load(open(os.path.join(SOCCERTRACK, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    n1 = int(np.sum(half == 1))
    out = []
    for h, hname in ((1, "1st"), (2, "2nd")):
        st = np.load(os.path.join(SOCCERTRACK, "ball", "{}_{}_ball.npz".format(mid, hname)))["status"]
        ev = [a for a in acts if a["gameTime"].strip().startswith("{} ".format(h))]
        codes = soccertrack_side_codes(ev, st)
        for a in ev:
            if codes.get(a.get("team")) is None or a["label"] not in POSS:
                continue
            k = int(a["frame"]) + (0 if h == 1 else n1 + 1)
            if k < len(half) and half[k] == h:
                out.append((k, 1 if codes[a["team"]] == 1 else -1))
    return sorted(out)


def fuse(form, par, ev, lo, sw):
    e, s = np.zeros(len(lo)), np.array(np.broadcast_to(sw, lo.shape), np.float64)
    if form == "after":
        for k, t in ev:
            e[k:k + HW + 1] += t * par
        e = np.clip(e, -2 * par, 2 * par)
    if "pair" in form:
        for (k0, t0), (k1, t1) in zip(ev[:-1], ev[1:]):
            if t0 == t1 and k1 - k0 <= GAP * FPS:
                e[k0:k1 + 1] = t0 * par
    if "cost" in form:
        c = 0.5 if form == "pair+cost" else par
        for (k0, t0), (k1, t1) in zip(ev[:-1], ev[1:]):
            if t0 != t1:
                s[max(k1 - HW, 0):k1 + HW + 1] *= c
    return 1 / (1 + np.exp(-(lo + e))), s


def score(lab, l3, p, alive, sw, g):
    c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
    return np.array([pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]])


def main():
    rows = {}
    for mid in MIDS:
        z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
        p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
        acts = actions(mid, from_soccertrack(mid)["half"])
        alive = g & np.isin(lab, (pm.H, pm.A))
        l3 = merged(lab, 3.0)
        base = score(lab, l3, p, alive, sw, g)
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        rng = np.random.default_rng(0)
        u, j = rng.random((len(acts), 2)), rng.integers(-3, 4, len(acts))
        for acc, rec, jit in SIMS:
            ev = sorted((k + (j[i] if jit else 0), t if u[i, 1] <= acc else -t) for i, (k, t) in enumerate(acts) if u[i, 0] <= rec)
            for form, par in FORMS:
                q, s = fuse(form, par, ev, lo, sw)
                rows.setdefault((acc, rec, form, par), []).append(score(lab, l3, q, alive, s, g) - base)
        print(mid, "done", flush=True)
    print("acc recall form par | mean gain 1.6 s, 3 s | ahead | worst")
    for k, v in rows.items():
        v = np.array(v)
        print("{:.2f} {:.1f} {} {} | {:+.4f} {:+.4f} | {} / {} | {:+.4f}".format(*k, v[:, 0].mean(), v[:, 1].mean(), int((v[:, 0] > 0).sum()), len(v), v[:, 0].min()))


if __name__ == "__main__":
    main()
