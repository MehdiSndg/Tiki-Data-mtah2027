"""Diagnosis on clean tracking: how far do simple ball-based possession rules get? (DFL, external)

    uv run python analysis/g4_dfl_rules.py

Alive frames only (DFL BallStatus = 1), so this measures the team call, not dead-ball detection; the
dead frames are given to every rule as D. Targets: DFL's own BallPossession, and the version merged
towards Metrica's definition (analysis/g3_dfl_conversion.py, merge < 1.6 s). Scored with the
official possession_metrics (spellF1, macroF1). Rules:

  nearest      team of the player nearest the ball, every frame
  sticky R     a team takes possession when one of its players is within R m of the ball, else
               the last team keeps it (CHALLENGE.md: a contested or loose ball stays with the last team)
  control R,T  as sticky, but the new team needs its player within R m for T consecutive seconds
               (the "control, not touch" hypothesis, literature/README.md)

Uses 3 of the 7 matches to keep the run short.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402
from g3_dfl_conversion import merge_short  # noqa: E402

# Params
DFL = os.path.join(ROOT, "data", "external", "dfl")
MATCHES = 3
FPS = 25


def load(mid):
    from floodlight.io.dfl import read_position_data_xml
    pos = glob.glob(os.path.join(DFL, "*positions*{}.xml".format(mid)))[0]
    info = glob.glob(os.path.join(DFL, "*matchinformation*{}.xml".format(mid)))[0]
    xy, poss, status, _, _ = read_position_data_xml(pos, info)
    out = []
    for h in xy:
        H, A, B = xy[h]["Home"].xy, xy[h]["Away"].xy, xy[h]["Ball"].xy
        dH = np.nanmin(np.hypot(H[:, 0::2] - B[:, :1], H[:, 1::2] - B[:, 1:2]), axis=1)
        dA = np.nanmin(np.hypot(A[:, 0::2] - B[:, :1], A[:, 1::2] - B[:, 1:2]), axis=1)
        s, p = status[h].code, poss[h].code
        truth = np.where(s == 0, pm.D, np.where(p == 1, pm.H, pm.A)).astype(np.int8)
        out.append((dH, dA, s == 1, truth))
    return out


def rule_nearest(dH, dA, alive):
    return np.where(~alive, pm.D, np.where(dH <= dA, pm.H, pm.A)).astype(np.int8)


def rule_control(dH, dA, alive, R, T):
    need = max(1, int(round(T * FPS)))
    out = np.full(len(dH), pm.D, np.int8)
    cur, runH, runA = None, 0, 0
    for i in range(len(dH)):
        if not alive[i]:
            cur, runH, runA = None, 0, 0
            continue
        runH = runH + 1 if dH[i] <= R and dH[i] <= dA[i] else 0
        runA = runA + 1 if dA[i] <= R and dA[i] < dH[i] else 0
        if runH >= need:
            cur = pm.H
        elif runA >= need:
            cur = pm.A
        if cur is None:                      # restart: first nearest team
            cur = pm.H if dH[i] <= dA[i] else pm.A
        out[i] = cur
    return out


def score(truth, pred):
    r = pm.evaluate(truth, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    mids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))[:MATCHES]
    halves = [h for mid in mids for h in load(mid)]
    rules = [("nearest", lambda dH, dA, al: rule_nearest(dH, dA, al))]
    for R in (1.0, 1.5, 2.0):
        rules.append(("sticky R={}".format(R), lambda dH, dA, al, R=R: rule_control(dH, dA, al, R, 0.0)))
    for T in (0.2, 0.4, 0.8):
        rules.append(("control R=1.5 T={}".format(T), lambda dH, dA, al, T=T: rule_control(dH, dA, al, 1.5, T)))
    for target in ("DFL raw", "DFL merged < 1.6 s"):
        print("\ntarget: {} ({} halves of {} matches; dead frames given as D to every rule)".format(target, len(halves), MATCHES))
        print("   {:<22} {:>8} {:>8} {:>12}".format("rule", "spellF1", "macroF1", "spells p/gt"))
        for name, fn in rules:
            res = []
            for dH, dA, alive, truth in halves:
                t = truth if target == "DFL raw" else merge_short(truth, 1.6)
                res.append(score(t, fn(dH, dA, alive)))
            res = np.array(res)
            print("   {:<22} {:>8.3f} {:>8.3f} {:>6.0f}/{:<5.0f}".format(name, res[:, 0].mean(), res[:, 1].mean(),
                                                                       res[:, 2].sum(), res[:, 3].sum()))


if __name__ == "__main__":
    main()
