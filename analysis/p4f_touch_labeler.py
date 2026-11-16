"""P4: would touches (kinks in the ball path) catch the one-touch possessions the control rule misses?

    uv run python analysis/p4f_touch_labeler.py     # -> eda_out/reports/P4f_touch_labeler.md

An understanding check, not a model: dead ball from the labels (team call only), external data.
  control         the control rule (R 1.5 m, T 0.4 s)
  touch           possession = team of the nearest player (<= TOUCH_R) at the last kink
  control+touch   the control rule, and a kink with the other team's player nearest within TOUCH_R
                  hands over at once
Caveat: the Metrica sample ball sits exactly on the player at events (P4e: median distance 0.0 m), so
its kinks may be partly built from the events; DFL is the honest test.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import _team_distances, control  # noqa: E402
from mtah.timing import kinks  # noqa: E402

FPS = 25
SEED = 0
TOUCH_R = 1.0
OUT = os.path.join(ROOT, "eda_out", "reports", "P4f_touch_labeler.md")


def touches(m, alive):
    dH, dA = _team_distances(m)
    k = kinks(m["bx"], m["by"], m["pitch"])
    k = k[alive[k]]
    team = np.where(dH[k] <= dA[k], pm.H, pm.A)
    ok = np.minimum(dH[k], dA[k]) <= TOUCH_R
    return k[ok], team[ok]


def touch_only(m, alive=None):
    """alive: frames in play; default the labels' H/A (team call only)."""
    alive = np.isin(m["labels"], (pm.H, pm.A)) if alive is None else alive
    k, team = touches(m, alive)
    base = control(m, dead=(m["half"] > 0) & ~alive)                      # the restart team and the frames before any touch
    out = base.copy()
    cur = np.full(len(out), -1)
    cur[k] = team
    for h in (1, 2):
        idx = np.flatnonzero((m["half"] == h) & alive)
        if not len(idx):
            continue
        last = -1
        for i in idx:
            if not alive[i - 1] if i > 0 else True:
                last = -1                                # a restart: forget the touch team
            if cur[i] >= 0:
                last = cur[i]
            if last >= 0:
                out[i] = last
    return out


def control_touch(m, R=1.5, T=0.4, alive=None):
    alive = np.isin(m["labels"], (pm.H, pm.A)) if alive is None else alive
    k, team = touches(m, alive)
    kick = np.full(len(alive), -1)
    kick[k] = team
    dH, dA = _team_distances(m)
    need = max(1, int(round(T * FPS)))
    out = np.full(len(dH), pm.D, np.int8)
    cur, runH, runA = None, 0, 0
    for i in range(len(dH)):
        if not alive[i]:
            cur, runH, runA = None, 0, 0
            continue
        runH = runH + 1 if dH[i] <= R and dH[i] <= dA[i] else 0
        runA = runA + 1 if dA[i] <= R and dA[i] < dH[i] else 0
        if kick[i] >= 0:
            cur = int(kick[i])
        elif runH >= need:
            cur = pm.H
        elif runA >= need:
            cur = pm.A
        if cur is None:
            cur = pm.H if dH[i] <= dA[i] else pm.A
        out[i] = cur
    return out


def spell_f1(m, pred):
    return pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [from_metrica_sample(g) for g in (1, 2)]
    dfl = [from_dfl(i) for i in ids]
    src = {("Metrica", "temiz"): met, ("DFL", "temiz"): dfl,
           ("Metrica", "bozulmuş"): [degrade(m, rates_for("metrica"), seed=SEED) for m in met],
           ("DFL", "bozulmuş"): [degrade(m, rates_for(i), seed=SEED) for m, i in zip(dfl, ids)]}
    lines = ["# P4f: dokunuşlar (top yolundaki kırılmalar) tek dokunuşluk possession'ları yakalar mı?", "",
             "`analysis/p4f_touch_labeler.py`. Ölü top etiketten (sadece takım kararı), dış veri. spellF1, maç ortalaması;",
             "parantezde tahmin / gerçek possession oranı. Model değil, anlama denemesi.", "",
             "| kaynak | girdi | dakikada dokunuş (kırılma + oyuncu ≤ 1 m) | control | touch | control+touch |", "|---|---|---|---|---|---|"]
    for (name, kind), ms in src.items():
        res = {"control": [], "touch": [], "control+touch": []}
        rate = []
        for m in ms:
            alive = np.isin(m["labels"], (pm.H, pm.A))
            rate.append(len(touches(m, alive)[0]) / (alive.sum() / FPS / 60))
            n_gt = len(pm.possession_spells(m["labels"], pm.valid_runs(m["labels"])))
            for key, fn in (("control", lambda x: control(x, dead="truth")), ("touch", touch_only), ("control+touch", control_touch)):
                p = fn(m)
                res[key].append((spell_f1(m, p), len(pm.possession_spells(p, pm.valid_runs(m["labels"]))) / n_gt))
        cells = ["{:.3f} ({:.2f})".format(*np.mean(res[k], axis=0)) for k in ("control", "touch", "control+touch")]
        lines.append("| {} | {} | {:.1f} | {} |".format(name, kind, np.mean(rate), " | ".join(cells)))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
