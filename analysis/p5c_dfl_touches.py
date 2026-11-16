"""Open item 4: an honest touch test on real tracking -- DFL with event times synchronised by ELASTIC.

    uv run python analysis/p5c_dfl_touches.py      # -> eda_out/reports/P5c_dfl_touches.md

Events: data/external/pathcrf/data/sportec_synced/<match>.parquet (Kim et al.; PathCRF repository,
MPL-2.0): DFL events re-timed to the tracking with ELASTIC, one frame_id per event. Every on-ball
event type counts as a touch (fouls do not).
1. Do kinks in the ball path (mtah.timing.kinks) mark touches? Precision: kinks within TOL frames
   of a touch; recall: touches with a kink that close; chance level from the kinks shifted by 1 s.
   On clean DFL, degraded DFL, and degraded DFL smoothed like our ATD (25 frames, P4g).
2. How good could a touch-based team call be? Oracle: the possession team switches at every touch
   to the toucher's team (the event's player), against the control rule and the kink-touch labeler
   of P4f; dead ball from the labels, so only the team call is judged.
The Metrica numbers of P4d (precision 61%, recall 84%) came from a partly synthetic ball.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.timing import kinks  # noqa: E402
from p4f_touch_labeler import SEED, control_touch, touch_only  # noqa: E402

FPS = 25
SYNCED = os.path.join(ROOT, "data", "external", "pathcrf", "data", "sportec_synced")
TOLS = (3, 6)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5c_dfl_touches.md")


def touch_frames(mid, m):
    """(index into the Match arrays, team code) of every on-ball event."""
    ev = pd.read_parquet(os.path.join(SYNCED, mid + ".parquet"))
    ev = ev[(ev.spadl_type != "foul") & ev.frame_id.notna()]
    n1 = int((m["half"] == 1).sum())
    idx = np.where(ev.period_id == 1, ev.frame_id - 10000, n1 + 1 + ev.frame_id - 100000).astype(int)
    team = np.where(ev.player_id.str.startswith("home"), pm.H, pm.A)
    ok = (idx >= 0) & (idx < len(m["bx"]))
    idx, team = idx[ok], team[ok]
    ok = m["half"][idx] == np.where(ev.period_id.values[ok] == 1, 1, 2)
    return idx[ok], team[ok]


def near(x, ref, tol):
    ref = np.sort(ref)
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.minimum(np.abs(ref[j - 1] - x), np.abs(ref[j] - x)) <= tol


def smooth(m, w):
    out = dict(m)
    for k in ("bx", "by"):
        s = pd.Series(m[k].astype(float)).rolling(w, center=True, min_periods=1).mean().values
        out[k] = np.where(np.isnan(m[k]), np.nan, s).astype(np.float32)
    return out


def oracle(m, idx, team):
    """The team of the last touch, inside alive stretches; the restart's team until the first one."""
    alive = np.isin(m["labels"], (pm.H, pm.A))
    out = control(m, dead="truth")
    last = np.full(len(out), -1)
    last[idx] = team
    cur = -1
    for i in range(len(out)):
        if not alive[i]:
            cur = -1
            continue
        if last[i] >= 0:
            cur = last[i]
        if cur >= 0:
            out[i] = cur
    return out


def spell_f1(m, pred):
    return pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    mids = [i.replace("DFL-MAT-", "") for i in ids]
    clean = [from_dfl(i) for i in ids]
    deg = [degrade(m, rates_for(i), seed=SEED) for m, i in zip(clean, ids)]
    lines = ["# P5c: gerçek takip verisinde dokunuşlar (DFL + ELASTIC ile senkronlanmış olaylar)", "",
             "`analysis/p5c_dfl_touches.py`. 7 DFL maçı; dokunuş = faul dışındaki her topla ilgili olay.", "",
             "## 1. Kırılma dokunuşu gösteriyor mu?", "",
             "| top | dakikada kırılma | precision ±3 kare (şans) | precision ±6 kare (şans) | recall ±3 | recall ±6 |", "|---|---|---|---|---|---|"]
    for name, ms in (("temiz", clean), ("bozulmuş", deg), ("bozulmuş + 25 kare yumuşatma (ATD gibi)", [smooth(m, 25) for m in deg])):
        rows = []
        for m, mid in zip(ms, mids):
            alive = np.isin(m["labels"], (pm.H, pm.A))
            idx, _ = touch_frames(mid, m)
            idx = idx[alive[idx]]
            k = kinks(m["bx"], m["by"], m["pitch"])
            k = k[alive[k]]
            mins = (alive & ~np.isnan(m["bx"])).sum() / FPS / 60
            r = [len(k) / mins]
            for t in TOLS:
                r += [near(k, idx, t).mean(), near(k + FPS, idx, t).mean()]
            r += [near(idx, k, t).mean() for t in TOLS]
            rows.append(r)
        r = np.mean(rows, axis=0)
        lines.append("| {} | {:.1f} | {:.0%} ({:.0%}) | {:.0%} ({:.0%}) | {:.0%} | {:.0%} |".format(name, *r))
        print(lines[-1], flush=True)
    lines += ["", "## 2. Dokunuşları kusursuz bilseydik takım kararı (spellF1, ölü top etiketten)", "",
              "| girdi | kontrol kuralı | kırılma dokunuşu (P4f) | kontrol + kırılma | **kusursuz dokunuş** |", "|---|---|---|---|---|"]
    for name, ms in (("temiz", clean), ("bozulmuş", deg)):
        res = []
        for m, mid in zip(ms, mids):
            idx, team = touch_frames(mid, m)
            res.append([spell_f1(m, control(m, dead="truth")), spell_f1(m, touch_only(m)), spell_f1(m, control_touch(m)),
                        spell_f1(m, oracle(m, idx, team))])
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | **{:.3f}** |".format(name, *np.mean(res, axis=0)))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
