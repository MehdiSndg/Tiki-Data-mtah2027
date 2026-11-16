"""P4 Task 5: does a still ball mark the dead ball, and do kinks in the ball path mark touches?

    uv run python analysis/p4d_still_and_kinks.py [det_dir det_cls]   # -> eda_out/reports/P4d_still_and_kinks.md

Still ball (mtah.timing.still_runs, < 0.5 m/s for >= 1 s):
  external    Metrica sample and DFL, clean: the share of still frames that are dead, and the share of
              dead blocks with a still run inside
  our match   the Match v2 ball (ATD where it sees the ball, else the detector, no broadcast): the share
              of still frames inside the classifier's dead ball (v1 chain), and how many Smart
              Tagging set pieces, by type, have a still run from 5 s before to 1 s after their start
              -- the goal kick is the classifier's blind spot (44% before)
Kinks (mtah.timing.kinks, a turn >= 30 deg or the speed doubled / halved, at >= 2 m/s):
  Metrica     precision: kinks within 3 frames of an event start; recall: passes, shots,
              recoveries with a kink that close; chance level from the same kinks shifted by 1 s
  our match   kinks per in-play minute on the ATD ball and on the detector-only frames, next to Metrica
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
from mtah.context import SETPIECE_TYPES  # noqa: E402
from mtah.external import DFL, METRICA, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.timing import kinks, still_runs  # noqa: E402
from mtah.video_match import YOLO_DIR, from_tactical_v2  # noqa: E402
from p4_common import ours  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4d_still_and_kinks.md")
NEAR = 3                              # frames, kink <-> event
BEFORE, AFTER = 5 * FPS, 1 * FPS      # set piece window for a still run
PUBLIC = (11185, 18685)


def runs_of(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def still_mask(m):
    s = np.zeros(len(m["bx"]), bool)
    for a, b in still_runs(m["bx"], m["by"], m["pitch"]):
        s[a:b] = True
    return s & (m["half"] > 0)


def external_still(ms):
    st = dead_st = blocks = blocks_st = alive = 0
    for m in ms:
        s, dead = still_mask(m), m["labels"] == pm.D
        st += s.sum()
        dead_st += (s & dead).sum()
        alive += ((m["half"] > 0) & ~dead).sum()
        for a, b in runs_of(dead & (m["half"] > 0)):
            blocks += 1
            blocks_st += s[a:b].any()
    return dead_st / max(1, st), blocks_st / max(1, blocks), (st - dead_st) / max(1, alive)


def near_any(x, ref, tol):
    ref = np.sort(ref)
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.minimum(np.abs(ref[j - 1] - x), np.abs(ref[j] - x)) <= tol


def main():
    det_dir = sys.argv[1] if len(sys.argv) > 1 else YOLO_DIR
    det_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [from_metrica_sample(g) for g in (1, 2)]
    lines = ["# P4d: hareketsiz top ve vuruş izleri", "",
             "`analysis/p4d_still_and_kinks.py`; bizim maçta top: Match v2 (`{}`, sınıf {}).".format(os.path.relpath(det_dir, ROOT), det_cls), "",
             "## 1. Hareketsiz top (< 0.5 m/s, ≥ 1 s) ve ölü top", "",
             "| kaynak | hareketsiz karelerin ölü top payı | içinde hareketsiz top olan ölü blok | oyun içi karelerin hareketsiz payı |", "|---|---|---|---|"]
    for name, ms in (("Metrica temiz (2 maç)", met), ("DFL temiz (7 maç)", [from_dfl(i) for i in ids])):
        lines.append("| {} | {:.0%} | {:.0%} | {:.1%} |".format(name, *external_still(ms)))
        print(lines[-1], flush=True)
    dead, codes = ours()
    v2 = from_tactical_v2(det_dir, det_cls)
    s = still_mask(v2)
    g = v2["half"] > 0
    lines.append("| bizim maç, v2 topu; ölü = sınıflandırıcı (GT'siz) | {:.0%} | {:.0%} | {:.1%} |".format(
        (s & dead).sum() / max(1, s.sum()), np.mean([s[a:b].any() for a, b in runs_of(dead)]), (s & g & ~dead).sum() / max(1, (g & ~dead).sum())))
    df = load_frames()
    kind = df["setpiece_type"].values
    sp = [(a, b, SETPIECE_TYPES[kind[a]]) for a, b in runs_of(df["setpiece"].values) if not (PUBLIC[0] <= a < PUBLIC[1])]
    lines += ["", "## 2. Smart Tagging duran topları (public hariç): yakınında hareketsiz top ve sınıflandırıcının ölü topu", "",
              "Pencere: duran top etiketinin başından 5 s önce ile 1 s sonrası.", "",
              "| tür | sayı | hareketsiz top var | sınıflandırıcı ölü top diyor | ikisinden biri |", "|---|---|---|---|---|"]
    for t in ("goal_kick", "throw_in", "free_kick", "corner", "kickoff", "hepsi"):
        sel = [(a, b) for a, b, k in sp if k == t or t == "hepsi"]
        if not sel:
            continue
        hs = np.array([s[max(0, a - BEFORE):a + AFTER].any() for a, _ in sel])
        hd = np.array([dead[max(0, a - BEFORE):a + AFTER].any() for a, _ in sel])
        lines.append("| {} | {} | {:.0%} | {:.0%} | {:.0%} |".format(t, len(sel), hs.mean(), hd.mean(), (hs | hd).mean()))
        print(lines[-1], flush=True)
    # kinks vs Metrica events
    lines += ["", "## 3. Vuruş izleri (yön ≥ 30° ya da hız ×2 / ÷2, ≥ 2 m/s)", "",
              "| | değer |", "|---|---|"]
    prec, prec0, rec, n_k, alive_min = [], [], [], 0, 0.0
    for gm, m in zip((1, 2), met):
        ev = pd.read_csv(os.path.join(METRICA, "Sample_Game_{}".format(gm), "Sample_Game_{}_RawEventsData.csv".format(gm)))
        alive = np.isin(m["labels"], (pm.H, pm.A))
        k = kinks(m["bx"], m["by"], m["pitch"])
        k = k[alive[k]]
        starts = ev["Start Frame"].values
        touch = ev[ev["Type"].isin(["PASS", "SHOT", "RECOVERY"])]["Start Frame"].values
        prec.append(near_any(k, starts, NEAR).mean())
        prec0.append(near_any(k + FPS, starts, NEAR).mean())
        rec.append(near_any(touch, k, NEAR).mean())
        n_k += len(k)
        alive_min += alive.sum() / FPS / 60
    lines += ["| Metrica: kırılmaların olaya ±3 kare yakın payı (precision) | {:.0%} (şans: {:.0%}) |".format(np.mean(prec), np.mean(prec0)),
              "| Metrica: pas/şut/top kazanmanın ±3 karede kırılması olan payı (recall) | {:.0%} |".format(np.mean(rec)),
              "| Metrica: oyun içi dakikada kırılma | {:.1f} |".format(n_k / alive_min)]
    alive = g & ~dead
    atd_ball = (v2["ball_source"] == 0)
    for name, sel in (("bizim ATD topu", atd_ball), ("bizim dedektör topu (ATD'nin olmadığı kareler)", v2["ball_source"] == 2)):
        bx = np.where(sel, v2["bx"], np.nan)
        by = np.where(sel, v2["by"], np.nan)
        k = kinks(bx, by, v2["pitch"])
        k = k[alive[k]]
        mins = (alive & sel).sum() / FPS / 60
        lines.append("| {}: kendi kapsadığı oyun içi dakikada kırılma | {:.1f} ({:.0f} dk) |".format(name, len(k) / max(mins, 1e-9), mins))
    for line in lines[-6:]:
        print(line)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
