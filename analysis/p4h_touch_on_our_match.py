"""P4: the touch labelers on our match. GT-free first; the public window only as a check.

    uv run python analysis/p4h_touch_on_our_match.py [det_dir det_cls]   # -> eda_out/reports/P4h_touch_on_our_match.md

Dead ball from the raw-ATD classifier (p4_common.ours). Ball: the raw ATD (v1), Match v2 (ATD
where it sees the ball, else the given detector; no broadcast), and v3: the detector's own box
wherever it has one (conf >= 0.3, not a spot), the v2 ball elsewhere -- an unsmoothed ball for kinks. For each labeler: kinks and touches
per in-play minute with a ball, possessions per 90 min (external proxies: 320-332; P4c), agreement
of the team call with the Smart Tagging side outside the public window, and the public spellF1.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.timing import kinks  # noqa: E402
from mtah.data import load_tactical  # noqa: E402
from mtah.video_match import YOLO_DIR, detector_ball, from_tactical_v2  # noqa: E402
from p4_common import ours  # noqa: E402
from p4f_touch_labeler import control_touch, touch_only, touches  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4h_touch_on_our_match.md")
PUBLIC = (11185, 18685)


def main():
    det_dir = sys.argv[1] if len(sys.argv) > 1 else YOLO_DIR
    det_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    dead, _ = ours()
    df = load_frames()
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(dead), bool)
    outside[PUBLIC[0]:PUBLIC[1]] = False
    lines = ["# P4h: dokunuş etiketleyicileri bizim maçta", "",
             "`analysis/p4h_touch_on_our_match.py`; v2 dedektörü `{}` (sınıf {}). Ölü top ham ATD sınıflandırıcısından.".format(os.path.relpath(det_dir, ROOT), det_cls),
             "Dış vekillerde 90 dk'da 320–332 possession (P4c). Public sadece kontrol; hiçbir şey ona göre seçilmez.", "",
             "| top | etiketleyici | dakikada kırılma / dokunuş | 90 dk'da possession | Smart Tagging takım uyumu (public dışı) | public spellF1 (kontrol) |",
             "|---|---|---|---|---|---|"]
    v2 = from_tactical_v2(det_dir, det_cls)
    xy, conf, spot = detector_ball(load_tactical()["hom"], len(dead), det_dir, det_cls)
    has = ~np.isnan(xy[:, 0]) & (conf >= 0.3) & ~spot
    v3 = dict(v2, bx=np.where(has, xy[:, 0], v2["bx"]).astype(np.float32), by=np.where(has, xy[:, 1], v2["by"]).astype(np.float32))
    for bname, m in (("v1 (ham ATD)", from_tactical()), ("v2 (ATD + dedektör)", v2), ("v3 (önce dedektör, sonra ATD)", v3)):
        g = m["half"] > 0
        alive = g & ~dead
        mins = (alive & ~np.isnan(m["bx"])).sum() / FPS / 60
        k = kinks(m["bx"], m["by"], m["pitch"])
        rate = "{:.1f} / {:.1f}".format(np.sum(alive[k]) / mins, len(touches(m, alive)[0]) / mins)
        for name, fn in (("control", lambda x: control(x, dead=dead)), ("touch", lambda x: touch_only(x, alive)),
                         ("control+touch", lambda x: control_touch(x, alive=alive))):
            c = np.where(g, fn(m), pm.X).astype(np.int8)
            c = np.where(alive | ~g, c, pm.D).astype(np.int8)
            n = len(pm.possession_spells(c, pm.valid_runs(c))) / (g.sum() / FPS / 60) * 90
            sel = alive & outside & (tag >= 0)
            r = evaluate(lambda _: c, [m], bootstrap=1000)
            lines.append("| {} | {} | {} | {:.0f} | {:.1%} | {:.3f} [{:.2f}–{:.2f}] |".format(
                bname, name, rate, n, np.mean(c[sel] == tag[sel]), r.spell_f1[0], r.boot_low[0], r.boot_high[0]))
            print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
