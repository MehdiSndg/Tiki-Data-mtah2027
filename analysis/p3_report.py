"""P3 Task 5: what does the video layer change on our match? GT-free first, public as a check.

    uv run python analysis/p3_report.py [det_dir det_cls]     # -> eda_out/reports/P3.md

v1 = mtah.match.from_tactical (raw ATD), v2 = mtah.video_match.from_tactical_v2 (fused ball,
recovered keepers). Both go through the same chain: the dead-ball classifier
(analysis/m1_dead_classifier.py, trained on all degraded external data, P1 periods applied) and
the control rule. Measured:
  in-play ball coverage   share of frames the classifier calls alive with a ball position
  keeper slots            share of play with a verified, recovered, or no keeper position
  dead ball               share, blocks, median (Metrica sample: ~37%, ~91 per match, 16-19 s)
  Smart Tagging side      agreement of the team call with the tagger's side, outside the public window
  public window           spellF1 with bootstrap: a correctness check, nothing is chosen on it
"""

import glob
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED, features, table, to_blocks  # noqa: E402
from mtah.data import GK_SLOTS  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import FIRST, NOT_PLAYED, SECOND  # noqa: E402
from mtah.video_match import YOLO_DIR, from_tactical_v2  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P3.md")
PUBLIC = (11185, 18685)


def blocks(dead):
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    return (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / FPS


def main():
    det_dir = sys.argv[1] if len(sys.argv) > 1 else YOLO_DIR
    det_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids] + \
            [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(train))
    df = load_frames()
    period = df["period"].values
    play = np.isin(period, [FIRST, SECOND])
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(period), bool)
    outside[PUBLIC[0]:PUBLIC[1]] = False
    lines = ["# P3: video katmanı bizim maçta neyi değiştiriyor?", "",
             "`analysis/p3_report.py`; dedektör: `{}` (sınıf {}). GT'siz ölçüler önce; public sadece kontrol.".format(
                 os.path.relpath(det_dir, ROOT), det_cls), "",
             "| | v1 (ham ATD) | v2 (video katmanı) | ölü top v1'den, takım kararı v2'den |", "|---|---|---|---|"]
    rows = {}
    v1, v2 = from_tactical(), from_tactical_v2(det_dir, det_cls)
    deads = {}
    for name, m in (("v1", v1), ("v2", v2)):
        g = m["half"] > 0
        p = np.zeros(len(g))
        p[g] = clf.predict_proba(features(m)[g])[:, 1]
        deads[name] = (to_blocks(p) | np.isin(period, NOT_PLAYED)) & g
    for name, m, dead in (("v1", v1, deads["v1"]), ("v2", v2, deads["v2"]), ("v2t", v2, deads["v1"])):
        g = m["half"] > 0
        alive = g & ~dead
        codes = np.where(g, control(m, dead=dead), pm.X).astype(np.int8)
        L = blocks(dead)
        r = {"ball": np.mean(~np.isnan(m["bx"][alive])),
             "dead": "{:.0%}, {} blok, medyan {:.1f} s".format(dead[g].mean(), len(L), np.median(L)),
             "side": np.mean(codes[alive & outside & (tag >= 0)] == tag[alive & outside & (tag >= 0)])}
        for t, s in ((0, "H"), (1, "A")):
            k = GK_SLOTS[t]
            r["gk" + s] = np.mean(~np.isnan(m["px"][play, k]))
        res = evaluate(lambda _: codes, [m], bootstrap=1000)
        r["public"] = "{:.3f} [{:.2f}–{:.2f}], {} / {} possession".format(res.spell_f1[0], res.boot_low[0], res.boot_high[0], res.n_pred[0], res.n_gt[0])
        rows[name] = r
        print(name, r, flush=True)
    for key, label, fmt in (("ball", "oyun içi top kapsamı", "{:.1%}"), ("gkH", "Barça kaleci konumu var (oyunda)", "{:.1%}"),
                            ("gkA", "Athletic kaleci konumu var (oyunda)", "{:.1%}"), ("dead", "ölü top", "{}"),
                            ("side", "Smart Tagging takım uyumu (public dışı)", "{:.1%}"), ("public", "public spellF1 (kontrol)", "{}")):
        lines.append("| {} | {} | {} | {} |".format(label, *(fmt.format(rows[k][key]) for k in ("v1", "v2", "v2t"))))
    lines += ["", "v2'nin topu ölü topta da dolu (yayın ATD'si, yerine konan top): ölü top sınıflandırıcısı \"top yok\"",
              "ipucunu kaybediyor ve eğitildiği dağılımın dışına çıkıyor. Ölü top tespiti ham ATD'den yapılmalı.", "",
              "v2'de kaleci konumu sadece görüntüyle doğrulanmış ya da forma rengiyle kurtarılmış karelerde var;",
              "v1'deki konum ise hayalet ve ekran dışı karelerde de dolu ama yanlış olabiliyor."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
