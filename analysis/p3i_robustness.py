"""P3: do small detector differences change the submission labels? GT-free.

    uv run python analysis/p3i_robustness.py [det_dir det_cls]     # -> eda_out/reports/P3_robustness.md

The determinism check (eda_out/reports/P3_determinism.md) found GPU fp32 and CPU fp32 differ by a few
boxes (6 in 200 frames, confidence up to 0.15 apart). Here the per-frame detector track that goes
into Match v2 is perturbed the way such a difference would: a share of the frames with a box lose
it, confidences move by Gaussian noise (crossing the 0.3 gate both ways), positions move a little.
The dead ball comes from the raw ATD (v1), as in the submission chain, so only the team call can
move. Reported per perturbation, over 5 seeds: the share of in-play frames whose label changes, and
spellF1 of the perturbed labels against the unperturbed ones (1.0 = the same spells).
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
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.video_match import YOLO_DIR, detector_ball, from_tactical_v2  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P3_robustness.md")
SEEDS = range(5)
# name: (share of box frames dropped, confidence noise sd, position noise sd in m)
PERTURB = {"kutu sil %1": (0.01, 0.0, 0.0), "güven ±0.05": (0.0, 0.05, 0.0),
           "donanım benzeri (%1 sil, güven ±0.05, konum 0.2 m)": (0.01, 0.05, 0.2),
           "kaba (%5 sil, güven ±0.15)": (0.05, 0.15, 0.0)}


def perturb(det, drop, conf_sd, pos_sd, rng):
    xy, conf, spot = det[0].copy(), det[1].copy(), det[2].copy()
    has = np.flatnonzero(~np.isnan(xy[:, 0]))
    gone = rng.choice(has, int(round(drop * len(has))), replace=False)
    xy[gone], conf[gone], spot[gone] = np.nan, 0.0, False
    conf[has] = np.clip(conf[has] + rng.normal(0, conf_sd, len(has)), 0, 1) if conf_sd else conf[has]
    if pos_sd:
        xy[has] += rng.normal(0, pos_sd, (len(has), 2)) / np.array(PITCH)
    return xy, conf, spot


def main():
    det_dir = sys.argv[1] if len(sys.argv) > 1 else YOLO_DIR
    det_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids] + \
            [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(train))
    v1 = from_tactical()
    g = v1["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(v1)[g])[:, 1]
    dead = (to_blocks(p) | np.isin(load_frames()["period"].values, NOT_PLAYED)) & g
    n = len(g)
    det = detector_ball(load_tactical()["hom"], n, det_dir, det_cls)

    def labels(track):
        m = from_tactical_v2(det=track)
        return np.where(g, control(m, dead=dead), pm.X).astype(np.int8)

    base = labels(det)
    runs = pm.valid_runs(base)
    alive = g & ~dead
    print("base: {} spells, {} frames with a box".format(len(pm.possession_spells(base, runs)), int((~np.isnan(det[0][:, 0])).sum())), flush=True)
    lines = ["# P3: etiketler küçük dedektör farklarına ne kadar dayanıklı?", "",
             "`analysis/p3i_robustness.py`; dedektör `{}` (sınıf {}), ölü top ham ATD'den (v1), {} tohum.".format(
                 os.path.relpath(det_dir, ROOT), det_cls, len(SEEDS)), "",
             "| bozma | değişen oyun içi kare (ort., en çok) | spellF1 bozulmuşa karşı bozulmamış (ort., en kötü) |", "|---|---|---|"]
    for name, (drop, conf_sd, pos_sd) in PERTURB.items():
        ch, sf = [], []
        for s in SEEDS:
            lab = labels(perturb(det, drop, conf_sd, pos_sd, np.random.default_rng(s)))
            ch.append(np.mean(lab[alive] != base[alive]))
            sf.append(pm.spell_scores(base, lab, runs)["mean_f1"])
        lines.append("| {} | {:.3%}, {:.3%} | {:.4f}, {:.4f} |".format(name, np.mean(ch), np.max(ch), np.mean(sf), np.min(sf)))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
