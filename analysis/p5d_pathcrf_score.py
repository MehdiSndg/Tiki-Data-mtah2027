"""PathCRF's team call against ours on its held-out DFL match, with the challenge's metric.

    uv run python analysis/p5d_pathcrf_score.py J03WR9 140      # after analysis/p5d_pathcrf_eval.py

PathCRF's per-frame team (eda_out/pathcrf_<match>_<trial>.npz) is put on our Match frames of
from_dfl; frames it does not cover and the dead ball come from the labels, as for the control rule
next to it, so only the team call is compared. Scored against DFL's raw BallPossession and against
the 1.6 s merged labels (our proxy), with spellF1 and the number of possessions.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import _dfl_raw, from_dfl  # noqa: E402
from mtah.labelers import control  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P5d_pathcrf_{}.md")


def main():
    match_id, trial = sys.argv[1], int(sys.argv[2])
    m = from_dfl("DFL-MAT-" + match_id)
    raw = _dfl_raw("DFL-MAT-" + match_id)["raw"]
    d = np.load(os.path.join(ROOT, "eda_out", f"pathcrf_{match_id}_{trial}.npz"), allow_pickle=True)   # our own file
    n1 = int((m["half"] == 1).sum())
    f, per = d["frame_id"].astype(int), d["period_id"].astype(int)
    idx = np.where(per == 1, f - 10000, n1 + 1 + f - 100000)
    team = np.where(d["pred_team"] == "home", pm.H, np.where(d["pred_team"] == "away", pm.A, -1))
    ok = (idx >= 0) & (idx < len(m["bx"])) & (team >= 0)
    rows = []
    for tname, truth in (("DFL ham", raw), ("DFL birleştirilmiş (1.6 s, bizim vekil)", m["labels"])):
        alive = np.isin(truth, (pm.H, pm.A))
        ctrl = control(dict(m, labels=truth), dead="truth")
        pc = ctrl.copy()                                   # PathCRF where it speaks, the rule's frames elsewhere
        pc[idx[ok]] = team[ok]
        pc = np.where(alive, pc, truth).astype(np.int8)
        covered = np.zeros(len(truth), bool)
        covered[idx[ok]] = True
        for name, pred in (("kontrol kuralı", ctrl), ("PathCRF ({})".format(trial), pc)):
            r = pm.evaluate(truth, pred)
            rows.append("| {} | {} | {:.3f} | {:.3f} | {} / {} |".format(
                tname, name, r["spell_f1_mean"], np.mean(pred[alive] == truth[alive]), r["n_pred_spells"], r["n_gt_spells"]))
            print(rows[-1], flush=True)
    lines = ["# P5d: PathCRF bizim ölçümümüzle (DFL'in kendi test maçı {})".format(match_id), "",
             "`analysis/p5d_pathcrf_eval.py` (makaleyi yeniden üretir) ve `analysis/p5d_pathcrf_score.py`. Hazır ağırlık {} "
             "(dinamik maskeli CRF), bu maçı eğitimde görmemiş. Ölü top etiketten: sadece takım kararı.".format(trial), "",
             ("Makaleyi yeniden üretme: kenar doğruluğu 0.695 (makale 0.6964), kare bazında takım possession doğruluğu 0.916 (makale 0.924)."
              if trial == 140 else "Deponun 140 denemesiyle aynı ayarlar, bu maç test olacak şekilde yeniden eğitildi (deneme {})."
              .format(trial)), "",
             "| hedef | etiketleyici | spellF1 | oyun içi takım doğruluğu | possession (tahmin / gerçek) |", "|---|---|---|---|---|"] + rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.format(match_id), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
