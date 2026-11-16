"""Open item 11: is scikit-learn's gradient boosting the right model for the dead ball? GT-free.

    uv run --with lightgbm --with xgboost-cpu python analysis/p5e_tabular_models.py
        # -> eda_out/reports/P5e_tabular_models.md (the two packages are not in the locked env)

The m1 experiment (analysis/m1_dead_classifier.py) with three learners on the same features:
HistGradientBoosting (ours), LightGBM, XGBoost, same depth-free defaults and 300 rounds at 0.05.
Cross-dataset on degraded data: train DFL -> test Metrica, train Metrica -> test DFL. Scored by
dead/alive balanced accuracy and by the control rule's spellF1 with the predicted dead ball.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED, features, table, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P5e_tabular_models.md")


def learners():
    from sklearn.ensemble import HistGradientBoostingClassifier
    import lightgbm
    import xgboost
    return {"HistGradientBoosting (bizim)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED),
            "LightGBM": lambda: lightgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1),
            "XGBoost": lambda: xgboost.XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, tree_method="hist")}


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = {"DFL": [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids],
           "Metrica": [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]}
    tables = {k: table(v) for k, v in src.items()}
    lines = ["# P5e: ölü top sınıflandırıcısı için tablo modelleri", "",
             "`analysis/p5e_tabular_models.py`. m1 deneyi, bozulmuş veride çapraz veri seti. Aynı özellikler, 300 tur, öğrenme hızı 0.05.", "",
             "| model | eğitim → test | ölü/canlı dengeli doğruluk | kontrol kuralı spellF1 | eğitim süresi s |", "|---|---|---|---|---|"]
    for name, make in learners().items():
        for train, test in (("DFL", "Metrica"), ("Metrica", "DFL")):
            t0 = time.time()
            clf = make().fit(*tables[train])
            secs = time.time() - t0
            ba, sf = [], []
            for m in src[test]:
                g = m["half"] > 0
                p = np.zeros(len(g))
                p[g] = clf.predict_proba(features(m)[g])[:, 1]
                dead = to_blocks(p) & g
                truth = m["labels"] == pm.D
                ba.append((np.mean(dead[g & truth]) + np.mean(~dead[g & ~truth])) / 2)
                sf.append(evaluate(lambda x: control(x, dead=dead), [m], bootstrap=0).spell_f1[0])
            lines.append("| {} | {} → {} | {:.3f} | {:.3f} | {:.0f} |".format(name, train, test, np.mean(ba), np.mean(sf), secs))
            print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
