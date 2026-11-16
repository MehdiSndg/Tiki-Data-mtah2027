"""Why does the learned team call split our match less than degraded DFL? GT-free.

    python analysis/p6g_shift.py      # -> eda_out/reports/P6g_shift.md

1. Adversarial validation: a classifier told to separate alive frames of our match (v1 chain's
   dead ball) from alive frames of degraded DFL (ball smoothed over 25 frames), on the team
   features; its AUROC and the features it leans on most (permutation importance on a held-out
   part) say which inputs differ.
2. Confidence of the team model (trained on all degraded external data): share of alive frames
   with P(home) within 0.35-0.65, and the mean |log-odds|, ours vs DFL.
3. Per-feature medians, ours vs DFL, for the top features.
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
from mtah.teammodel import fit, team_features  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6g_shift.md")
N = 60000


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl
    from mtah.match import from_tactical
    from p4_common import ours
    rng = np.random.default_rng(0)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ext = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    Fe, ye = [], []
    for e in ext:
        F, names = team_features(e)
        a = np.isin(e["labels"], (pm.H, pm.A))
        Fe.append(F[a])
        ye.append((e["labels"][a] == pm.H).astype(int))
    clf = fit(Fe, ye)
    m = from_tactical()
    dead, _ = ours()
    Fo, _ = team_features(m)
    Fo = Fo[(m["half"] > 0) & ~dead]
    De = np.concatenate(Fe)
    a = De[rng.choice(len(De), N, replace=False)]
    b = Fo[rng.choice(len(Fo), N, replace=False)]
    X = np.r_[a, b]
    y = np.r_[np.zeros(N), np.ones(N)]
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    adv = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], adv.predict_proba(X[te])[:, 1])
    imp = permutation_importance(adv, X[te][:20000], y[te][:20000], n_repeats=3, random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)[:15]
    pe, po = clf.predict_proba(a)[:, 1], clf.predict_proba(b)[:, 1]
    lo = lambda p: np.abs(np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1)))
    lines = ["# P6g: bizim maç ile bozulmuş DFL arasındaki kayma (GT'siz)", "",
             "`analysis/p6g_shift.py`. Oyun içi kareler, her taraftan {} örnek.".format(N), "",
             "Ayırıcı sınıflandırıcının AUROC'u: {:.3f} (0.5 = ayırt edilemez).".format(auc), "",
             "Takım modelinin kararsızlığı: P(ev) 0.35–0.65 arası kare payı DFL {:.1%}, bizim {:.1%}; ortalama |log-odds| DFL {:.2f}, bizim {:.2f}.".format(
                 np.mean(np.abs(pe - 0.5) < 0.15), np.mean(np.abs(po - 0.5) < 0.15), lo(pe).mean(), lo(po).mean()), "",
             "| özellik | önem (AUROC düşüşü) | DFL medyan | bizim medyan | DFL NaN | bizim NaN |", "|---|---|---|---|---|---|"]
    for k in order:
        lines.append("| {} | {:.3f} | {:.2f} | {:.2f} | {:.1%} | {:.1%} |".format(
            names[k], imp.importances_mean[k], np.nanmedian(a[:, k]), np.nanmedian(b[:, k]),
            np.mean(np.isnan(a[:, k])), np.mean(np.isnan(b[:, k]))))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
