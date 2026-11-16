"""Where does our match differ from the degraded DFL the models learn on? (Our match under-splits whatever the decode, P9s;
P6g separated the two with AUROC 0.997.) For every production team feature (mtah.teammodel.features): its univariate
AUROC between our match's played frames and the 7 degraded DFL matches' played frames, and the standardised mean shift
(our - DFL, in DFL standard deviations); also the frames where the ball is alive by our v14 dead ball vs DFL's labels.
No labels of our match are read. CPU, light.

    python analysis/p9t_domain_shift.py      # -> eda_out/reports/P9t_domain_shift.md
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.teammodel import TOUCHES, MOTION, team_features  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P9t_domain_shift.md")


def auroc(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    x = np.r_[a, b]
    r = pd.Series(x).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    sub = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v14_chain.csv")).label.values
    ours = from_tactical()
    F, names = team_features(ours, TOUCHES, MOTION)
    alive = np.isin(sub, ("H", "A"))
    Fo = F[alive]
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    Fd = []
    for i in ids:
        m = degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0)
        f, _ = team_features(m, TOUCHES, MOTION)
        Fd.append(f[np.isin(m["labels"], (pm.H, pm.A))])
        print(i, flush=True)
    Fd = np.concatenate(Fd)
    rows = []
    rng = np.random.default_rng(0)
    so = rng.choice(len(Fo), min(len(Fo), 60000), replace=False)
    sd_ = rng.choice(len(Fd), min(len(Fd), 120000), replace=False)
    for j, n in enumerate(names):
        a, b = Fo[so, j].astype(float), Fd[sd_, j].astype(float)
        mu, sd = np.nanmean(b), np.nanstd(b) + 1e-9
        rows.append({"feature": n, "auroc": auroc(a, b), "shift_sd": (np.nanmean(a) - mu) / sd, "ours_mean": np.nanmean(a), "dfl_mean": mu})
    df = pd.DataFrame(rows)
    df["sep"] = (df.auroc - 0.5).abs()
    df = df.sort_values("sep", ascending=False)
    lines = ["# P9t: bizim maç ve bozulmuş DFL, takım özelliklerinde (oyundaki kareler)", "", "`analysis/p9t_domain_shift.py`.", "",
             "| özellik | AUROC | kayma (DFL sd) | bizim ort. | DFL ort. |", "|---|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:+.2f} | {:.3g} | {:.3g} |".format(r.feature, r.auroc, r.shift_sd, r.ours_mean, r.dfl_mean) for _, r in df.head(25).iterrows()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
