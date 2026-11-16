"""Diagnostics of saved BAR label arrays (.npy from booster_met --save-bar): public window (diagnostic only) and
the label-free out-of-bounds check.   tools/bar_labels_diag.py <labels.npy> [...]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah import config as config_mod  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402
from mtah.validation.scoring import score_blocks, summary_line  # noqa: E402
from mtah.validation.splits import public_blocks  # noqa: E402

ctx = adapt.bar_context(config_mod.load("main_ext_met_dt"))
f, per = ctx["feats"], ctx["data"].period > 0
vis, oobd = f["ball_vis"].to_numpy() > 0, f["ball_oob_dist"].to_numpy()
off = np.asarray(pm.label_codes(pm.read_label_csv(str(OUTPUTS / "main" / "submission.csv"))))
graded = per & (off != 3)
deep = graded & vis & (oobd > 2.0)
d = np.diff(np.r_[0, deep.astype(np.int8), 0])
long = np.zeros(len(deep), bool)
for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
    if e - s >= 50:
        long[s:e] = True
blocks = public_blocks()
for name, lab in [("official", off)] + [(Path(p).stem, np.load(p)) for p in sys.argv[1:]]:
    lab = np.asarray(lab)
    r = score_blocks(lab, blocks, bootstrap=0)["ALL"]
    st = sanity.stats(lab)
    print(summary_line(name, r)[:118], f"| poss {st['n_possessions']} D {st['dead_share']:.3f} home {st['home_share']:.3f}",
          f"| OOB>0.35 D {np.mean(lab[graded & vis & (oobd > 0.35)] == 2):.3f} OOB>2m>=2s D {np.mean(lab[long] == 2):.3f}",
          f"| differs from official {np.mean(lab[graded] != off[graded]):.3f}", flush=True)
