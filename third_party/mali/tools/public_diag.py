"""Public-window DIAGNOSTIC of blend variants (11 Nov). Never a selection input (CLAUDE.md rule 3):
prints aggregate scores of saved BAR probabilities decoded with the official tuned decoder.
  .venv\Scripts\python.exe tools/public_diag.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah import config as config_mod  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, m0  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402
from mtah.validation.scoring import score_blocks, summary_line  # noqa: E402
from mtah.validation.splits import public_blocks  # noqa: E402


def main():
    dt = config_mod.load("main_ext_met_dt")
    dcfg = adapt.blend_cfg(dt, adapt.params(dt)[0])
    data = adapt.bar_context(dt)["data"]
    regime = contact.regime_for(data, dcfg)
    blocks = public_blocks()
    ball = np.load(OUTPUTS / "booster_met" / "labels_ball.npz")["BAR"]
    mb = np.load(OUTPUTS / "ev_ext_mb_lgbm" / "probs.npy")
    fp = OUTPUTS / "booster_met" / "labels_ball_fill.npz"
    fill = np.load(fp)["BAR"] if fp.is_file() else None

    def show(name, labels):
        r = score_blocks(np.asarray(labels), blocks, bootstrap=1000)["ALL"]
        st = sanity.stats(labels)
        print(summary_line(name, r), f" | full match poss {st['n_possessions']} D {st['dead_share']:.3f}", flush=True)

    show("official f4080c4c", pm.label_codes(pm.read_label_csv(str(OUTPUTS / "main" / "submission.csv"))))
    for run in sys.argv[1:] or ["ev_ext_met", "ev_ext_met_s2", "ev_ext_met_bl", "ev_ext_met_bl_s2"]:
        net, bnd = np.load(OUTPUTS / run / "bar_net_adapted.npy"), np.load(OUTPUTS / run / "bar_bnd_adapted.npy")
        extra = [(f.stem.replace("labels_ball_", ""), np.load(f)["BAR"]) for f in sorted((OUTPUTS / "booster_met").glob("labels_ball_*.npz"))]
        for bname, pb in [("mb", mb), ("ball", ball)] + extra:
            show(f"{run}+{bname}", m0.decode_match(log_pool(pb, net, 0.25), dcfg, bnd, regime=regime))


if __name__ == "__main__":
    main()
