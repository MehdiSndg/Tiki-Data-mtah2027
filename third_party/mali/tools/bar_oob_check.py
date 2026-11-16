"""Label-free sanity check on the provided game: how often does a blend call D while the VISIBLE ball is physically
out of bounds (ball_oob_flag; geometry, no labels)? Also describes the frames where two blends disagree on D.
  tools/bar_oob_check.py <run> [<run> ...]      (official submission is always the reference)
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
from mtah.validation.official import pm  # noqa: E402


def main():
    dt = config_mod.load("main_ext_met_dt")
    dcfg = adapt.blend_cfg(dt, adapt.params(dt)[0])
    ctx = adapt.bar_context(dt)
    data, f = ctx["data"], ctx["feats"]
    regime = contact.regime_for(data, dcfg)
    per = data.period > 0
    vis = f["ball_vis"].to_numpy() > 0
    oobd = f["ball_oob_dist"].to_numpy()
    p80 = f["pl_p80_speed"].to_numpy()
    bs = f["ball_speed"].to_numpy()
    off = np.asarray(pm.label_codes(pm.read_label_csv(str(OUTPUTS / "main" / "submission.csv"))))
    graded = per & (off != 3)
    bands = [("oob > 0.35 m", graded & vis & (oobd > 0.35)), ("oob > 1 m", graded & vis & (oobd > 1.0)),
             ("oob > 2 m", graded & vis & (oobd > 2.0)), ("oob > 2 m for >= 2 s", None)]
    deep = graded & vis & (oobd > 2.0)
    d = np.diff(np.r_[0, deep.astype(np.int8), 0])
    long = np.zeros(len(deep), bool)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= 50:
            long[s:e] = True
    bands[3] = ("oob > 2 m for >= 2 s", long)
    fill = np.load(OUTPUTS / "booster_met" / "labels_ball_fill_dd1.5_dfl1.5_v2.npz")["BAR"]
    mb = np.load(OUTPUTS / "ev_ext_mb_lgbm" / "probs.npy")

    def show(name, lab):
        lab = np.asarray(lab)
        print(f"{name:<34} D share {np.mean(lab[graded] == 2):.3f} | D rate on " +
              "; ".join(f"{b} (n {int(m.sum())}): {np.mean(lab[m] == 2):.3f}" for b, m in bands), flush=True)
        return lab

    show("official f4080c4c", off)
    for run in sys.argv[1:]:
        net, bnd = np.load(OUTPUTS / run / "bar_net_adapted.npy"), np.load(OUTPUTS / run / "bar_bnd_adapted.npy")
        for bname, pb in (("mb", mb), ("drift", fill)):
            lab = show(f"{run}+{bname}", m0.decode_match(log_pool(pb, net, 0.25), dcfg, bnd, regime=regime))
        for nm, m in (("official D, candidate live", graded & (off == 2) & (lab != 2)),
                      ("official live, candidate D", graded & (off != 2) & (lab == 2))):
            print(f"   {nm}: {int(m.sum())} frames ({m.sum() / graded.sum():.3f}) | ball visible {vis[m].mean():.2f} "
                  f"| visible & oob>0.35 {np.mean(vis[m] & (oobd[m] > 0.35)):.3f} | p80 tempo median {np.nanmedian(p80[m]):.2f} "
                  f"| ball speed median {np.nanmedian(bs[m]):.2f}")


if __name__ == "__main__":
    main()
