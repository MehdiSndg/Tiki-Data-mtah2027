"""Public-window DIAGNOSTIC (aggregates only, never selection): where do two blend variants differ on GT-dead frames?
  tools/public_dead_diag.py <run> <boosterA> <boosterB>     boosters: mb | ball | fill
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
from mtah.validation.splits import merged_gt, public_blocks  # noqa: E402


def booster(name):
    if name == "mb":
        return np.load(OUTPUTS / "ev_ext_mb_lgbm" / "probs.npy")
    return np.load(OUTPUTS / "booster_met" / {"ball": "labels_ball.npz", "fill": "labels_ball_fill.npz"}[name])["BAR"]


def main():
    run, a, b = sys.argv[1:4]
    dt = config_mod.load("main_ext_met_dt")
    dcfg = adapt.blend_cfg(dt, adapt.params(dt)[0])
    ctx = adapt.bar_context(dt)
    data, feats = ctx["data"], ctx["feats"]
    gt = merged_gt(public_blocks())
    net, bnd = np.load(OUTPUTS / run / "bar_net_adapted.npy"), np.load(OUTPUTS / run / "bar_bnd_adapted.npy")
    regime = contact.regime_for(data, dcfg)
    vis = ~np.isnan(data.ball[:, 0])
    cols = [c for c in ("ball_speed", "pl_p80_speed", "pl_mean_speed", "ball_oob_dist", "ball_gap_len") if c in feats.columns]
    dead = gt == 2
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    segs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    print("GT dead segments (s):", [round((e - s) / 25, 1) for s, e in segs], "| dead frames", int(dead.sum()), "ball visible %.2f" % vis[dead].mean())
    out = {}
    for name in (a, b):
        pb = booster(name)
        lab = np.asarray(m0.decode_match(log_pool(pb, net, 0.25), dcfg, bnd, regime=regime))
        out[name] = (lab, pb)
        print(f"\n[{name}] D recall {np.mean(lab[dead] == 2):.2f}; per segment recall:",
              [round(float(np.mean(lab[s:e] == 2)), 2) for s, e in segs])
        print("   per segment: missed frames at start / end (s):",
              [(round(float(np.argmax(lab[s:e] == 2)) / 25, 1) if (lab[s:e] == 2).any() else None,
                round(float(np.argmax(lab[s:e][::-1] == 2)) / 25, 1) if (lab[s:e] == 2).any() else None) for s, e in segs])
        for nm, m in (("visible", dead & vis), ("hidden", dead & ~vis)):
            print(f"   GT dead & ball {nm}: n {int(m.sum())} recall {np.mean(lab[m] == 2):.2f} booster P(D) median {np.median(pb[m, 2]):.2f} net P(D) median {np.median(net[m, 2]):.2f}")
        live = (gt < 2)
        print(f"   GT live called D: {np.mean(lab[live] == 2):.3f}")
    la, lb = out[a][0], out[b][0]
    lost = dead & (la == 2) & (lb != 2)
    print(f"\nGT-dead frames D under {a} but not under {b}: {int(lost.sum())} | ball visible {vis[lost].mean():.2f}")
    for c in cols:
        x = feats[c].to_numpy()
        print(f"   {c}: lost median {np.nanmedian(x[lost]):.2f} | all GT dead {np.nanmedian(x[dead]):.2f} | GT live {np.nanmedian(x[gt < 2]):.2f}")
    print("   booster P(D) on lost frames: %s %.2f -> %s %.2f" % (a, np.median(out[a][1][lost, 2]), b, np.median(out[b][1][lost, 2])))


if __name__ == "__main__":
    main()
