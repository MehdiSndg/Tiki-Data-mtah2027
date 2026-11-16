"""Does the ball imputer (mtah.ballimpute, P9aa) carry over to our match? GT-free: an imputer trained on DFL (all but
J03WR9) and Metrica 1-3, degraded as in production, is tested on artificial gaps cut where the ball IS seen: 10-30 s
stretches of frames with a ball sighting, blanked, then regressed from the players and compared with the hidden ball.
The same cuts on J03WR9 (degraded, held out) give the reference error. Our match's labels are never read (they are
replaced by X before anything is computed). GPU.

    python analysis/p9ab_imputer_transfer.py      # -> eda_out/reports/P9ab_imputer_transfer.md
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah import ballimpute as BI  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.teammodel import features  # noqa: E402

FPS = 25
HOLD = "J03WR9"
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ab_imputer_transfer.md")


def cuts(m, rng, n_cuts=150):
    """Up to n_cuts disjoint 10-30 s stretches inside a half where the ball is seen on >= 90% of frames."""
    seen = np.isfinite(m["bx"]) & (m["half"] > 0)
    c = np.r_[0, np.cumsum(seen)]
    taken = np.zeros(len(seen), bool)
    out = []
    for _ in range(20 * n_cuts):
        if len(out) >= n_cuts:
            break
        ln = int(rng.uniform(10, 30) * FPS)
        a = int(rng.integers(0, len(seen) - ln))
        b = a + ln
        if taken[a:b].any() or (c[b] - c[a]) < 0.9 * ln or len(set(m["half"][a:b])) > 1 or m["half"][a] <= 0:
            continue
        taken[max(0, a - FPS):b + FPS] = True
        out.append((a, b))
    return out


def blank(m, cs):
    out = dict(m)
    bx, by = np.asarray(m["bx"], np.float32).copy(), np.asarray(m["by"], np.float32).copy()
    for a, b in cs:
        bx[a:b], by[a:b] = np.nan, np.nan
    out["bx"], out["by"] = bx, by
    return out


def errors(net, mu, sd, m, rng):
    cs = cuts(m, rng)
    mb = blank(m, cs)
    X = BI.inputs(features(mb), mb)
    off = BI.predict_offset(net, mu, sd, X, mb["half"] > 0, "cuda")
    f, z = BI.fill(mb, off, 10), BI.fill(mb, np.zeros_like(off), 10)
    sel = np.zeros(len(m["bx"]), bool)
    for a, b in cs:
        sel[a:b] = True
    sel &= np.isfinite(m["bx"])
    L, W = m["pitch"]
    e = lambda q: np.hypot((q["bx"][sel] - m["bx"][sel]) * L, (q["by"][sel] - m["by"][sel]) * W)
    return len(cs), e(f), e(z)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    data, hold = [], None
    for m, (r, k) in zip(base, raws):
        if HOLD in k:
            hold = m
            continue
        data.append((BI.inputs(features(m), m),) + BI.offset_target(m, r["bx"], r["by"]) + (m["half"] > 0,))
    net, mu, sd = BI.train_imputer(data, 0, "cuda")
    ours = from_tactical()
    ours["labels"] = np.full(len(ours["bx"]), pm.X, np.int8)
    ours.pop("public_gt", None)
    lines = ["# P9ab: top tahmincisinin bizim maça taşınması (GT'siz, yapay boşluklar)", "", "`analysis/p9ab_imputer_transfer.py`.", "",
             "Yapay 10–30 s boşluklar (topun görüldüğü yerlerde), hata m: medyan / %75 / %90, tahminci ve son görülen top", ""]
    for name, m in ((HOLD + " (bozulmuş, eğitimde yok)", hold), ("bizim maç", ours)):
        n, ei, el = errors(net, mu, sd, m, np.random.default_rng(0))
        lines.append("- {}: {} boşluk, {} kare; tahminci {:.1f} / {:.1f} / {:.1f}; son görülen {:.1f} / {:.1f} / {:.1f}".format(
            name, n, len(ei), np.median(ei), np.percentile(ei, 75), np.percentile(ei, 90), np.median(el), np.percentile(el, 75), np.percentile(el, 90)))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
