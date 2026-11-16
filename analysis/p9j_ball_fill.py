"""How much would a better ball track give our team call? The second teammate's match-specific video detector brings
BAR-ATH's lost ball from 40.7% of frames to 27.3% (mostly short gaps); their simulation said +0.04 for 30% recovered,
while our realistic second-source fill gave the GRU -0.003 (P7b) and a perfect fill was never measured with the GRU.
Here the degraded DFL ball is restored from the true one, in train and test alike: none; each gap shorter than 10 s
with probability 0.5 ("short half"); every gap shorter than 10 s ("short all"); every gap ("all"). Team GRU (v11
recipe: DFL + Metrica, swapped, short x2), DFL leave one match out, team call only (dead ball from the labels), flat
switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p9j_ball_fill.py [only=<match>,...]      # -> eda_out/reports/P9j_ball_fill.md
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = {"none": (0.0, 0.0), "short half": (10.0, 0.5), "short all": (10.0, 1.0), "all": (1e9, 1.0)}
OUT = os.path.join(ROOT, "eda_out", "reports", "P9j_ball_fill{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def fill(deg, raw, max_s, p, seed):
    """deg with the true ball put back in gaps (runs where deg lost a ball raw has) shorter than max_s, each with
    probability p; returns the match and the share of lost frames restored."""
    bx, by = deg["bx"].copy(), deg["by"].copy()
    lost = np.isnan(bx) & np.isfinite(raw["bx"]) & (deg["half"] > 0)
    rng = np.random.RandomState(seed)
    e = np.diff(np.r_[0, lost.astype(np.int8), 0])
    got = 0
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if (b - a) / FPS < max_s and rng.rand() < p:
            bx[a:b], by[a:b] = raw["bx"][a:b], raw["by"][a:b]
            got += b - a
    out = dict(deg)
    out["bx"], out["by"] = bx.astype(np.float32), by.astype(np.float32)
    return out, got / max(1, lost.sum())


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    data, share = {}, {}
    for v, (mx, p) in VARIANTS.items():
        ms, sh = zip(*[fill(d, r, mx, p, 17 + k) for k, (d, (r, _)) in enumerate(zip(base, raws))])
        share[v] = float(np.mean(sh[:len(ids)]))
        data[v] = [(m, features(m), features(swap_teams(m))) for m in ms]
        print("features", v, round(share[v], 3), flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    for t in range(len(ids)):
        if ONLY and not any(o in ids[t] for o in ONLY):
            continue
        r = {"match": ids[t]}
        for v in VARIANTS:
            tr = []
            for k, (m, F, Fs) in enumerate(data[v]):
                if k == t:
                    continue
                sw, sh = swap_teams(m), G.short_mask(m["labels"])
                for mm, FF in ((m, F), (sw, Fs)):
                    tr.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            m, F, _ = data[v][t]
            g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
            p = fit_predict(tr, (F, g), seeds=SEEDS)
            r[v] = pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9j: top dolgusu simülasyonu, GRU (DFL, sadece takım, tohum 3–5)", "", "`analysis/p9j_ball_fill.py`.", "",
             "Geri gelen kayıp kare payı (DFL ort.): " + ", ".join("{} %{:.0f}".format(v, 100 * share[v]) for v in VARIANTS), "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    lines += ["{} − none: {:+.3f}, {} / {}".format(v, (df[v] - df["none"]).mean(), int((df[v] > df["none"]).sum()), len(df)) for v in list(VARIANTS)[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
