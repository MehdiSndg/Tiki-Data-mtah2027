"""Where does the perfect-ball-fill gain (P9j: +0.021) come from, and how accurate must a filler be? (Research report
"Yapısal sıçrama adayları", item 6, decision run before building any ball imputer.) The degraded DFL ball restored
from the true one, in train and test alike: none; only in-play frames of gaps >= 10 s; only dead frames of gaps
>= 10 s; every gap with time-correlated noise of sigma 1 / 2 / 4 / 8 m. Team GRU (v11 recipe), DFL leave one match out,
team call only, flat switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p9y_ball_fill_split.py [only=<match>,...]      # -> eda_out/reports/P9y_ball_fill_split.md
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
VARIANTS = {"none": None, "long in-play": ("long", "alive", 0.0), "long dead": ("long", "dead", 0.0), "all s1": ("all", "any", 1.0),
            "all s2": ("all", "any", 2.0), "all s4": ("all", "any", 4.0), "all s8": ("all", "any", 8.0)}
OUT = os.path.join(ROOT, "eda_out", "reports", "P9y_ball_fill_split{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def fill(deg, raw, spec, seed):
    """deg with the true ball put back per spec = (gaps: "long" (>= 10 s) / "all", frames: "alive" / "dead" / "any",
    noise sigma in m, smooth over ~2 s); returns the match and the share of lost frames restored."""
    bx, by = deg["bx"].copy(), deg["by"].copy()
    lost = np.isnan(bx) & np.isfinite(raw["bx"]) & (deg["half"] > 0)
    if spec is None:
        return dict(deg), 0.0
    gaps, frames, sigma = spec
    rng = np.random.RandomState(seed)
    L, W = deg["pitch"]
    lab = deg["labels"]
    use = np.zeros(len(bx), bool)
    e = np.diff(np.r_[0, lost.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if gaps == "long" and (b - a) / FPS < 10:
            continue
        use[a:b] = True
    if frames == "alive":
        use &= np.isin(lab, (pm.H, pm.A))
    elif frames == "dead":
        use &= lab == pm.D
    nx, ny = np.zeros(len(bx)), np.zeros(len(bx))
    if sigma > 0:
        k = np.ones(2 * FPS) / (2 * FPS)
        nx = np.convolve(rng.normal(0, 1, len(bx)), k, "same")
        ny = np.convolve(rng.normal(0, 1, len(bx)), k, "same")
        nx, ny = nx / (nx.std() + 1e-9) * sigma / L, ny / (ny.std() + 1e-9) * sigma / W
    bx[use], by[use] = raw["bx"][use] + nx[use], raw["by"][use] + ny[use]
    out = dict(deg)
    out["bx"], out["by"] = bx.astype(np.float32), by.astype(np.float32)
    return out, use.sum() / max(1, lost.sum())


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    data, share = {}, {}
    for v, spec in VARIANTS.items():
        ms, sh = zip(*[fill(d, r, spec, 17 + k) for k, (d, (r, _)) in enumerate(zip(base, raws))])
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
    lines = ["# P9y: top dolgusunun kaynağı ve gereken doğruluk (GRU, DFL, sadece takım, tohum 3–5)", "", "`analysis/p9y_ball_fill_split.py`.", "",
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
