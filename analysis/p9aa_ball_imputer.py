"""A real ball filler for long gaps (P9y: the true ball on the in-play frames of gaps >= 10 s gives +0.015 on DFL,
5 / 7, and about +0.012 with 2-4 m of noise). mtah.ballimpute's player-set GRU, one per match trained on the other
nine (DFL + Metrica degraded as in production), regresses the lost ball from the players; each match's lost ball is
then filled from the imputer that did not see it, in gaps >= 10 s (imp10) or in every gap (imp0), in train and test
alike. Team GRU (v11 recipe), DFL leave one match out and Metrica 1-3, team call only, flat switch, fresh seeds (3, 4,
5); also scored against labels merged at 3.0 s. Imputer error on the in-play frames of long gaps, against the last
sighting. A test match's imputer saw the training matches' balls, and theirs saw the test ball (a weak leak through
the training fills, accepted for a decision run). GPU.

    python analysis/p9aa_ball_imputer.py [only=<match>,...]      # -> eda_out/reports/P9aa_ball_imputer.md
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
from mtah import ballimpute as BI  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

SEEDS = (3, 4, 5)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("none", "imp10", "imp0")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9aa_ball_imputer{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def fit_predict(train, test):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda") for s in SEEDS]
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def long_inplay(m, raw, min_gap_s=10):
    lost = np.isnan(m["bx"]) & (m["half"] > 0)
    use = np.zeros(len(lost), bool)
    e = np.diff(np.r_[0, lost.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if (b - a) / FPS >= min_gap_s:
            use[a:b] = True
    return use & np.isin(m["labels"], (pm.H, pm.A)) & np.isfinite(raw["bx"])


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    names = [m["name"] for m in base]
    Ft = [features(m) for m in base]
    imp = [(BI.inputs(F, m),) + BI.offset_target(m, r["bx"], r["by"]) + (m["half"] > 0,) for F, m, (r, _) in zip(Ft, base, raws)]
    print("imputer inputs done", flush=True)
    offs, err = [], []
    for j in range(len(base)):
        net, mu, sd = BI.train_imputer([d for k, d in enumerate(imp) if k != j], 0, "cuda")
        off = BI.predict_offset(net, mu, sd, imp[j][0], imp[j][3], "cuda")
        offs.append(off)
        m, r = base[j], raws[j][0]
        sel = long_inplay(m, r)
        L, W = m["pitch"]
        f = BI.fill(m, off, 10)
        e_imp = np.hypot((f["bx"][sel] - r["bx"][sel]) * L, (f["by"][sel] - r["by"][sel]) * W)
        z = BI.fill(m, np.zeros_like(off), 10)
        e_last = np.hypot((z["bx"][sel] - r["bx"][sel]) * L, (z["by"][sel] - r["by"][sel]) * W)
        err.append((names[j], int(sel.sum()), np.median(e_imp), np.percentile(e_imp, 75), np.median(e_last), np.percentile(e_last, 75)))
        print("imputer", err[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT[:-3] + "_offsets.npz", **{n: o for n, o in zip(names, offs)})
    data = {}
    for v in VARIANTS:
        ms = base if v == "none" else [BI.fill(m, o, 10 if v == "imp10" else 0) for m, o in zip(base, offs)]
        data[v] = [(m, features(m), features(swap_teams(m))) for m in ms]
        print("features", v, flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    for t in range(len(base)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
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
            pred = np.where(g, decode(fit_predict(tr, (F, g)), alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9aa: oyunculardan top tahmini ile uzun boşluk dolgusu (GRU, sadece takım, tohum 3–5)", "", "`analysis/p9aa_ball_imputer.py`.", "",
             "Tahminci hatası, ≥ 10 s boşlukların oyun karelerinde (m): maç, kare, medyan / %75 tahminci, medyan / %75 son görülen top", ""]
    lines += ["- {}: {}, {:.1f} / {:.1f}, {:.1f} / {:.1f}".format(*e) for e in err]
    lines += ["", "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += [""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for v in VARIANTS[1:]:
                for sfx in ("", " @3s"):
                    a, b = q[v + sfx], q["none" + sfx]
                    lines.append("{}{}: {} − none {:+.3f}, {} / {}".format(src, sfx, v, (a - b).mean(), int((a > b).sum()), len(q)))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
