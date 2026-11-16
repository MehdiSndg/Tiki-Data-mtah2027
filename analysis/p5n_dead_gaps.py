"""Open item 14 / inventory E60: the dead-ball classifier calls 61-74% of the in-play frames without a
ball "dead". Where, and do features of the gap itself help? External data, cross-dataset, 3 seeds.

    uv run python analysis/p5n_dead_gaps.py      # -> eda_out/reports/P5n_dead_gaps.md

1. Diagnosis: on degraded DFL and Metrica, the in-play frames without a ball (alive in the labels,
   no ball after degradation), the share the m1 classifier calls dead, by gap length, where the
   ball was last seen, how fast the players move, and whether a keeper holds the ball.
2. Gap features, all GT-free and computable on our ATD (the classifier may look forward):
   where the ball reappears (distance from where it vanished, the rate of that move, distance of
   the reappearance to the lines and to the restart spots), its speed when it reappears, and how
   far the players' centroid moved during the gap. The m1 classifier with and without them:
   balanced accuracy, dead share of in-play gaps, recall of the true dead ball, and the control
   rule's spellF1 with the predicted dead ball (the full chain).
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import features, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from p5h_keeper_hold import held  # noqa: E402

FPS = 25
SEEDS = (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5n_dead_gaps.md")


def gap_features(m):
    L, W = m["pitch"]
    n = len(m["bx"])
    bX, bY = m["bx"].astype(float) * L, m["by"].astype(float) * W
    seen = ~np.isnan(bX)
    idx = np.arange(n)
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().values
    nxt = pd.Series(np.where(seen, idx, np.nan)).bfill().values
    li, ni = np.nan_to_num(last, nan=0).astype(int), np.nan_to_num(nxt, nan=n - 1).astype(int)
    lX, lY, nX, nY = bX[li], bY[li], bX[ni], bY[ni]
    dur = np.maximum(1, ni - li) / FPS
    move = np.hypot(nX - lX, nY - lY)
    spots = [(L / 2, W / 2), (0, 0), (0, W), (L, 0), (L, W), (11, W / 2), (L - 11, W / 2)]
    spot_d = np.min([np.hypot(nX - sx, nY - sy) for sx, sy in spots], axis=0)
    step = np.r_[np.hypot(np.diff(bX), np.diff(bY)) * FPS, np.nan]
    sp_after = pd.Series(np.minimum(step, 40)).rolling(10, min_periods=1).median().shift(-9).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmean(m["px"].astype(float), axis=1) * L
    cshift = np.abs(cx[ni] - cx[li]) / dur
    f = {"reappear_d": move, "reappear_rate": move / dur, "next_line_d": np.minimum(np.minimum(nY, W - nY), np.minimum(nX, L - nX)),
         "next_spot_d": spot_d, "next_speed": sp_after[ni], "centroid_shift": cshift}
    out = pd.DataFrame(f)
    out[seen] = 0.0
    return out.fillna(-1)


def table(ms, extra):
    Xs, ys = [], []
    for m in ms:
        g = m["half"] > 0
        f = features(m)
        if extra:
            f = pd.concat([f, gap_features(m)], axis=1)
        Xs.append(f[g])
        ys.append(m["labels"][g] == pm.D)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def predict(clf, m, extra):
    g = m["half"] > 0
    f = features(m)
    if extra:
        f = pd.concat([f, gap_features(m)], axis=1)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(f[g])[:, 1]
    return to_blocks(p) & g


def spell_f1(truth, pred):
    return pm.spell_scores(truth, pred, pm.valid_runs(truth))["mean_f1"]


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = {"DFL": [(from_dfl(i), i) for i in ids], "Metrica": [(from_metrica_sample(g), "metrica") for g in (1, 2)]}
    diag, comp = [], []
    for seed in SEEDS:
        deg = {s: [degrade(m, rates_for(k), seed=seed) for m, k in ms] for s, ms in clean.items()}
        for extra in (False, True):
            clf = {s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(*table(v, extra))
                   for s, v in deg.items()}
            for s, ms in deg.items():
                other = clf["Metrica" if s == "DFL" else "DFL"]
                for (cm, _), d in zip(clean[s], ms):
                    alive = np.isin(d["labels"], (pm.H, pm.A))
                    truth = d["labels"] == pm.D
                    g = d["half"] > 0
                    dead = predict(other, d, extra)
                    gap = alive & np.isnan(d["bx"])
                    pred = control(d, dead=dead)
                    comp.append({"src": s, "seed": seed, "extra": extra,
                                 "ba": (np.mean(dead[g & truth]) + np.mean(~dead[g & ~truth])) / 2,
                                 "gap_dead": dead[gap].mean(), "dead_recall": dead[truth].mean(),
                                 "f1": spell_f1(d["labels"], pred)})
                    if extra:
                        continue
                    # diagnosis, base classifier
                    L, W = d["pitch"]
                    bX = pd.Series(d["bx"].astype(float) * L).ffill().values
                    bY = pd.Series(d["by"].astype(float) * W).ffill().values
                    runs = np.diff(np.r_[0, np.isnan(d["bx"]).astype(np.int8), 0])
                    glen = np.zeros(len(gap))
                    for a, z in zip(np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)):
                        glen[a:z] = (z - a) / FPS
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        X, Y = d["px"].astype(float) * L, d["py"].astype(float) * W
                        sp = np.nanmedian(np.hypot(np.r_[np.full((5, X.shape[1]), np.nan), X[5:] - X[:-5]],
                                                   np.r_[np.full((5, X.shape[1]), np.nan), Y[5:] - Y[:-5]]) * FPS / 5, axis=1)
                    where = np.where(np.minimum(bY, W - bY) < 3, "yan çizgiye < 3 m",
                                     np.where(np.minimum(bX, L - bX) < 16.5, "kaleye < 16.5 m", "orta"))
                    kh = held(cm) & gap
                    for name, key in (("boşluk süresi", np.digitize(glen, [2, 5, 10])), ("topun son görüldüğü yer", where),
                                      ("oyuncu hızı medyanı", np.digitize(sp, [1.4, 2.5])), ("kaleci tutuyor", kh.astype(int))):
                        for v in np.unique(key[gap]):
                            sel = gap & (key == v)
                            diag.append({"src": s, "seed": seed, "by": name, "value": str(v), "share": sel.sum() / gap.sum(),
                                         "dead": dead[sel].mean()})
            print("seed", seed, flush=True)
    D = pd.DataFrame(diag).groupby(["by", "value", "src"])[["share", "dead"]].mean().unstack("src")
    C = pd.DataFrame(comp).groupby(["src", "extra"])[["ba", "gap_dead", "dead_recall", "f1"]].agg(["mean", "min", "max"])
    names = {"boşluk süresi": {"0": "< 2 s", "1": "2–5 s", "2": "5–10 s", "3": "≥ 10 s"},
             "oyuncu hızı medyanı": {"0": "< 1.4 m/s", "1": "1.4–2.5", "2": "≥ 2.5"}, "kaleci tutuyor": {"0": "hayır", "1": "evet"}}
    lines = ["# P5n: oyun sürerken top yokken sınıflandırıcının \"ölü\" demesi (E60, madde 14)", "",
             "`analysis/p5n_dead_gaps.py`. Bozulmuş dış veri, çapraz veri seti, 3 tohum ortalaması.", "",
             "## 1. Teşhis (temel sınıflandırıcı): oyun içi boşluk karelerinde \"ölü\" denen pay", "",
             "| kırılım | değer | DFL: boşlukların payı | DFL: ölü denen | Metrica: payı | Metrica: ölü denen |", "|---|---|---|---|---|---|"]
    for (by, v), r in D.iterrows():
        lines.append("| {} | {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(by, names.get(by, {}).get(v, v),
                     r[("share", "DFL")], r[("dead", "DFL")], r[("share", "Metrica")], r[("dead", "Metrica")]))
    lines += ["", "## 2. Boşluk özellikleri eklenince (ortalama [en düşük–en yüksek], maçlar ve tohumlar)", "",
              "| kaynak | özellikler | dengeli doğruluk | oyun içi boşlukta \"ölü\" | gerçek ölü topu yakalama | tam zincir spellF1 |",
              "|---|---|---|---|---|---|"]
    for (s, extra), r in C.iterrows():
        cell = lambda c, f: "{} [{}–{}]".format(f.format(r[(c, "mean")]), f.format(r[(c, "min")]), f.format(r[(c, "max")]))
        lines.append("| {} | {} | {} | {} | {} | {} |".format(s, "m1 + boşluk" if extra else "m1", cell("ba", "{:.3f}"),
                                                             cell("gap_dead", "{:.0%}"), cell("dead_recall", "{:.0%}"), cell("f1", "{:.3f}")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
