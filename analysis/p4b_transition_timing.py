"""P4 Task 3: when is a hand-over marked, relative to the new team's first contact with the ball?

    uv run python analysis/p4b_transition_timing.py     # -> eda_out/reports/P4b_transition_timing.md

For every direct H<->A change in a labeling, mtah.timing.contact_lag: seconds from the onset of the
new team's nearest contact (a player within R m of the ball) to the label change; + = label later.
Labelings:
  Metrica proxy    our labels from Metrica's events (the new team's first owning event)
  DFL raw          DFL's own BallPossession, before our 1.6 s merge
  control rule     on the same clean matches, dead ball from the labels; and on our match (v1 chain,
                   dead ball from the classifier), GT-free
  public GT        description only, on our ATD (often without a ball: many hand-overs have no lag)
Also: the signed offset from each proxy hand-over to the nearest control-rule hand-over to the same
team within 3 s (+ = the rule is later), and the share with no rule hand-over that close.
"""

import glob
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED, features, table, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, _dfl_raw, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.timing import contact_lag, transitions  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4b_transition_timing.md")
PUBLIC = (11185, 18685)
MATCH_S = 3.0


def lags(m, codes, R, lo=0, hi=None):
    f, new = transitions(codes)
    keep = (f >= lo) & (f < (hi if hi is not None else len(codes)))
    return np.array([contact_lag(m, int(t), int(team), R=R) for t, team in zip(f[keep], new[keep])])


def offsets(ref, pred):
    fr, tr = transitions(ref)
    fp, tp = transitions(pred)
    out = []
    for t, team in zip(fr, tr):
        c = fp[tp == team]
        if not len(c):
            out.append(np.nan)
            continue
        j = np.argmin(np.abs(c - t))
        out.append((c[j] - t) / FPS if abs(c[j] - t) <= MATCH_S * FPS else np.nan)
    return np.array(out)


def describe(x):
    ok = x[~np.isnan(x)]
    if not len(ok):
        return "{} | — | — | — | —".format(len(x))
    q = np.percentile(ok, [25, 50, 75])
    return "{} | {:.0%} | {:+.2f} | {:+.2f} | {:+.2f}".format(len(x), len(ok) / len(x), *q)


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [from_metrica_sample(g) for g in (1, 2)]
    dfl = [from_dfl(i) for i in ids]
    rows = []
    for R in (1.0, 1.5):
        rows.append(("Metrica vekili", R, np.concatenate([lags(m, m["labels"], R) for m in met])))
        rows.append(("DFL ham (BallPossession)", R, np.concatenate([lags(m, _dfl_raw(i)["raw"], R) for m, i in zip(dfl, ids)])))
        rows.append(("kontrol kuralı, Metrica temiz", R, np.concatenate([lags(m, control(m, dead="truth"), R) for m in met])))
        rows.append(("kontrol kuralı, DFL temiz", R, np.concatenate([lags(m, control(m, dead="truth"), R) for m in dfl])))
        print(rows[-1][0], R, flush=True)
    # our match, v1 chain (classifier trained on all degraded external data, as in the P3 report)
    train = [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids] + \
            [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(train))
    v1 = from_tactical()
    g = v1["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(v1)[g])[:, 1]
    dead = (to_blocks(p) | np.isin(load_frames()["period"].values, NOT_PLAYED)) & g
    ours = np.where(g, control(v1, dead=dead), pm.X).astype(np.int8)
    for R in (1.0, 1.5):
        rows.append(("kontrol kuralı, bizim maç (GT'siz)", R, lags(v1, ours, R)))
        rows.append(("public GT (sadece betimleme)", R, lags(v1, v1["labels"], R, *PUBLIC)))
    lines = ["# P4b: el değişimi temasa göre ne zaman işaretleniyor?", "",
             "`analysis/p4b_transition_timing.py`. Gecikme = etiket değişimi − yeni takımın en yakın temas başlangıcı",
             "(oyuncu topa R m'den yakın), saniye; + etiket temastan sonra. ±2 s içinde temas yoksa ölçülemez.", "",
             "| etiketleme | R (m) | el değişimi | ölçülebilen | Ç1 | medyan | Ç3 |", "|---|---|---|---|---|---|---|"]
    for name, R, x in rows:
        lines.append("| {} | {} | {} |".format(name, R, describe(x)))
    lines += ["", "## Kural vekile göre önde mi geride mi?", "",
              "Her vekil el değişiminden aynı takıma giden en yakın kural el değişimine (±{} s) işaretli fark; + kural sonra.".format(MATCH_S), "",
              "| kaynak | vekil el değişimi | eşleşen | Ç1 | medyan | Ç3 |", "|---|---|---|---|---|---|"]
    for name, ms, ref in (("Metrica temiz", met, lambda m: m["labels"]), ("DFL temiz, ham etikete", dfl, None),
                          ("DFL temiz, birleştirilmiş etikete", dfl, lambda m: m["labels"])):
        x = []
        for k, m in enumerate(ms):
            r = ref(m) if ref is not None else _dfl_raw(ids[k])["raw"]
            x.append(offsets(r, control(m, dead="truth")))
        lines.append("| {} | {} |".format(name, describe(np.concatenate(x))))
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
