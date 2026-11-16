"""P4 Task 2: where does spellF1 go? GT-free (external data only; the public window plays no part).

    uv run python analysis/p4a_error_budget.py        # -> eda_out/reports/P4a_error_budget.md

Part 1, the budget. On every Metrica sample and DFL match, clean and degraded, three labelings:
  team only   control rule, dead ball from the labels: only the team call can be wrong
  dead only   the labels' own team, dead ball from the classifier: only the dead ball can be wrong
              (frames the classifier calls alive inside a true D take the last team before it)
  both        control rule + classifier dead ball: the full chain
The dead-ball classifier is the m1 one, trained cross-dataset (DFL -> Metrica, Metrica -> DFL), on
degraded data for the degraded matches and clean data for the clean ones.

Part 2, the price of each error type. The labels themselves are perturbed and scored against the
unperturbed labels: a constant delay on every hand-over, a delay on every dead-ball edge, random
jitter on hand-overs, short spells removed, short spurious spells inserted.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED, features, table, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.timing import transitions  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4a_error_budget.md")


def spell_f1(truth, pred):
    return pm.spell_scores(truth, pred, pm.valid_runs(truth))["mean_f1"]


def team_with_dead(labels, dead, half):
    """The labels' team on frames called alive (the last team where the labels say D), D elsewhere."""
    own = np.where(np.isin(labels, (pm.H, pm.A)), labels, -1).astype(float)
    own[own < 0] = np.nan
    out = np.full(len(labels), pm.X, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        team = pd.Series(own[idx]).ffill().bfill().values
        out[idx] = np.where(dead[idx], pm.D, np.nan_to_num(team, nan=pm.D)).astype(np.int8)
    return out


def classifier_dead(clf, m):
    g = m["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(m)[g])[:, 1]
    return to_blocks(p) & g


# --- part 2: perturb the labels ---------------------------------------------------------------

def delay_handovers(c, k):
    out = c.copy()
    f, _ = transitions(c)
    for t in f:
        old = c[t - 1]
        stop = t + k
        seg = c[t:stop]
        stop = t + (np.flatnonzero(seg != c[t])[0] if np.any(seg != c[t]) else len(seg))
        out[t:stop] = old
    return out


def delay_dead_edges(c, k):
    """Every D block starts and ends k frames late (the previous label runs on, D runs on)."""
    out = c.copy()
    d = (c == pm.D).astype(np.int8)
    e = np.diff(np.r_[0, d, 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if a > 0 and c[a - 1] in (pm.H, pm.A):
            out[a:min(a + k, z)] = c[a - 1]
        if z < len(c) and c[z] in (pm.H, pm.A):
            nxt = z + np.argmax(c[z:] != c[z]) if np.any(c[z:] != c[z]) else len(c)
            out[z:min(z + k, nxt)] = pm.D
    return out


def jitter_handovers(c, sd, rng):
    out = c.copy()
    f, _ = transitions(c)
    for t in f:
        k = int(round(rng.normal(0, sd)))
        if k > 0:
            seg = c[t:t + k]
            stop = t + (np.flatnonzero(seg != c[t])[0] if np.any(seg != c[t]) else len(seg))
            out[t:stop] = c[t - 1]
        elif k < 0:
            seg = c[max(0, t + k):t][::-1]
            back = np.flatnonzero(seg != c[t - 1])[0] if np.any(seg != c[t - 1]) else len(seg)
            out[t - back:t] = c[t]
    return out


def drop_short(c, max_s):
    out = c.copy()
    for a, b, code in pm.segments(c, pm.valid_runs(c)):
        if code in (pm.H, pm.A) and (b - a) < max_s * FPS and a > 0 and out[a - 1] in (pm.H, pm.A) and out[a - 1] != code:
            out[a:b] = out[a - 1]
    return out


def insert_spurious(c, per_min, length_s, rng):
    out = c.copy()
    alive = np.flatnonzero(np.isin(c, (pm.H, pm.A)))
    k = int(round(per_min * len(alive) / FPS / 60))
    L = int(round(length_s * FPS))
    for s in rng.choice(alive, k, replace=False):
        seg = c[s:s + L]
        if np.all(seg == c[s]):
            out[s:s + L] = pm.A if c[s] == pm.H else pm.H
    return out


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = {"Metrica": [from_metrica_sample(g) for g in (1, 2)], "DFL": [from_dfl(i) for i in ids]}
    deg = {"Metrica": [degrade(m, rates_for("metrica"), seed=SEED) for m in clean["Metrica"]],
           "DFL": [degrade(m, rates_for(i), seed=SEED) for m, i in zip(clean["DFL"], ids)]}
    lines = ["# P4a: hata bütçesi", "", "`analysis/p4a_error_budget.py`. Sadece dış veri; public pencere kullanılmadı.",
             "Hedef vekil: Metrica'da olay verisinden türettiğimiz etiket, DFL'de DFL'in kendi etiketi (1.6 s birleştirme).", "",
             "## 1. Bütçe: spellF1, maçların ortalaması", "",
             "| kaynak | girdi | sadece takım hatası | sadece ölü top hatası | ikisi birden (tam zincir) |", "|---|---|---|---|---|"]
    for kind, src in (("temiz", clean), ("bozulmuş", deg)):
        clfs = {}
        for train, test in (("DFL", "Metrica"), ("Metrica", "DFL")):
            clfs[test] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(src[train]))
        for name, ms in src.items():
            r = []
            for m in ms:
                dead = classifier_dead(clfs[name], m)
                r.append((spell_f1(m["labels"], control(m, dead="truth")),
                          spell_f1(m["labels"], team_with_dead(m["labels"], dead, m["half"])),
                          spell_f1(m["labels"], np.where(m["half"] > 0, control(m, dead=dead), pm.X).astype(np.int8))))
            r = np.array(r)
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(name, kind, *r.mean(axis=0)))
            print(lines[-1], flush=True)
    lines += ["", "## 2. Her hata türünün bedeli: etiketi bozup bozulmamışa karşı spellF1", "",
              "Temiz etiketler; Metrica 2 maç + DFL 7 maç ortalaması (rastgele olanlar 3 tohum).", "",
              "| bozma | Metrica | DFL |", "|---|---|---|"]
    rng = lambda s: np.random.default_rng(s)  # noqa: E731
    cases = [("her el değişimi 2 kare (0.08 s) geç", lambda c, s: delay_handovers(c, 2)),
             ("her el değişimi 5 kare (0.2 s) geç", lambda c, s: delay_handovers(c, 5)),
             ("her el değişimi 10 kare (0.4 s) geç", lambda c, s: delay_handovers(c, 10)),
             ("her el değişimi 25 kare (1 s) geç", lambda c, s: delay_handovers(c, 25)),
             ("el değişimleri rastgele ±0.4 s (sd 10 kare)", lambda c, s: jitter_handovers(c, 10, rng(s))),
             ("el değişimleri rastgele ±1 s (sd 25 kare)", lambda c, s: jitter_handovers(c, 25, rng(s))),
             ("ölü top kenarları 10 kare (0.4 s) geç", lambda c, s: delay_dead_edges(c, 10)),
             ("ölü top kenarları 25 kare (1 s) geç", lambda c, s: delay_dead_edges(c, 25)),
             ("ölü top kenarları 50 kare (2 s) geç", lambda c, s: delay_dead_edges(c, 50)),
             ("1 s'den kısa possession'lar yok", lambda c, s: drop_short(c, 1.0)),
             ("2 s'den kısa possession'lar yok", lambda c, s: drop_short(c, 2.0)),
             ("dakikada 0.5 sahte 0.6 s'lik possession", lambda c, s: insert_spurious(c, 0.5, 0.6, rng(s))),
             ("dakikada 1 sahte 0.6 s'lik possession", lambda c, s: insert_spurious(c, 1.0, 0.6, rng(s)))]
    for label, fn in cases:
        res = []
        for name in ("Metrica", "DFL"):
            res.append(np.mean([spell_f1(m["labels"], fn(m["labels"], s)) for m in clean[name] for s in range(3)]))
        lines.append("| {} | {:.3f} | {:.3f} |".format(label, *res))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
