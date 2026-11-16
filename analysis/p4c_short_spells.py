"""P4 Task 4: how many short possessions are there, and do we make too many or too few?

    uv run python analysis/p4c_short_spells.py     # -> eda_out/reports/P4c_short_spells.md

Spell durations (possession_metrics.possession_spells, per half) for: the Metrica proxy, DFL raw,
DFL merged (our target proxy), the control rule on clean and degraded external matches (dead ball
from the labels, so only the team call shapes the spells), and the control rule on our match (v1
chain, GT-free). The public window is given as a count only, for description.
Then, on our match, where the rule's short spells (< 2 s) come from: right after a ball gap, with
players of both teams near the ball, or neither.
"""

import glob
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, _dfl_raw, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import _team_distances, control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from p4_common import ours  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4c_short_spells.md")
PUBLIC = (11185, 18685)


def spells(codes):
    return [(a, b) for a, b, _ in pm.possession_spells(codes, pm.valid_runs(codes))]


def summary(code_list):
    d, graded = [], 0
    for c in code_list:
        d += [(b - a) / FPS for a, b in spells(c)]
        graded += np.sum(c != pm.X)
    d = np.array(d)
    per90 = len(d) / (graded / FPS / 60) * 90
    return "{:.0f} | {:.1f} | {:.1f} | {:.0%} | {:.0%} | {:.0%}".format(
        per90, np.median(d), np.mean(d), np.mean(d < 1), np.mean(d < 2), np.mean(d < 5))


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [from_metrica_sample(g) for g in (1, 2)]
    dfl = [from_dfl(i) for i in ids]
    met_d = [degrade(m, rates_for("metrica"), seed=SEED) for m in met]
    dfl_d = [degrade(m, rates_for(i), seed=SEED) for m, i in zip(dfl, ids)]
    dead, codes = ours()
    v1 = from_tactical()
    rows = [("Metrica vekili", [m["labels"] for m in met]),
            ("kontrol kuralı, Metrica temiz", [control(m, dead="truth") for m in met]),
            ("kontrol kuralı, Metrica bozulmuş", [control(m, dead="truth") for m in met_d]),
            ("DFL ham", [_dfl_raw(i)["raw"] for i in ids]),
            ("DFL birleştirilmiş (1.6 s, hedef vekil)", [m["labels"] for m in dfl]),
            ("kontrol kuralı, DFL temiz", [control(m, dead="truth") for m in dfl]),
            ("kontrol kuralı, DFL bozulmuş", [control(m, dead="truth") for m in dfl_d]),
            ("kontrol kuralı, bizim maç (GT'siz)", [codes])]
    lines = ["# P4c: kısa possession'lar", "", "`analysis/p4c_short_spells.py`. Dış veride ölü top etiketten (sadece takım kararı possession'ları şekillendiriyor).", "",
             "| etiketleme | 90 dk'da possession | medyan s | ortalama s | < 1 s | < 2 s | < 5 s |", "|---|---|---|---|---|---|---|"]
    for name, cl in rows:
        lines.append("| {} | {} |".format(name, summary([np.asarray(c, np.int8) for c in cl])))
        print(lines[-1], flush=True)
    pub = v1["labels"][PUBLIC[0]:PUBLIC[1]]
    n_pub = len(pm.possession_spells(pub, pm.valid_runs(pub)))
    lines += ["", "Public pencere (sadece betimleme): 5 dakikada {} possession, 90 dakikaya oranla ~{:.0f}.".format(n_pub, n_pub / 5 * 90)]
    # where do our short spells come from?
    dH, dA = _team_distances(v1)
    seen = ~np.isnan(v1["bx"])
    sh = [(a, b) for a, b in spells(codes) if (b - a) < 2 * FPS]
    after_gap = both = 0
    for a, b in sh:
        lo = max(0, a - FPS)
        if np.mean(~seen[lo:a]) > 0.5:
            after_gap += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if np.nanmin(np.maximum(dH[a:b], dA[a:b])) < 3.0:
                both += 1
    n = max(1, len(sh))
    lines += ["", "## Bizim maçta kuralın kısa (< 2 s) possession'ları nereden geliyor?", "",
              "| | possession | oran |", "|---|---|---|",
              "| hepsi | {} | %100 |".format(len(sh)),
              "| hemen öncesindeki 1 s'nin yarısından çoğunda top yok | {} | {:.0%} |".format(after_gap, after_gap / n),
              "| bir anda iki takımdan da oyuncu topa 3 m'den yakın | {} | {:.0%} |".format(both, both / n)]
    for line in lines[-6:]:
        print(line)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
