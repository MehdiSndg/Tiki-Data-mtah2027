"""P2 Task 5: the first comparison table of the evaluation harness.

    uv run python analysis/p2_report.py          # -> eda_out/reports/P2.md   (~5 min)

Labelers (mtah/labelers.py) x sources x {clean, degraded}:
  Metrica sample 1-2 (labels derived from Metrica's events), DFL 7 matches (merged < 1.6 s),
  both clean and degraded (mtah.degrade, per-source rates, seed 0), and our match's public window
  (correctness check only: nothing here is chosen on it).
The dead ball is given two ways: from the labels (the team call alone) and the labeler's own
tracking-only estimate. v0 (analysis/v0_pipeline.py) needs the video and broadcast signals, so it
only has a public-window figure.

Also the kickoff ball tolerance variant asked for on 22 Oct (4 m kept, 2 m tested): which kickoffs
each finds on our match, GT-free.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import mtah.periods as periods_mod  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import load_tactical  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import baseline_smooth, control, nearest  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "reports", "P2.md")
LABELERS = [
    ("baseline_smooth50", lambda m: baseline_smooth(m, 50)),
    ("nearest", nearest),
    ("control, ölü top etiketten", lambda m: control(m, dead="truth")),
    ("control, ölü top kendi tahmini", lambda m: control(m, dead="own")),
]


def kickoff_variant():
    m = load_tactical()
    out = {}
    for tol in (4.0, 2.0):
        saved = periods_mod.K2_BALL
        periods_mod.K2_BALL = tol
        try:
            code, kick = periods_mod.periods(m, asset_paths()["atd_meta"])
        finally:
            periods_mod.K2_BALL = saved
        out[tol] = (kick, code)
    changed = int(np.sum(out[4.0][1] != out[2.0][1]))
    return out, changed


def main():
    dfl_ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = {"Metrica": [from_metrica_sample(g) for g in (1, 2)], "DFL": [from_dfl(i) for i in dfl_ids]}
    sources = {}
    for name, ms in clean.items():
        sources[name + " temiz"] = ms
        sources[name + " bozulmuş"] = [degrade(m, rates_for(m["name"]), seed=0) for m in ms]
    public = from_tactical()

    lines = ["# P2: ilk karşılaştırma tablosu", "",
             "`analysis/p2_report.py` üretti. spellF1 (sıralama metriği) kaynaktaki maçların ortalaması ± maçlar arası",
             "standart sapma; parantez içinde tahmin edilen / gerçek possession sayısı oranı. Public pencere sadece",
             "doğruluk kontrolü: 17 possession, bootstrap 5–95% aralığıyla. Hiçbir parametre burada seçilmedi.", "",
             "| etiketleyici | " + " | ".join(sources) + " | public (kontrol) |",
             "|---|" + "---|" * (len(sources) + 1)]
    macro = list(lines[-2:])
    for name, fn in LABELERS:
        row, mrow = [], []
        for src, ms in sources.items():
            r = evaluate(fn, ms, bootstrap=0)
            row.append("{:.3f} ± {:.3f} ({:.2f})".format(r.spell_f1.mean(), r.spell_f1.std(ddof=0), r.n_pred.sum() / r.n_gt.sum()))
            mrow.append("{:.3f}".format(r.macro_f1.mean()))
        if "etiketten" in name:
            row.append("—")
            mrow.append("—")
        else:
            r = evaluate(fn, [public], bootstrap=1000)
            row.append("{:.3f} [{:.2f}–{:.2f}]".format(r.spell_f1[0], r.boot_low[0], r.boot_high[0]))
            mrow.append("{:.3f}".format(r.macro_f1[0]))
        lines.append("| {} | ".format(name) + " | ".join(row) + " |")
        macro.append("| {} | ".format(name) + " | ".join(mrow) + " |")
        print(name, "done")
    v0 = os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")
    if os.path.exists(v0):
        codes = pm.label_codes(pm.read_label_csv(v0))
        r = evaluate(lambda _: codes, [public], bootstrap=1000)
        lines.append("| v0 (video + yayın; sadece bizim maç) | " + " | ".join("—" for _ in sources) +
                     " | {:.3f} [{:.2f}–{:.2f}] |".format(r.spell_f1[0], r.boot_low[0], r.boot_high[0]))
    lines += ["", "## macroF1", ""] + macro
    lines += ["", "\"Ölü top etiketten\" satırı public pencerede verilmedi: orada etiket public GT olurdu.", ""]

    var, changed = kickoff_variant()
    lines += ["## Kickoff top toleransı: 4 m (kullanılan) ve 2 m (literatür)", "",
              "| tolerans | 1. yarı kickoff karesi | 2. yarı kickoff karesi |", "|---|---|---|"]
    for tol in (4.0, 2.0):
        k = var[tol][0]
        lines.append("| {:.0f} m | {} | {} |".format(tol, k.get(1), k.get(2)))
    lines += ["", "İki tolerans arasında dönem kodu değişen kare sayısı: {}. GT kullanılmadı.".format(changed)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
