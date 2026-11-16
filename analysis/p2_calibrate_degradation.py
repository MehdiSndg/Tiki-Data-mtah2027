"""P2 Task 4: does degraded external data look like our ATD? GT-free.

    uv run python analysis/p2_calibrate_degradation.py      # -> eda_out/reports/P2_calibration.md

One statistics function is run on our raw ATD (graded frames) and on every degraded external
match, so the comparison is like for like:

  ball missing    share of half frames with no ball
  gap median/p90  seconds, runs of missing ball inside a half
  players         visible per team per frame, median and p5
  double count    share of frames with more than 10 outfield players in a team (p1_report's measure)

Two rates are chosen on a small grid: still_frames (how long the ball must lie still before the ATD
drops it: our ATD has few gaps under 2 s, a 10-frame rule made half of them shorter) and piece_drop,
by the acceptance criterion of the plan: ball missing within 5 points and gap median within 3 s of
our ATD, then the closest players p5. Labels are only used inside degrade (where the dead ball is),
never to score anything.
"""

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

# Params
FPS = 25
GRID = {"still_frames": (25, 50), "loss_per_min": (0.0, 0.35, 0.7, 1.0)}
PIECE_DROP = 0.1                  # first grid: players p5 within 0.5 of ours at 0.1 for every still_frames
DUP_GRID = (0.07, 0.2, 0.35, 0.5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P2_calibration.md")
KEYS = ("ball missing", "alive missing", "gap median s", "gap p90 s", "players median", "players p5", "double count")


def match_stats(m, alive=None):
    """alive: in-play frames -- the labels for external data; for ours the v0 rule estimate (GT-free)."""
    g = m["half"] > 0
    if alive is None:
        alive = np.isin(m["labels"], (pm.H, pm.A))
    miss = np.isnan(m["bx"]) & g
    e = np.diff(np.r_[0, miss.astype(np.int8), 0])
    gaps = (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / FPS
    pres = ~np.isnan(m["px"][g])
    counts, extra = [], []
    for t in (0, 1):
        cols = m["slot_team"] == t
        counts.append(pres[:, cols].sum(axis=1))
        out = cols.copy()
        if m["gk"][t] >= 0:
            out[m["gk"][t]] = False
        extra.append(pres[:, out].sum(axis=1) > 10)
    c = np.concatenate(counts)
    return {"ball missing": miss[g].mean(), "alive missing": miss[g & alive].mean(), "gap median s": np.median(gaps) if len(gaps) else 0.0,
            "gap p90 s": np.percentile(gaps, 90) if len(gaps) else 0.0,
            "players median": np.median(c), "players p5": np.percentile(c, 5),
            "double count": np.mean(np.concatenate(extra))}


def mean_stats(matches, over, seeds=(0,)):
    rows = [match_stats(degrade(m, dict(rates_for(m["name"]), **over), seed=s)) for m in matches for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS}


def fmt(s):
    return "| {:.1%} | {:.1%} | {:.1f} | {:.1f} | {:.0f} | {:.0f} | {:.1%} |".format(*(s[k] for k in KEYS))


def main():
    v0 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")).label.values
    ours = match_stats(from_tactical(), alive=np.isin(v0, ["H", "A"]))
    dfl_ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    groups = {"Metrica": [from_metrica_sample(g) for g in (1, 2)], "DFL": [from_dfl(i) for i in dfl_ids]}
    lines = ["# P2 kalibrasyon: bozulmuş dış veri bizim ATD'ye benziyor mu?", "",
             "`analysis/p2_calibrate_degradation.py` üretti. GT kullanılmadı. Kabul ölçütü (plan): top kayıp",
             "oranı ±5 puan, boşluk medyanı ±3 s. Bizim ATD'nin oyun içi kareleri v0 kurallarının tahmini (GT'siz; v0 ölü",
             "topu kısmen top kaybından bulduğu için bu oran aşağı yönlü yanlı olabilir).", "",
             "| kaynak | top kayıp | oyun içi top kayıp | boşluk medyan s | boşluk p90 s | oyuncu medyan | oyuncu p5 | çift sayım |",
             "|---|---|---|---|---|---|---|---|", "| **bizim ATD** " + fmt(ours)]
    for name, ms in groups.items():
        lines.append("| {} temiz ".format(name) + fmt({k: float(np.mean([match_stats(m)[k] for m in ms])) for k in KEYS}))
    lines += ["", "## Izgara (still_frames × loss_per_min; piece_drop 0.1), bozulmuş, seed 0", "",
              "| kaynak | still_frames | loss_per_min | top kayıp | oyun içi top kayıp | boşluk medyan s | boşluk p90 s | oyuncu medyan | oyuncu p5 | çift sayım | kabul |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dbd, pd_ in itertools.product(GRID["still_frames"], GRID["loss_per_min"]):
        rates = {"still_frames": dbd, "loss_per_min": pd_, "piece_drop": PIECE_DROP}
        worst = 0.0
        ok_all = True
        for name, ms in groups.items():
            s = mean_stats(ms, rates)
            ok = abs(s["ball missing"] - ours["ball missing"]) <= 0.05 and abs(s["gap median s"] - ours["gap median s"]) <= 3
            ok_all &= ok
            worst = max(worst, abs(s["players p5"] - ours["players p5"]))
            lines.append("| {} | {} | {} ".format(name, dbd, pd_) + fmt(s) + " {} |".format("✓" if ok else "✗"))
        print("still_frames {} loss_per_min {}: both accepted {}, worst players-p5 gap {:.1f}".format(dbd, pd_, ok_all, worst))
    lines += ["", "Tek oran iki kaynağa uymuyor: DFL'nin oyun içi top kaybını tutturan oran (1.0/dk) Metrica'yı",
              "aşıyor, çünkü Metrica'nın tracking'i oyun içinde topu zaten kaybediyor (temizde %5.1). Toplam top kaybı",
              "ise veri setinin kendi ölü top payına bağlı (DFL %44.8, Metrica %37.6): onu hiçbir oran düzeltmez.",
              "**Karar:** still_frames = 50, loss_per_min = 1.0 (DFL), Metrica için 0 (`mtah.degrade.SOURCE_RATES`);",
              "kabul ölçütü olarak toplam yerine oyun içi top kaybı (±3 puan) kullanılır.", "",
              "## Çift sayım (dup_share), kaynak oranlarıyla", "",
              "| kaynak | dup_share | top kayıp | oyun içi top kayıp | boşluk medyan s | boşluk p90 s | oyuncu medyan | oyuncu p5 | çift sayım |",
              "|---|---|---|---|---|---|---|---|---|"]
    gap = {}
    for d in DUP_GRID:
        gap[d] = 0.0
        for name, ms in groups.items():
            st = mean_stats(ms, {"dup_share": d})
            gap[d] = max(gap[d], abs(st["double count"] - ours["double count"]))
            lines.append("| {} | {} ".format(name, d) + fmt(st))
        print("dup_share {}: worst double-count gap {:.1%}".format(d, gap[d]))
    best = min(gap, key=gap.get)
    lines += ["", "**Seçim:** dup_share = {} (çift sayım farkı en çok {:.1%}).".format(best, gap[best])]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("dup_share", best, "->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
