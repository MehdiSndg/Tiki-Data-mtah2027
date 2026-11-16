"""P4: what do the hand-overs the control rule misses (and invents) look like? External data only.

    uv run python analysis/p4e_missed_handovers.py    # -> eda_out/reports/P4e_missed_handovers.md

Clean Metrica sample (proxy labels) and clean DFL (merged labels), dead ball from the labels, so
only the team call is judged. A proxy hand-over is matched when the rule hands over to the same team
within 3 s; a rule hand-over is spurious when no proxy hand-over to its team lies within 3 s.
For the missed ones: how long the new possession lasts in the proxy, how close the new team ever gets
to the ball around it (+/- 1 s), and whether the ball was in the air-like fast phase (> 15 m/s).
"""

import glob
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.timing import transitions  # noqa: E402

FPS = 25
TOL = 3 * FPS
OUT = os.path.join(ROOT, "eda_out", "reports", "P4e_missed_handovers.md")


def team_dist(m, team):
    L, W = m["pitch"]
    d = np.hypot((m["px"].astype(float) - m["bx"][:, None]) * L, (m["py"].astype(float) - m["by"][:, None]) * W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.where(m["slot_team"] == team, d, np.nan), axis=1)


def spell_len(codes, t):
    seg = codes[t:]
    stop = np.flatnonzero(seg != codes[t])
    return (stop[0] if len(stop) else len(seg)) / FPS


def study(ms):
    rows, spur, n_rule = [], 0, 0
    for m in ms:
        ref, pred = m["labels"], control(m, dead="truth")
        fr, tr = transitions(ref)
        fp, tp = transitions(pred)
        dist = {0: team_dist(m, 0), 1: team_dist(m, 1)}
        L, W = m["pitch"]
        bs = np.r_[np.nan, np.hypot(np.diff(m["bx"]) * L, np.diff(m["by"]) * W) * FPS]
        for t, team in zip(fr, tr):
            c = fp[tp == team]
            hit = len(c) and np.min(np.abs(c - t)) <= TOL
            lo, hi = max(0, t - FPS), t + FPS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows.append((hit, spell_len(ref, t), np.nanmin(dist[team][lo:hi]),
                             np.nanmin(dist[team][lo:hi] - dist[1 - team][lo:hi]), np.nanmax(bs[max(0, t - FPS):t])))
        for t, team in zip(fp, tp):
            n_rule += 1
            c = fr[tr == team]
            spur += not (len(c) and np.min(np.abs(c - t)) <= TOL)
    return np.array(rows, float), spur, n_rule


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    lines = ["# P4e: kuralın kaçırdığı ve uydurduğu el değişimleri", "",
             "`analysis/p4e_missed_handovers.py`. Temiz dış veri, ölü top etiketten (sadece takım kararı). Eşleşme: aynı takıma ±3 s.", ""]
    for name, ms in (("Metrica temiz (vekil)", [from_metrica_sample(g) for g in (1, 2)]),
                     ("DFL temiz (birleştirilmiş)", [from_dfl(i) for i in ids])):
        r, spur, n_rule = study(ms)
        hit, miss = r[r[:, 0] == 1], r[r[:, 0] == 0]
        lines += ["## " + name, "",
                  "- vekil el değişimi {}, kural yakaladı {:.0%}; kural el değişimi {}, uydurma (vekilde karşılığı yok) {:.0%}".format(
                      len(r), len(hit) / len(r), n_rule, spur / max(1, n_rule)), "",
                  "| | yakalanan | kaçırılan |", "|---|---|---|"]
        for label, k, fmt in (("yeni possession'ın vekildeki süresi, medyan s", 1, "{:.1f}"),
                              ("yeni possession < 2 s", 1, "<2"),
                              ("yeni takımın topa en yakın mesafesi (±1 s), medyan m", 2, "{:.1f}"),
                              ("yeni takım hiç 1.5 m'ye girmiyor", 2, ">1.5"),
                              ("yeni takım hiç rakipten yakın değil", 3, ">0"),
                              ("öncesindeki 1 s'de top > 15 m/s (uzun pas, şut, sekme)", 4, ">15")):
            vals = []
            for grp in (hit, miss):
                x = grp[:, k]
                x = x[~np.isnan(x)]
                if fmt == "<2":
                    vals.append("{:.0%}".format(np.mean(x < 2)))
                elif fmt.startswith(">"):
                    vals.append("{:.0%}".format(np.mean(x > float(fmt[1:]))))
                else:
                    vals.append(fmt.format(np.median(x)))
            lines.append("| {} | {} | {} |".format(label, *vals))
        lines.append("")
        print("\n".join(lines[-12:]), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
