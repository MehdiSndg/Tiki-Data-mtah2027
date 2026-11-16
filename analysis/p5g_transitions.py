"""Open item 5: the scorer's transition report, on external data -- which kind of transition do we get wrong?

    uv run python analysis/p5g_transitions.py      # -> eda_out/reports/P5g_transitions.md

The scorer matches every label change one-to-one within +/- tol frames on its destination label
(possession_metrics.transition_scores, the same pairs evaluation/run_evaluation.py writes to
reports/<name>_transitions.csv). Here those pairs are split by kind:
  hand-over   H -> A or A -> H          (the team call)
  dead start  H/A -> D                  (the ball goes dead)
  restart     D -> H/A                  (play resumes; matched only if the team is right too)
For each kind on degraded external data: how many there are, recall, the median latency of the
matched ones, and the spurious ones, with the length of the spell a spurious change starts (a
flicker or a long wrong stretch), and for every true hand-over the distance to our nearest change
to the same team (is a "missed" one late, or absent?). Two chains: team call only (dead ball from the labels) and the
full chain (m1 classifier trained cross-dataset, as P4a). Our match is not used: its only labels
are the public window.
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
from m1_dead_classifier import SEED, table  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from p4a_error_budget import classifier_dead  # noqa: E402

FPS = 25
TOLS = (12, 25)
SEED = int(os.environ.get("MTAH_SEED", SEED))      # other degradation seeds: MTAH_SEED=1, 2 (analysis/seed_combine.py)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5g_transitions{}.md".format("" if SEED == 0 else "_seed{}".format(SEED)))


def kind(frm, to):
    if frm in (pm.H, pm.A) and to in (pm.H, pm.A):
        return "el değişimi"
    if to == pm.D:
        return "ölü top başlangıcı"
    if frm == pm.D:
        return "tekrar başlama"
    return "diğer"


def spell_len(codes, frame):
    seg = codes[frame:]
    other = np.flatnonzero(seg != codes[frame])
    return (other[0] if len(other) else len(seg)) / FPS


def rows_for(m, pred, chain):
    runs = pm.valid_runs(m["labels"])
    gt_t = pm.transitions(m["labels"], runs)
    pr_t = pm.transitions(pm.fill_abstentions(pred, runs), runs)
    out = []
    to = np.array([t[2] for t in pr_t])
    at = np.array([t[0] for t in pr_t])
    for g in gt_t:
        if kind(g[1], g[2]) == "el değişimi":
            same = at[to == g[2]]
            off = (same - g[0])[np.argmin(np.abs(same - g[0]))] / FPS if len(same) else np.inf
            out.append({"src": "Metrica" if m["name"].startswith("metrica") else "DFL", "chain": chain, "tol": 0,
                        "side": "offset", "off": off})
    for tol in TOLS:
        pairs, missed, spurious = pm.match_transitions(gt_t, pr_t, tol)
        src = "Metrica" if m["name"].startswith("metrica") else "DFL"
        base = {"src": src, "chain": chain, "tol": tol, "match": m["name"]}
        for g, p in pairs:
            out.append(dict(base, side="gt", kind=kind(g[1], g[2]), status="matched", lat=(p[0] - g[0]) / FPS))
        for g in missed:
            out.append(dict(base, side="gt", kind=kind(g[1], g[2]), status="missed"))
        for p in spurious:
            out.append(dict(base, side="pred", kind=kind(p[1], p[2]), status="spurious", dur=spell_len(pred, p[0])))
        for p in [p for _, p in pairs]:
            out.append(dict(base, side="pred", kind=kind(p[1], p[2]), status="matched"))
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = {"DFL": [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids],
           "Metrica": [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]}
    clf = {k: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(v))
           for k, v in src.items()}
    rows = []
    for name, ms in src.items():
        other = clf["Metrica" if name == "DFL" else "DFL"]
        for m in ms:
            rows += rows_for(m, control(m, dead="truth"), "takım kararı (ölü top etiketten)")
            rows += rows_for(m, control(m, dead=classifier_dead(other, m)), "tam zincir")
            print(m["name"], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P5g: puanlayıcının geçiş raporu dış veride: hangi tür geçişi kaçırıyoruz?", "",
             "`analysis/p5g_transitions.py`. Bozulmuş dış veri, kontrol kuralı. Eşleşme puanlayıcının kendi kuralıyla: "
             "±tol kare, varış etiketi aynı olmalı. Gecikme + ise biz geç kalıyoruz.", ""]
    for tol in TOLS:
        lines += ["## tol = {} kare ({:.1f} s)".format(tol, tol / FPS), "",
                  "| zincir | kaynak | tür | GT'de 90 dk başına | recall | gecikme medyan s (Ç1–Ç3) | bizde 90 dk başına | "
                  "sahte (precision dışı) | sahtelerin başlattığı aralık medyan s |", "|---|---|---|---|---|---|---|---|---|"]
        t = df[df.tol == tol]
        for (chain, s), g in t.groupby(["chain", "src"], sort=False):
            mins = {"DFL": 90 * len(ids), "Metrica": 90 * 2}[s] / 90
            for k in ("el değişimi", "ölü top başlangıcı", "tekrar başlama"):
                gt, pr = g[(g.side == "gt") & (g.kind == k)], g[(g.side == "pred") & (g.kind == k)]
                mt = gt[gt.status == "matched"]
                sp = pr[pr.status == "spurious"]
                lines.append("| {} | {} | {} | {:.0f} | {:.0%} | {:+.2f} ({:+.2f}…{:+.2f}) | {:.0f} | {:.0%} | {:.1f} |".format(
                    chain, s, k, len(gt) / mins, len(mt) / max(1, len(gt)), mt.lat.median(), mt.lat.quantile(0.25),
                    mt.lat.quantile(0.75), len(pr) / mins, len(sp) / max(1, len(pr)), sp.dur.median()))
        lines.append("")
    sp = df[(df.tol == 25) & (df.status == "spurious")]
    lines += ["## Sahte geçişlerin başlattığı aralığın uzunluğu (tol 25)", "",
              "| zincir | kaynak | < 1 s | 1–2 s | 2–5 s | ≥ 5 s |", "|---|---|---|---|---|---|"]
    for (chain, s), g in sp.groupby(["chain", "src"], sort=False):
        d = g.dur
        lines.append("| {} | {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(
            chain, s, np.mean(d < 1), np.mean((d >= 1) & (d < 2)), np.mean((d >= 2) & (d < 5)), np.mean(d >= 5)))
    off = df[df.side == "offset"]
    lines += ["", "## Her gerçek el değişimine en yakın, aynı takıma giden geçişimiz (işaretli s; + biz sonra)", "",
              "| zincir | kaynak | ≤ 0.5 s | 0.5–1 s | 1–3 s, biz geç | 1–3 s, biz erken | 3–10 s | > 10 s ya da yok |",
              "|---|---|---|---|---|---|---|---|"]
    for (chain, s), g in off.groupby(["chain", "src"], sort=False):
        o = g.off.values
        a = np.abs(o)
        lines.append("| {} | {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(
            chain, s, np.mean(a <= 0.5), np.mean((a > 0.5) & (a <= 1)), np.mean((o > 1) & (o <= 3)),
            np.mean((o < -1) & (o >= -3)), np.mean((a > 3) & (a <= 10)), np.mean(a > 10)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
