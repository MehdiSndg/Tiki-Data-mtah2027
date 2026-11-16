"""Train on coarser targets? P9k: the four official full-match baselines are reproduced when DFL's labels are merged at
~3 s (not our 1.6 s), and P9l: against such labels a higher switch cost wins (+0.019). The team GRU + TCN (v11 recipe:
DFL + Metrica, swapped, short x2) trained with DFL (and, in one variant, Metrica) labels merged at 1.6 or 3.0 s, tested
on DFL leave one match out against labels merged at JUDGE seconds (and on Metrica 1-3 merged the same way), decoded
with each flat switch cost in COSTS; the cost is chosen nested (best on the other test matches). Team call only (dead
ball from the test labels), fresh seeds (3, 4, 5). GPU.

    python analysis/p9m_coarse_targets.py <dfl|metrica> [only=<match>,...]      # -> eda_out/reports/P9m_coarse_targets_<judge>.md
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
from mtah.external import DFL, _dfl_raw, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

SEEDS = (3, 4, 5)
JUDGE = 3.0
VARIANTS = {"train 1.6": (1.6, 0.0), "train 3.0": (3.0, 0.0), "train 3.0 + Metrica 3.0": (3.0, 3.0)}
COSTS = (8.0, 12.0, 20.0, 30.0)
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]


def main(judge, *rest):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met0 = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    dfl0 = [degrade(from_dfl(i, merge_s=0.0), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    # the degradation does not read H / A, so the positions are the same for every merge: features once
    T = [features(m) for m in dfl0] + [features(m) for m in met0]
    Ts = [features(swap_teams(m)) for m in dfl0] + [features(swap_teams(m)) for m in met0]
    relabel = lambda ms, s: [dict(m, labels=merged(m["labels"], s)) if s > 0 else m for m in ms]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    allm = dfl0 + met0
    tests = [(k, merged(allm[k]["labels"], JUDGE), allm[k]["name"]) for k in (range(len(ids), len(allm)) if judge == "metrica" else range(len(ids)))]
    rows = []
    for t, lab, name in tests:
        if ONLY and not any(o in name for o in ONLY):
            continue
        g = lab != pm.X
        alive = np.isin(lab, (pm.H, pm.A))
        r = {"match": name}
        for v, (mg, mm_) in VARIANTS.items():
            ms = relabel(dfl0, mg) + relabel(met0, mm_)
            tr = []
            for k in range(len(ms)):
                if k == t:
                    continue
                m, sw = ms[k], swap_teams(ms[k])
                sh = G.short_mask(m["labels"])
                for mm, F in ((m, T[k]), (sw, Ts[k])):
                    tr.append((F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            hp = ms[t]["half"] > 0
            p = (fit_predict(tr, (T[t], hp), seeds=SEEDS) + fit_predict(tr, (T[t], hp), make_net=G._tcn, seeds=SEEDS)) / 2
            for c in COSTS:
                r["{} c{:g}".format(v, c)] = pm.spell_scores(lab, np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(lab))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "match"]
    lines = ["# P9m: kaba hedefle eğitim, {} hakemi birleştirme {} s (GRU + TCN, sadece takım, tohum 3–5)".format(judge, JUDGE), "",
             "`analysis/p9m_coarse_targets.py {}`.".format(judge), "", "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    for v in VARIANTS:
        sc = []
        for _, r in df.iterrows():
            others = df[df.match != r.match]
            best = max(COSTS, key=lambda c: others["{} c{:g}".format(v, c)].mean()) if len(others) else 12.0
            sc.append(r["{} c{:g}".format(v, best)])
        df[v + " nested"] = sc
        lines.append("{}: iç içe ceza ortalaması {:.3f}".format(v, np.mean(sc)))
    base = df["train 1.6 nested"]
    for v in list(VARIANTS)[1:]:
        lines.append("{} − train 1.6 (iç içe): {:+.3f}, {} / {}".format(v, (df[v + " nested"] - base).mean(), int((df[v + " nested"] > base).sum()), len(df)))
    out = os.path.join(ROOT, "eda_out", "reports", "P9m_coarse_targets_{}{}.md".format(judge, "_" + "_".join(ONLY) if ONLY else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main(*sys.argv[1:])
