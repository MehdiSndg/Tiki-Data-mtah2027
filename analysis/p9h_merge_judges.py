"""How much should DFL's possession labels be merged? The team GRU + TCN (v11's recipe: DFL + Metrica, swapped, short
x2) trained with DFL labels merged at MERGES seconds, judged where the test labels do not move with the merge:
Metrica 1-3 (leave one game out) and ELASTIC's annotated touch / reception timeline on J03WMX, J03WN1, J03WPY (the
dead ball from DFL's raw status; leave one DFL match out). Team call only (dead ball from the test labels), flat switch,
fresh seeds (3, 4, 5). P9g: the 1.6 s merge leaves 15% fewer switches than ELASTIC's receptions. GPU.

    python analysis/p9h_merge_judges.py <metrica|elastic> <elastic_repo>      # -> eda_out/reports/P9h_merge_<judge>.md
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
from p9g_elastic_labels import HALF_START, MATCHES, timeline  # noqa: E402

SEEDS = (3, 4, 5)
MERGES = (0.0, 0.8, 1.6)


def elastic_labels(repo, mid):
    d = _dfl_raw(mid)
    n1 = int((d["half"] == 1).sum())
    idx = lambda h, f: (f - HALF_START[1]) if h == 1 else n1 + 1 + (f - HALF_START[2])
    el = timeline(pd.read_parquet(os.path.join(repo, "benchmark", "gt", "{}.parquet".format(mid))), len(d["raw"]), idx)
    out = np.where(d["raw"] == pm.D, pm.D, el)
    return np.where(d["half"] > 0, out, pm.X).astype(np.int8)


def main(judge, repo=None):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    dfl = {mg: [degrade(from_dfl(i, merge_s=mg), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids] for mg in MERGES}
    # the degradation does not read the labels, so the positions are the same for every merge: features once
    T = [features(m) for m in dfl[MERGES[0]]] + [features(m) for m in met]
    Ts = [features(swap_teams(m)) for m in dfl[MERGES[0]]] + [features(swap_teams(m)) for m in met]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    if judge == "metrica":
        tests = [(len(ids) + j, met[j]["labels"], met[j]["name"]) for j in range(3)]
    else:
        tests = [(next(k for k, i in enumerate(ids) if i.endswith(mid)), elastic_labels(repo, mid), mid + "+elastic") for mid in MATCHES]
    rows = []
    for t, lab, name in tests:
        g = lab != pm.X
        alive = np.isin(lab, (pm.H, pm.A))
        r = {"match": name}
        for mg in MERGES:
            ms = dfl[mg] + met
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
            r["merge {:g}".format(mg)] = pm.spell_scores(lab, np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(lab))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["merge {:g}".format(mg) for mg in MERGES]
    lines = ["# P9h: DFL etiket birleştirmesi, {} hakemi (GRU + TCN, sadece takım, tohum 3–5)".format(judge), "", "`analysis/p9h_merge_judges.py {}`.".format(judge), "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{} − merge 1.6: {:+.3f}, {} / {} maçta önde.".format(c, (df[c] - df["merge 1.6"]).mean(), int((df[c] > df["merge 1.6"]).sum()), len(df)) for c in cols[:-1]]
    out = os.path.join(ROOT, "eda_out", "reports", "P9h_merge_{}.md".format(judge))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
