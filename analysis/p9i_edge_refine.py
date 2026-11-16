"""Dead-ball block edges refined by a learned picker (P9b: edges cost DFL +0.012, the largest dead-ball term). For each
edge of a predicted block (v11's dead ball on P9a's out-of-fold probabilities), every frame within +-WIN is a
candidate; a HistGB trained on the other matches (leave one match out) scores whether the true edge lies within TOL
of it, from the candidate's offset, the dead-ball probability and its slope there, and the dead-ball v2 features; the
edge moves to the best candidate when its score passes a threshold (nested). The test matches are rebuilt exactly as
in P9a (carried-dead-ball degradation, seed 0). Team decode = v14's six-net mean, production switch cost. CPU.

    python analysis/p9i_edge_refine.py <oof_dir>      # -> eda_out/reports/P9i_edge_refine.md
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
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO, blocks, load  # noqa: E402

WIN, TOL = 75, 6                  # +-3 s of candidates; a candidate is right within 0.24 s of the true edge
THRESHOLDS = (0.0, 0.3, 0.5, 0.7)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9i_edge_refine.md")


def rebuild(name):
    key = name.split("+")[0]
    if key.startswith("metrica"):
        return degrade(from_metrica_sample(int(key[-1])), dict(rates_for("metrica"), smooth_frames=25, dead_carry_p=0.25), seed=0)
    return degrade(from_dfl(key), dict(rates_for(key), smooth_frames=25, dead_carry_p=0.25), seed=0)


def candidates(z, F, pred):
    """Rows (one per candidate frame of each predicted edge) with features, the edge it belongs to and a target."""
    q = (z["qh"] + z["qg"]) / 2
    dq = np.r_[0, np.diff(pd.Series(q).rolling(13, center=True, min_periods=1).mean().values)]
    truth = (z["labels"] == pm.D) & z["g"]
    t_starts = np.array([a for a, _ in blocks(truth)])
    t_ends = np.array([b for _, b in blocks(truth)])
    rows, n = [], len(q)
    Fv = F.values
    for bi, (a, b) in enumerate(blocks(pred)):
        for kind, e, tr in (("start", a, t_starts), ("end", b, t_ends)):
            lo, hi = max(1, e - WIN), min(n - 1, e + WIN)
            near = tr[np.abs(tr - e) <= WIN] if len(tr) else np.array([])
            for c in range(lo, hi + 1):
                rows.append([bi, kind == "start", e, c, c - e, q[c], dq[c] * (1 if kind == "start" else -1),
                             int(len(near) > 0 and np.min(np.abs(near - c)) <= TOL)] + list(Fv[c]))
    cols = ["block", "is_start", "edge", "cand", "off", "q", "dq", "y"] + ["f_" + c for c in F.columns]
    return pd.DataFrame(rows, columns=cols)


def apply(pred, C, score, thr):
    out = pred.copy()
    C = C.assign(s=score)
    for (bi, st), grp in C.groupby(["block", "is_start"]):
        best = grp.loc[grp.s.idxmax()]
        if best.s < thr:
            continue
        e, c = int(best.edge), int(best.cand)
        if st:
            out[min(e, c):max(e, c)] = c < e        # earlier start extends the block, later start shrinks it
        else:
            out[min(e, c):max(e, c)] = c > e
    return out


def main(d):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Z = load(d)
    names = sorted(Z)
    C = {}
    for nme in names:
        z = Z[nme]
        m = rebuild(nme)
        assert (m["labels"] == z["labels"]).all(), nme
        pred = to_blocks((z["qh"] + z["qg"]) / 2) & z["g"]
        C[nme] = (candidates(z, features_v2(m), pred), pred)
        print(nme, len(C[nme][0]), "candidates", flush=True)
    feat = [c for c in C[names[0]][0].columns if c not in ("block", "edge", "cand", "y")]
    rows = []
    for nme in names:
        tr = pd.concat([C[k][0] for k in names if k != nme], ignore_index=True)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[feat], tr.y)
        Ct, pred = C[nme]
        s = clf.predict_proba(Ct[feat])[:, 1]
        z = Z[nme]
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        cost = HI * (1 - z["b"]) + LO * z["b"]
        sc = lambda dead: pm.spell_scores(z["labels"], np.where(z["g"], decode(p, z["g"] & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(z["labels"]))["mean_f1"]
        r = {"match": nme, "v14": sc(pred)}
        for thr in THRESHOLDS:
            r["thr {}".format(thr)] = sc(apply(pred, Ct, s, thr))
        r["truth"] = sc((z["labels"] == pm.D) & z["g"])
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v14"] + ["thr {}".format(t) for t in THRESHOLDS] + ["truth"]
    lines = ["# P9i: ölü top blok kenarlarını öğrenen seçici (OOF, tam zincir, takım v14)", "", "`analysis/p9i_edge_refine.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        lines.append("| ort. {} | ".format(src) + " | ".join("{:.3f}".format(q[c].mean()) for c in cols) + " |")
    lines.append("")
    dfl = df[df.match.str.startswith("DFL")]
    nest = []
    for _, r in dfl.iterrows():
        best = max(THRESHOLDS, key=lambda t: dfl[dfl.match != r.match]["thr {}".format(t)].mean())
        nest.append(r["thr {}".format(best)] - r.v14)
    lines.append("İç içe eşik (DFL): v14'e fark {:+.3f}, {} / {} maçta önde.".format(np.mean(nest), int(np.sum(np.array(nest) > 0)), len(nest)))
    for t in THRESHOLDS:
        for src in ("DFL", "metrica"):
            q = df[df.match.str.startswith(src)]
            lines.append("eşik {} {}: {:+.3f}, {} / {}".format(t, src, (q["thr {}".format(t)] - q.v14).mean(), int((q["thr {}".format(t)] > q.v14).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
