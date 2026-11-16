"""P9dk: P9ct's sequence stacker (BiGRU over the out-of-fold channels, nested leave-one-match-out, three seeds) with
Mali's team probability as a seventh channel and his dead probability among the side channels, decoded by the
semi-Markov decoder (mtah/semimarkov.py). Against v26's fixed fusion on the same matches (our six members' mean at
0.75 + Mali's team at 0.25 in log-odds; dead = mean of our two models and his). P9ct failed narrowly with our six
members alone (DFL +0.004 / +0.006, Metrica −0.000 / +0.000): they are copies of one recipe; this adds an independent
channel. Rule: DFL ahead at 1.6 s and not behind at 3 s, Metrica not behind at either. Local GPU.

    python analysis/p9dk_stacker_mali.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p9ct_seq_stacker as S  # noqa: E402
from mtah import semimarkov  # noqa: E402
from p9de_team_fuse import lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = []
    for m in S.load():
        X = theirs("mehmet_ali_topkara", m["name"], m["g"])
        if X is None:
            print("no Mali prediction for", m["name"])
            continue
        m["mali"] = X
        m["mem"] = np.concatenate([m["mem"], np.nan_to_num(lg(X["team"]))[:, None].astype(np.float32)], 1)
        m["other"] = np.concatenate([m["other"], X["p_dead"][:, None].astype(np.float32)], 1)
        M.append(m)
    rows = []
    for i, m in enumerate(M):
        ps = np.mean([S.fit_predict(M[:i] + M[i + 1:], m, s, dev) for s in S.SEEDS], axis=0).astype(np.float64)
        z, g, lab, X = m["z"], m["g"], m["lab"], m["mali"]
        b = np.nan_to_num(z["b"])
        dead = (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + X["p_dead"]) / 3
        l3 = merged(lab, 3.0)
        sc = lambda p: np.array([pm.spell_scores(Lb, np.where(g, semimarkov.decode_halves(p, dead, g, b), pm.X).astype(np.int8), pm.valid_runs(Lb))["mean_f1"] for Lb in (lab, l3)])
        fixed = sg(0.75 * lg(np.nan_to_num(m["p"], nan=0.5)) + 0.25 * lg(X["team"]))
        base, new, half = sc(fixed), sc(ps), sc(sg((lg(ps) + lg(fixed)) / 2))
        alive = g & np.isin(lab, (pm.H, pm.A))
        acc = lambda p: np.mean((p[alive] > 0.5) == (lab[alive] == pm.H))
        rows.append((m["name"], base, new - base, half - base))
        np.save(S.C("oof_stackmali_{}.npy".format(m["name"])), ps.astype(np.float32))
        print(m["name"], "fixed fusion {:.4f} {:.4f} stacker {:+.4f} {:+.4f} mean of both {:+.4f} {:+.4f} | frame accuracy {:.4f} -> {:.4f}".format(
            *base, *(new - base), *(half - base), acc(fixed), acc(ps)), flush=True)
    for name, col in (("stacker", 2), ("mean of stacker and fixed fusion", 3)):
        d = np.array([r[col] for r in rows if "DFL" in r[0]])
        q = np.array([r[col] for r in rows if "DFL" not in r[0]])
        ok = d.mean(0)[0] > 0 and d.mean(0)[1] >= 0 and (q.mean(0) >= 0).all()
        print("{}: DFL {:+.4f} {:+.4f} ({} / 7); Metrica {:+.4f} {:+.4f} ({} / 3) -> {}".format(name, *d.mean(0), int((d[:, 0] > 0).sum()), *q.mean(0), int((q[:, 0] > 0).sum()),
                                                                                    "HOLDS" if ok else "does not hold"))


if __name__ == "__main__":
    main()
