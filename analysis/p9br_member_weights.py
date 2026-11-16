"""P9br: nested member weights on v19's out-of-fold chain. v19 averages its six team members (single- and two-head BigGRU,
wide TCN, player set) with equal weights in probability. Here the three architectures get weights from {0.5, 1, 2} and the
mix is taken either in probability or in log-odds (50 settings, equal weights in probability = v19). Each of the 10 matches
(DFL 7 + Metrica 3, external data only) is scored with the setting that is best on the other nine (mean of the 1.6 s and
3 s referees); the single adoption rule on those nested scores, written down before the run (9 Nov): DFL and Metrica not
behind at either referee, DFL ahead at 1.6 s. Exit 0 when it holds. CPU.

    python analysis/p9br_member_weights.py      # -> eda_out/reports/P9br_member_weights.md
"""

import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

C = lambda *p: os.path.join(ROOT, "cache", *p)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9br_member_weights.md")
EPS = 1e-4


def settings():
    """(wb, wt, ws, mode); v19 first; settings that only rescale another are dropped."""
    out, seen = [], set()
    for mode in ("prob", "logit"):
        for w in [(1, 1, 1)] + list(itertools.product((0.5, 1, 2), repeat=3)):
            key = (tuple(np.round(np.array(w) / max(w), 6)), mode)
            if key not in seen:
                seen.add(key)
                out.append(w + (mode,))
    return out


def mix(members, w, mode):
    """members: [(big, tcn, set)] for the single head and the two-head; w: the three architecture weights."""
    ws = np.array([w[0], w[1], w[2], w[0], w[1], w[2]], np.float64)
    P = np.stack([np.clip(np.asarray(m, np.float64), EPS, 1 - EPS) for m in members])
    if mode == "prob":
        return (ws[:, None] * P).sum(0) / ws.sum()
    L = (ws[:, None] * np.log(P / (1 - P))).sum(0) / ws.sum()
    return 1 / (1 + np.exp(-L))


def main():
    S = settings()
    names, score = [], []
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab = z["g"], z["labels"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        sw = HI * (1 - z["b"]) + LO * z["b"]
        members = [big["pg"], d3["pt"], z["ps"], big["mg"], t1["mt"], z["ms"]]
        r = []
        for *w, mode in S:
            c = np.where(g, decode(mix(members, w, mode), g & ~dead, sw), pm.X).astype(np.int8)
            r.append((pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]))
        names.append(n[4:-4])
        score.append(r)
        print(n, "done", flush=True)
    A = np.array(score)                         # (match, setting, referee)
    obj = A.mean(2)
    pick = [int(np.argmax(np.delete(obj, i, 0).mean(0))) for i in range(len(names))]
    nest = np.array([A[i, pick[i]] for i in range(len(names))])
    base = A[:, 0]
    lines = ["# P9br: v19 OOF zincirinde üye ağırlıkları, iç içe LOMO", "", "`analysis/p9br_member_weights.py`.", "",
             "| maç | v19 | v19 @3s | iç içe | iç içe @3s | seçilen |", "|---|---|---|---|---|---|"]
    lines += ["| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(n, *base[i], *nest[i], S[pick[i]]) for i, n in enumerate(names)]
    best = int(np.argmax(obj.mean(0)))
    lines += ["", "en iyi tek ayar (bilgi, iç içe değil): {} ortalama {:+.4f}".format(S[best], obj[:, best].mean() - obj[:, 0].mean()), ""]
    ok = len(names) == 10
    d = {}
    for src in ("DFL", "metrica"):
        k = np.array([src in n for n in names])
        dd = nest[k] - base[k]
        d[src] = dd.mean(0)
        lines.append("{}: iç içe − v19 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(src, dd[:, 0].mean(), int((dd[:, 0] > 0).sum()), len(dd), dd[:, 1].mean(), int((dd[:, 1] > 0).sum()), len(dd)))
    ok = ok and d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar: " + ("tuttu" if ok else "alınmadı"))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
