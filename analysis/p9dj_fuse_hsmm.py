"""P9dj: fusions through the semi-Markov decoder (mtah/semimarkov.py, the submission's decoder since v25), DFL 7 +
Metrica 3, our labels at 1.6 s / 3 s. Base: our six members' mean and our two dead-ball models (v25's chain without
video). Arms add, in the probabilities the decoder reads: Mali's P(home | alive) at log-odds weight 0.25, Mali's
P(dead) as a third dead model, the PFF-trained member (cache/oof_pff, when present) at 0.25 / as a seventh member.
Every model reads its own degraded copy of the match (upper bound for the teammates' part).

    python analysis/p9dj_fuse_hsmm.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    rows = {}
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        l3 = merged(lab, 3.0)
        b = np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        qh, qg = np.nan_to_num(z["qh"]), np.nan_to_num(z["qg"])
        run = lambda team, dead: np.array([pm.spell_scores(L, np.where(g, semimarkov.decode_halves(team, dead, g, b), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        r = {"base (v25's decoder)": run(p, (qh + qg) / 2)}
        M = theirs("mehmet_ali_topkara", name, g)
        if M is not None:
            mt = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
            r["+ mali team 0.25"] = run(mt, (qh + qg) / 2)
            r["+ mali dead"] = run(p, (qh + qg + M["p_dead"]) / 3)
            r["+ mali team 0.25 + dead"] = run(mt, (qh + qg + M["p_dead"]) / 3)
        if os.path.exists(C("oof_pff", n)):
            pp = np.nan_to_num(np.load(C("oof_pff", n))["pp"].astype(np.float64), nan=0.5)
            r["+ member as 7th"] = run((6 * p + pp) / 7, (qh + qg) / 2)
            r["+ member 0.25"] = run(sg(0.75 * lg(p) + 0.25 * lg(pp)), (qh + qg) / 2)
            if M is not None:
                r["+ mali 0.2 + member 0.2 + mali dead"] = run(sg(0.6 * lg(p) + 0.2 * lg(M["team"]) + 0.2 * lg(pp)), (qh + qg + M["p_dead"]) / 3)
        for k, v in r.items():
            rows.setdefault(k, {})[name] = v
        print(name, "done", flush=True)
    base = rows["base (v25's decoder)"]
    print("\narm | DFL 1.6 s / 3 s | gain over the base (ahead) | Metrica 1.6 s / 3 s | gain (ahead)")
    for k, v in rows.items():
        out = [k]
        for src in ("DFL", "metrica"):
            ms = [m for m in v if src in m]
            a, bb = np.array([v[m] for m in ms]), np.array([base[m] for m in ms])
            out.append("{:.4f} / {:.4f} | {:+.4f} / {:+.4f} ({} / {})".format(*a.mean(0), *(a - bb).mean(0), int(((a - bb)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
