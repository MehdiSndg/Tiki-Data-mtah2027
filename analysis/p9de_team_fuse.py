"""P9de: the teammates' deliveries (team/PROTOCOL.md; cache/team_in/<name>/external) on our ruler and in our chain.
DFL 7 + Metrica 3, every prediction out of fold, each model on its own degraded copy of the match (so the fusion gain
here is an upper bound: on our match all models read the same tracking). Labels: ours at 1.6 s and 3 s.
Arms: ours (v19c's chain), each teammate alone (their decision where given; their probabilities through our decoder),
and fusions: the teammate's P(home | alive) as extra weight in the team log-odds mean, and their P(dead) as a third
dead-ball model.

    python analysis/p9de_team_fuse.py
"""
import glob
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
T = lambda *p: os.path.join(ROOT, "cache", "team_in", *p)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
sg = lambda x: 1 / (1 + np.exp(-x))


def theirs(who, name, g, sub=""):
    """Their arrays on our frame grid (our grid = theirs with one separator frame between the halves), or None."""
    key = "dfl_" + name[-6:] if "DFL" in name else "metrica_" + name[-1]
    path = T(who, "external", sub, key + ".npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if len(z["p_home"]) != g.sum():
        print("  length mismatch", who, name, len(z["p_home"]), int(g.sum()))
        return None
    out = {}
    for k in ("p_home", "p_away", "p_dead", "pred", "labels"):
        if k in z.files:
            a = np.zeros(len(g), np.float64 if k.startswith("p_") else np.int8)
            if not k.startswith("p_"):
                a[:] = pm.X
            a[g] = z[k]
            out[k] = a
    out["team"] = out["p_home"] / np.maximum(out["p_home"] + out["p_away"], 1e-6)
    return out


def main():
    rows, names = {}, []
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        l3 = merged(lab, 3.0)
        sw = HI * (1 - z["b"]) + LO * z["b"]
        p = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        dq = (z["qh"] + z["qg"]) / 2
        sc = lambda c: np.array([pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        chain = lambda team, dead: sc(np.where(g, decode(team, g & ~dead, sw), pm.X).astype(np.int8))
        d0 = to_blocks(dq) & g
        r = {"ours": chain(p, d0)}
        M, E, B = theirs("mehmet_ali_topkara", name, g), theirs("enes", name, g), theirs("mehmet_ali_topkara", name, g, "cand_mix_pff")
        for tag, X in (("mali", M), ("enes", E), ("maliB", B)):
            if X is None:
                continue
            if "pred" in X:
                r[tag + " own decision"] = sc(np.where(g, X["pred"], pm.X).astype(np.int8))
                r[tag + " label agreement with ours"] = np.array([np.mean(X["labels"][g] == lab[g])] * 2)
            dx = to_blocks(X["p_dead"]) & g
            r[tag + " through our decoder"] = chain(X["team"], dx)
            r[tag + " team, our dead"] = chain(X["team"], d0)
            for w in (0.25, 0.5):
                r["ours + {} team w {}".format(tag, w)] = chain(sg((1 - w) * lg(p) + w * lg(X["team"])), d0)
            r["ours + {} dead".format(tag)] = chain(p, to_blocks((z["qh"] + z["qg"] + X["p_dead"]) / 3) & g)
            r["ours + {} team 0.5 + dead".format(tag)] = chain(sg(0.5 * lg(p) + 0.5 * lg(X["team"])), to_blocks((z["qh"] + z["qg"] + X["p_dead"]) / 3) & g)
        if M is not None and E is not None:
            r["ours + mali + enes team thirds + both dead"] = chain(sg((lg(p) + lg(M["team"]) + lg(E["team"])) / 3), to_blocks((z["qh"] + z["qg"] + M["p_dead"] + E["p_dead"]) / 4) & g)
        for k, v in r.items():
            rows.setdefault(k, {})[name] = v
        names.append(name)
        print(name, "done", flush=True)
    base = rows["ours"]
    print("\narm | DFL 1.6 s / 3 s (n) | gain vs ours on the same matches (ahead) | Metrica 1.6 s / 3 s (n) | gain (ahead)")
    for k, v in rows.items():
        out = [k]
        for src in ("DFL", "metrica"):
            ms = [m for m in v if src in m]
            if not ms:
                out.append("-")
                continue
            a, b = np.array([v[m] for m in ms]), np.array([base[m] for m in ms])
            out.append("{:.4f} / {:.4f} ({}) | {:+.4f} / {:+.4f} ({} / {})".format(*a.mean(0), len(ms), *(a - b).mean(0), int(((a - b)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
