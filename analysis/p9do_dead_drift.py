"""P9do: a wandering dead ball in the dead-ball classifier's training copies (Mali's 11 Nov idea, metrica_src.drift_dead_ball).
Label-free fact on our match: the ATD ball outside the pitch moves at a median 2.76 m/s (p25 1.26, p75 4.81; 6% under
0.3 m/s), while our simulated copies' dead ball moves at 0.7-1.2 m/s with 24-36% nearly still: a classifier trained
on them may learn "still ball -> dead" and call our match's moving dead ball live.
Drift: inside every dead stretch of a degraded copy the visible ball gets a smooth offset path, a new velocity every
50 frames, speed drawn from our match's outside-the-pitch speeds, random direction, detrended to its own ends.
Arms (HistGB on dead-ball v2 features, held-out folds, three draws per match as v8): train plain / train plain + one
drifted draw per match. Judges, never trained on: the held-out match's plain copy and its drifted copy; spellF1 with
the labels' team and the predicted dead ball (1.0 = perfect dead ball). Label-free on our match: dead share and frame
agreement with the video head's dead calls (an independent model).

    python analysis/p9do_dead_drift.py [row step = 3]
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p4_common as P4  # noqa: E402
from m1_dead_classifier import SEED, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9dc_dead_real import feats  # noqa: E402


def bar_speeds():
    from mtah.match import from_tactical
    m = from_tactical()
    L, W = m["pitch"]
    g = m["half"] > 0
    x, y = m["bx"].astype(float) * L, m["by"].astype(float) * W
    oob = g & np.isfinite(x) & ((x < -0.5) | (x > L + 0.5) | (y < -0.5) | (y > W + 0.5))
    v = np.hypot(np.diff(x), np.diff(y)) * 25
    ok = oob[1:] & oob[:-1] & np.isfinite(v)
    return np.percentile(v[ok], np.arange(5, 100, 5))


def drift(m, speeds, seed, block=50):
    out = dict(m)
    L, W = m["pitch"]
    bx, by = m["bx"].astype(np.float64) * L, m["by"].astype(np.float64) * W
    rng = np.random.default_rng(seed)
    dead = (m["labels"] == pm.D) & (m["half"] > 0)
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        n = e - s
        if n < 2 * block:
            continue
        k = n // block + 2
        ang = rng.uniform(0, 2 * np.pi, k)
        v = rng.choice(speeds, k)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
        vt = np.stack([np.interp(np.arange(n), np.arange(k) * block, v[:, j]) for j in (0, 1)], 1)
        off = np.cumsum(vt, 0) / 25.0
        off -= np.linspace(0, 1, n)[:, None] * off[-1]
        bx[s:e] += off[:, 0]
        by[s:e] += off[:, 1]
    out["bx"], out["by"] = (bx / L).astype(np.float32), (by / W).astype(np.float32)
    return out


def main(step=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    sp = bar_speeds()
    print("our match's outside-the-pitch ball speed percentiles 5..95:", np.round(sp, 2), flush=True)
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    srcs = [(i, (lambda i=i: from_dfl(i)), i) for i in ids] + [("metrica{}".format(g), (lambda g=g: from_metrica_sample(g)), "metrica") for g in (1, 2, 3)]
    plain, drifted = {}, {}
    for name, raw, rk in srcs:
        for d in P4.DRAWS_V6:
            mk = lambda raw=raw, rk=rk, d=d: degrade(raw(), dict(rates_for(rk), **P4.CARRY_RATES), seed=d)
            plain[(name, d)] = feats("sim_{}_{}".format(name, d), mk)
        drifted[name] = feats("simdrift_{}".format(name), lambda raw=raw, rk=rk, name=name: drift(degrade(raw(), dict(rates_for(rk), **P4.CARRY_RATES), seed=7), sp, hash(name) % 1000))
        print("features", name, flush=True)
    rows = lambda F, lab: (F[np.flatnonzero(lab != pm.X)[::step]], lab[np.flatnonzero(lab != pm.X)[::step]] == pm.D)
    fit = lambda X, y: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)
    R = {}
    clf_all = {}
    for t, _, _ in srcs:
        tr = [n for n, _, _ in srcs if n != t]
        base = [rows(*plain[(n, d)][:2]) for n in tr for d in P4.DRAWS_V6]
        extra = [rows(*drifted[n][:2]) for n in tr]
        arms = {"plain": fit(np.concatenate([b[0] for b in base]), np.concatenate([b[1] for b in base])),
                "+ drifted draw": fit(np.concatenate([b[0] for b in base + extra]), np.concatenate([b[1] for b in base + extra]))}
        for judge, (F, lab, _) in (("plain copy", plain[(t, 0)]), ("drifted copy", drifted[t])):
            g = lab != pm.X
            for a, clf in arms.items():
                p = np.zeros(len(g))
                p[g] = clf.predict_proba(F[g])[:, 1]
                dead = to_blocks(p) & g
                f1 = pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"]
                R.setdefault(("DFL" if "metrica" not in t else "Metrica", judge, a), []).append(f1)
                print(t, judge, a, "spellF1 {:.4f}".format(f1), flush=True)
    for k, v in sorted(R.items()):
        print("MEAN", " | ".join(k), "{:.4f} (n {})".format(np.mean(v), len(v)))
    # label-free on our match: both recipes trained on everything
    from mtah.match import from_tactical
    m = from_tactical()
    g = m["half"] > 0
    F = features_v2(m).values.astype(np.float32)
    vq = np.load(os.path.join(ROOT, "cache", "barath_dense_avg5.npz"))["q_dead"].astype(np.float64)
    allb = [rows(*plain[(n, d)][:2]) for n, _, _ in srcs for d in P4.DRAWS_V6]
    alle = [rows(*drifted[n][:2]) for n, _, _ in srcs]
    for a, parts in (("plain", allb), ("+ drifted draw", allb + alle)):
        clf = fit(np.concatenate([b[0] for b in parts]), np.concatenate([b[1] for b in parts]))
        p = np.zeros(len(g))
        p[g] = clf.predict_proba(F[g])[:, 1]
        dead = to_blocks(p) & g
        vd = vq[g] > 0.5
        print("OUR MATCH", a, "dead share {:.3f}; agreement with the video head's dead calls {:.4f}; video-dead frames called dead {:.3f}, video-live called live {:.3f}".format(
            dead[g].mean(), np.mean(dead[g] == vd), dead[g][vd].mean(), 1 - dead[g][~vd].mean()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
