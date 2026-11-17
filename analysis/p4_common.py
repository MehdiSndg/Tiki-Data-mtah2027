"""Shared by the P4 scripts: our match through the v1 chain, cached (cache/p4_ours.npz).

    from p4_common import ours
    dead, codes = ours()          # classifier dead ball (+ P1 PRE2/POSTH), control rule labels

The dead-ball classifier is analysis/m1_dead_classifier.py trained on all degraded external data
(7 DFL + 2 Metrica), exactly as analysis/p3_report.py does it; no ground truth of ours is read.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "p4_ours.npz")


def dead_train_mask(m, xmask=False):
    """The frames a dead-ball model trains on: the halves, and with xmask not the X frames (Metrica's before a half's
    first event and after its last, otherwise counted as alive; P9ar)."""
    keep = np.asarray(m["half"]) > 0
    return keep & (np.asarray(m["labels"]) != pm.X) if xmask else keep


def dead_recipe_tag(metrica=(1, 2), xmask=False, nseeds=3):
    """The part of a dead-ball model's file name that tells v8's recipe (Metrica 1-3, X frames out, 6 seeds) from v4-v7's."""
    return ("" if tuple(metrica) == (1, 2) else "_m" + "".join(map(str, metrica))) + ("x" if xmask else "") + ("" if nseeds == 3 else "_s{}".format(nseeds))


REALISTIC_RATES = {"smooth_frames": 25}                      # P9dv: the realistic-dead copies' degradation before deadsim


def pff_dead_copies(pff, seed=0):
    """P9fq (16 Nov): PFF World Cup games as dead-ball training copies made to look like our ATD: PFF's seen-or-estimated
    ball (fills its short broadcast dropouts), one degraded copy each, no extra ball losses, the ball smoothed as ours;
    no carried / simulated dead ball (PFF's own stoppages are real). Labels: PFF's event-derived dead ball."""
    from mtah.degrade import degrade, rates_for
    from mtah.pff import from_pff
    for gid in pff:
        yield degrade(from_pff(gid, ball="all"), dict(rates_for("pff"), loss_per_min=0.0, smooth_frames=25), seed=seed)


def dead_classifier(degraded=True, featurize=None, extra_rates=None, draws=None, metrica=(1, 2), xmask=False, mix=False, pff=()):
    """draws: degradation seeds, every training match once per seed (default m1's SEED alone); metrica: the Metrica
    sample games trained on; xmask: X frames left out (dead_train_mask). v8 uses Metrica 1-3 and xmask (P9ar)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from m1_dead_classifier import SEED, table
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [from_dfl(i) for i in ids] + [from_metrica_sample(g) for g in metrica]
    if mix and degraded and featurize is not None:              # P9dw: + the realistic-dead copies; built one at a time, float32
        import pandas as pd
        from mtah.deadsim import realistic_dead
        keys = ids + ["metrica"] * len(metrica)
        Xs, ys = [], []
        for d in (draws or (SEED,)):
            for m, k in zip(train, keys):
                for c in (degrade(m, dict(rates_for(k), **(extra_rates or {})), seed=d),
                          realistic_dead(degrade(m, dict(rates_for(k), **REALISTIC_RATES), seed=d), m, seed=1000 + d)):
                    keep = dead_train_mask(c, xmask)
                    Xs.append(featurize(c)[keep].astype(np.float32))
                    ys.append(c["labels"][keep] == pm.D)
        for c in pff_dead_copies(pff):                          # P9fq; nothing when pff is empty
            keep = dead_train_mask(c, xmask)
            Xs.append(featurize(c)[keep].astype(np.float32))
            ys.append(c["labels"][keep] == pm.D)
        # memory (14 Nov: the build peaked at 9 GB inside fit): the raw matches go, and the float32 tables are moved one by
        # one into the float64 table sklearn would otherwise copy them into (same values: float32 -> float64 is exact)
        train.clear()
        del m, c
        cols = list(Xs[0].columns)
        X = np.empty((sum(len(x) for x in Xs), len(cols)))
        i = 0
        while Xs:
            x = Xs.pop(0).to_numpy()
            X[i:i + len(x)] = x
            i += len(x)
        y = np.concatenate(ys)
        ys.clear()
        X = pd.DataFrame(X, columns=cols, copy=False)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)
    if degraded:
        train = [degrade(m, dict(rates_for(k), **(extra_rates or {})), seed=d) for d in (draws or (SEED,))
                 for m, k in zip(train, ids + ["metrica"] * len(metrica))]
    if featurize is not None:
        import pandas as pd
        X = pd.concat([featurize(m)[dead_train_mask(m, xmask)] for m in train], ignore_index=True)
        y = np.concatenate([m["labels"][dead_train_mask(m, xmask)] == pm.D for m in train])
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(train))


def ours():
    if os.path.exists(CACHE):
        with np.load(CACHE) as d:
            return d["dead"], d["codes"]
    from m1_dead_classifier import features, to_blocks
    from mtah.frames import load_frames
    from mtah.labelers import control
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    clf = dead_classifier()
    v1 = from_tactical()
    g = v1["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(v1)[g])[:, 1]
    dead = (to_blocks(p) | np.isin(load_frames()["period"].values, NOT_PLAYED)) & g
    codes = np.where(g, control(v1, dead=dead), pm.X).astype(np.int8)
    np.savez_compressed(CACHE, dead=dead, codes=codes)
    return dead, codes


CACHE_V2 = os.path.join(ROOT, "cache", "p4_ours_v2.npz")


def dead_v2():
    """Our match's dead ball from the v2 features (mtah.deadball; +0.005 spellF1 on DFL LOMO, P6j),
    periods as ours(). Cached in cache/p4_ours_v2.npz."""
    if os.path.exists(CACHE_V2):
        with np.load(CACHE_V2) as d:
            return d["dead"]
    from m1_dead_classifier import to_blocks
    from mtah.deadball import features_v2
    from mtah.frames import load_frames
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    clf = dead_classifier(featurize=features_v2)
    m = from_tactical()
    g = m["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features_v2(m)[g])[:, 1]
    dead = (to_blocks(p) | np.isin(load_frames()["period"].values, NOT_PLAYED)) & g
    np.savez_compressed(CACHE_V2, dead=dead)
    return dead


def dead_v3():
    """dead_v2 plus the broadcast + stillness rule (mtah.deadball.broadcast_still_dead)."""
    from mtah.deadball import broadcast_still_dead
    from mtah.frames import load_frames
    return dead_v2() | broadcast_still_dead(load_frames(), os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))


CACHE_V4 = os.path.join(ROOT, "cache", "p4_ours_v4.npz")
CARRY_RATES = {"smooth_frames": 25, "dead_carry_p": 0.25}     # P6z's "new" training degradation


def dead_v4():
    """Dead ball v4: the v2 features' classifier trained on a degradation that keeps a carried dead ball
    (mtah.degrade.carry_dead_ball; P6z: DFL full chain 0.567 -> 0.590 on such data, Smart Tagging set
    pieces with a dead ball before them 83.3 -> 89.6% on our match), plus the broadcast + stillness rule."""
    if os.path.exists(CACHE_V4):
        with np.load(CACHE_V4) as d:
            return d["dead"]
    from m1_dead_classifier import to_blocks
    from mtah.deadball import broadcast_still_dead, features_v2
    from mtah.frames import load_frames
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    clf = dead_classifier(featurize=features_v2, extra_rates=CARRY_RATES)
    m = from_tactical()
    g = m["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features_v2(m)[g])[:, 1]
    fr = load_frames()
    dead = ((to_blocks(p) | np.isin(fr["period"].values, NOT_PLAYED)) & g) | broadcast_still_dead(
        fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    np.savez_compressed(CACHE_V4, dead=dead)
    return dead



CACHE_V5 = os.path.join(ROOT, "cache", "p4_ours_v5.npz")


def dead_gru_path(draws=None, metrica=(1, 2), xmask=False, nseeds=3, mix=False, pff=()):
    """models/dead_gru<recipe>_<hash>.pt; the hash covers G's model sources and CARRY_RATES, and with mix also the
    realistic copies' recipe (mtah/deadsim.py and REALISTIC_RATES), which G._hash does not."""
    from mtah import grumodel as G
    from m1_dead_classifier import SEED
    draws = draws or (SEED,)
    extra = repr(sorted(CARRY_RATES.items()))
    if mix:
        extra += open(os.path.join(ROOT, "mtah", "deadsim.py")).read() + repr(sorted(REALISTIC_RATES.items()))
    from mtah.members_c import pff_tag
    return os.path.join(ROOT, "models", "dead_gru{}{}{}{}_{}.pt".format("" if draws == (SEED,) else "_d" + "".join(map(str, draws)), dead_recipe_tag(metrica, xmask, nseeds),
                                                                     "_mix" if mix else "", pff_tag(pff), G._hash(extra)))


def dead_gru(dev=None, draws=None, metrica=(1, 2), xmask=False, nseeds=3, mix=False, pff=()):
    """A GRU on the dead-ball v2 features (mtah.grumodel's net and recipe, 3 seeds) trained on the same data
    as dead_v4's classifier (7 DFL + Metrica 1-2, carried dead ball); cached in models/."""
    import torch
    from mtah import grumodel as G
    from mtah.deadball import features_v2
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from m1_dead_classifier import SEED
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    draws = draws or (SEED,)
    path = dead_gru_path(draws, metrica, xmask, nseeds, mix, pff)
    if os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        nets = []
        for sd_ in st["nets"]:
            n = G._net(len(st["mu"])).to(dev)
            n.load_state_dict(sd_)
            nets.append(n.eval())
        return G.GRUEnsemble(nets, st["mu"], st["sd"], dev)
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [from_dfl(i) for i in ids] + [from_metrica_sample(g) for g in metrica]
    if mix:                                                      # P9dx: + the realistic-dead copies; one copy at a time
        from mtah.deadsim import realistic_dead
        Fs, meta = [], []
        for d in draws:
            for m, k in zip(train, ids + ["metrica"] * len(metrica)):
                for c in (degrade(m, dict(rates_for(k), **CARRY_RATES), seed=d),
                          realistic_dead(degrade(m, dict(rates_for(k), **REALISTIC_RATES), seed=d), m, seed=1000 + d)):
                    Fs.append(features_v2(c).values.astype(np.float32))
                    meta.append(((c["labels"] == pm.D).astype(np.float32), dead_train_mask(c, xmask).astype(np.float32), c["half"] > 0))
        mu, sd = G.feature_stats(Fs)                              # no concatenated copy (review 13 Nov)
        model = G.GRUEnsemble([], mu, sd, dev)
        data = [(model.norm(F),) + mt for F, mt in zip(Fs, meta)]
        del Fs
    else:
        train = [degrade(m, dict(rates_for(k), **CARRY_RATES), seed=d) for d in draws for m, k in zip(train, ids + ["metrica"] * len(metrica))]
        train += list(pff_dead_copies(pff))                     # P9fq; nothing when pff is empty
        Fs = [features_v2(m).values.astype(np.float32) for m in train]
        allF = np.concatenate(Fs)
        mu, sd = np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6
        model = G.GRUEnsemble([], mu, sd, dev)
        data = [(model.norm(F), (m["labels"] == pm.D).astype(np.float32), dead_train_mask(m, xmask).astype(np.float32), m["half"] > 0) for F, m in zip(Fs, train)]
    model.nets = [G.train(data, s, dev) for s in (G.SEEDS if nseeds == 3 else tuple(range(nseeds)))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"nets": [n.state_dict() for n in model.nets], "mu": mu, "sd": sd}, path)
    return model


def dead_v5(draws=None, cache=None, metrica=(1, 2), xmask=False, nseeds=3):
    """Dead ball v5: dead_v4 with the classifier's probability averaged with dead_gru's (P7c: full chain
    0.666 -> 0.674 on the carried-dead-ball DFL, 6 of 7), then blocks, the not-played periods and the
    broadcast + stillness rule as v4. draws: degradation seeds for both models' training (dead_v6)."""
    cache = cache or CACHE_V5
    if os.path.exists(cache):
        with np.load(cache) as d:
            return d["dead"]
    from m1_dead_classifier import to_blocks
    from mtah import grumodel as G
    from mtah.deadball import broadcast_still_dead, features_v2
    from mtah.frames import load_frames
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    clf = dead_classifier(featurize=features_v2, extra_rates=CARRY_RATES, draws=draws, metrica=metrica, xmask=xmask)
    m = from_tactical()
    g = m["half"] > 0
    F = features_v2(m)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(F[g])[:, 1]
    gm = dead_gru(draws=draws, metrica=metrica, xmask=xmask, nseeds=nseeds)
    q = np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0)
    fr = load_frames()
    dead = ((to_blocks((p + np.where(g, q, 0)) / 2) | np.isin(fr["period"].values, NOT_PLAYED)) & g) | broadcast_still_dead(
        fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    np.savez_compressed(cache, dead=dead)
    return dead


CACHE_V6 = os.path.join(ROOT, "cache", "p4_ours_v6.npz")
DRAWS_V6 = (0, 1, 2)


def dead_v6():
    """Dead ball v6: dead_v5 with both models trained on three degradation draws of every match (P7h)."""
    return dead_v5(draws=DRAWS_V6, cache=CACHE_V6)


def close_short_gaps(dead, g, min_frames=None):
    """Alive stretches shorter than min_frames (to_blocks' MIN_BLOCK) between two dead frames of the same half become
    dead. to_blocks closes them before the not-played periods and the broadcast rule are joined in; afterwards slivers
    can be left, and each one is a false possession spell (research report, dead-ball audit)."""
    from m1_dead_classifier import MIN_BLOCK
    n = min_frames or MIN_BLOCK
    out = dead.copy()
    e = np.diff(np.r_[0, (g & ~dead).astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < n and a > 0 and z < len(dead) and dead[a - 1] and dead[z] and g[a - 1] and g[z]:
            out[a:z] = True
    return out


def dead_v7():
    """Dead ball v7: dead_v6 with the short alive slivers left by the union closed (close_short_gaps)."""
    from mtah.frames import load_frames
    g = load_frames()["half"].values > 0
    return close_short_gaps(dead_v6(), g)


CACHE_V8 = os.path.join(ROOT, "cache", "p4_ours_v8.npz")


def dead_v8(nseeds=6):
    """Dead ball v8 (v17): dead_v7 with both models trained on Metrica 1-3 (as every LOMO measurement was) and without
    the X frames (P9ar: +0.001 on DFL and Metrica), and the dead-ball GRU averaged over 6 seeds (P9ag: score neutral; less
    build-to-build variance, which put a spurious 3 s block into the public window of the Kaggle v14 build).
    nseeds=3: the submitted v17 file (Kaggle build of 17d9d60, before the 6-seed change)."""
    from mtah.frames import load_frames
    g = load_frames()["half"].values > 0
    return close_short_gaps(dead_v5(draws=DRAWS_V6, cache=CACHE_V8, metrica=(1, 2, 3), xmask=True, nseeds=nseeds), g)

