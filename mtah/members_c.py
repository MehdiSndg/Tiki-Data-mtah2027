"""The team members trained on recipe C (P9ef): the production recipe of mtah.grumodel.external_gru (same nets, features,
standardisation, targets, short-spell weight, team swap, seeds, draws), with two changes to the training copies:
  * Metrica's ball is unglued before the degradation (mtah.unglue; the sample ball sits on a player, our ATD's does not)
  * every match gets one more copy with a realistic dead ball (mtah.deadsim; seed 1, realistic_dead seed 1001). With
    draws >= 2 its live frames repeat draw 1's (deadsim leaves live frames alone), so draw 1's live frames count twice
    for the tcn128 member (draws=3); kept, because P9ef / P9ej judged exactly this recipe (review 13 Nov).
P9ef (one member, held-out folds): on the realistic judge DFL +0.0285 / +0.0222 (7/7), Metrica +0.0065 / +0.0052.
mtah/grumodel.py is not edited (its sources key every cached model); this module reuses its pieces.

    from mtah.members_c import external_gru_c
    model = external_gru_c(arch="tcn128", multitask=False, seeds=range(6), draws=3)   # cached in models/
    p = mtah.grumodel.predict(model, m)
"""

import glob
import os

import numpy as np

from mtah import grumodel as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNGLUE_SEED = 11
REAL_SEED = 1
RECIPE_FILES = tuple(os.path.join(ROOT, f) for f in ("mtah/deadsim.py", "mtah/unglue.py"))   # not in G.HASH_FILES


def recipe_key():
    """The part of the file key that G._hash does not cover: the realism sources and recipe C's seeds."""
    return "".join(open(f).read() for f in RECIPE_FILES) + "u{}r{}".format(UNGLUE_SEED, REAL_SEED)


ST_LABELLED = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")   # SoccerTrack v2 with ball status


def st_tag(st):
    """Model-file tag for the SoccerTrack matches added to the training copies (P9fg): '' for none."""
    import hashlib
    return "_st" + hashlib.md5(",".join(sorted(st)).encode()).hexdigest()[:6] if st else ""


def pff_tag(pff):
    """Model-file tag for the PFF World Cup matches added to the training copies (P9fo): '' for none."""
    import hashlib
    return "_pff" + hashlib.md5(",".join(map(str, sorted(pff))).encode()).hexdigest()[:6] if pff else ""


def training_matches_c(draws=1, exclude=None, smooth=None, st=(), pff=()):
    """Recipe C's training copies of the 7 DFL + 3 Metrica matches (exclude: a match name left out, for held-out folds;
    smooth: the ball smoothing in frames, default mtah.teammodel.SMOOTH; recipe D = 13, the ATD-like value, P9es;
    st: SoccerTrack v2 match ids added with their ball-status labels merged at 1.6 s, degraded with their own rates,
    P9fg: the labelled set whose possession statistics are closest to the organisers' public window;
    pff: PFF World Cup game ids, ball where PFF gives one (seen or estimated, which fills its short broadcast dropouts),
    labels merged at 1.6 s, one degraded copy each with no extra ball losses (broadcast already loses it), P9fo)."""
    from mtah.deadsim import realistic_dead
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.teammodel import SMOOTH
    smooth = SMOOTH if smooth is None else smooth
    from mtah.unglue import unglue
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(unglue(from_metrica_sample(g), seed=UNGLUE_SEED), "metrica") for g in (1, 2, 3)]
    if st:
        from mtah.external import from_soccertrack
        raws += [(from_soccertrack(mid, merge_s=1.6), "st" + mid) for mid in st]
    out = []
    for r, k in raws:
        if exclude and exclude in r["name"]:
            continue
        out += [degrade(r, dict(rates_for(k), smooth_frames=smooth), seed=d) for d in range(draws)]
        out.append(realistic_dead(degrade(r, dict(rates_for(k), smooth_frames=smooth), seed=REAL_SEED), r, seed=1000 + REAL_SEED))
    if pff:
        from mtah.pff import from_pff
        for gid in pff:
            out.append(degrade(from_pff(gid, merge_s=1.6, ball="all"), dict(rates_for("pff"), loss_per_min=0.0, smooth_frames=smooth), seed=0))
    return out


def cache_path(target, arch, multitask, seeds, draws, short_k=2.0, exclude=None, smooth=None, st=(), pff=()):
    """models/<target>_gruC<recipe tag>_<hash of the model sources, the realism sources and the recipe seeds>.pt"""
    from mtah.teammodel import SMOOTH
    tag = G.model_tag(True, True, short_k, arch, None, multitask, False, tuple(seeds), False, draws) + ("_ex" + exclude if exclude else "")
    tag += "" if smooth is None or smooth == SMOOTH else "_b{}".format(smooth)
    tag += st_tag(st) + pff_tag(pff)
    return os.path.join(ROOT, "models", "{}_gruC{}_{}.pt".format(target, tag, G._hash(recipe_key())))


def external_gru_c(dev=None, target="team", arch="gru", multitask=False, seeds=G.SEEDS, draws=1, short_k=2.0, exclude=None, cache=True, smooth=None,
                   st=(), pff=()):
    """external_gru(swap=True, metrica=True, short_k=2.0, arch, multitask, seeds, draws) on recipe C's copies."""
    import torch
    import possession_metrics as pm
    from mtah.match import swap_teams
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    make = (lambda nf: G.widen(G.NETS[arch], nf)) if multitask else G.NETS[arch]
    heads = 2 if multitask else 1
    path = cache_path(target, arch, multitask, seeds, draws, short_k, exclude, smooth, st, pff)
    if cache and os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        nets = []
        for sd_ in st["nets"]:
            n = make(len(st["mu"])).to(dev)
            n.load_state_dict(sd_)
            nets.append(n.eval())
        return G.GRUEnsemble(nets, st["mu"], st["sd"], dev, arch, heads)
    ms = training_matches_c(draws, exclude, smooth, st, pff)
    ms = ms + [swap_teams(m) for m in ms]
    y1 = (lambda m: (m["labels"] == pm.H).astype(np.float32)) if target == "team" else (lambda m: G.boundary_target(m["labels"]))
    alive = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    w1 = lambda m: alive(m) * (np.where(G.short_mask(m["labels"]), short_k, 1.0).astype(np.float32) if short_k != 1.0 else 1.0)
    if multitask:
        y = lambda m: np.stack([y1(m), G.boundary_target(m["labels"])], 1)
        w = lambda m: np.stack([w1(m), alive(m)], 1).astype(np.float32)
    else:
        y, w = y1, w1
    Fs, rest = [], []                             # 16 Nov, peak memory: each match copy dropped once featurised
    for i in range(len(ms)):
        m, ms[i] = ms[i], None
        Fs.append(G.model_features(arch, m))
        rest.append((y(m), w(m), m["half"] > 0))
        del m
    del ms
    mu, sd = G.feature_stats(Fs)
    model = G.GRUEnsemble([], mu, sd, dev, arch, heads)
    data = []
    for i in range(len(Fs)):                      # one raw feature array freed per normalised one
        F, Fs[i] = Fs[i], None
        data.append((model.norm(F),) + rest[i])
        del F
    del Fs, rest
    model.nets = [G.train(data, s, dev, make_net=make) for s in seeds]
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"nets": [n.state_dict() for n in model.nets], "mu": mu, "sd": sd}, path)
    return model
