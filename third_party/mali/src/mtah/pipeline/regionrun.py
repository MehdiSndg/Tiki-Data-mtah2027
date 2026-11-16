"""Region-fold submission (Gün 5, rapor 6 Q4; MASTER_PLAN Ek N.6, N.14): `run.py main` when the
config has validation.region_folds.

For every fold k the config `<config>_rf<k>` (validation.region_cv: [k, n]) is trained end to end
(booster + adapted net), holding out region k's weak anchors. With validation.region_seeds
[s0, s1, ...] every fold is trained once per seed set (all booster / net / masking seeds derive from
`seed`; the first set keeps the plain fold names, later sets get the suffix `_s<seed>`).

Stitching: region k <- fold k. With several seed sets the fold probabilities are averaged
(arithmetic mean), then sharpened by a temperature tau chosen so that the ensemble's mean entropy on
graded frames equals the members' mean entropy (rapor 6 Q4: averaging softens the probabilities and
the semi-Markov decoder then under-segments); boundary probabilities are pooled with the 80th
percentile instead of the mean. The stitched probabilities are decoded ONCE with the blend decoder
(incl. the contact gate) -> outputs/<config name>/submission.csv. No frame of the submission is
predicted by a model that trained on that frame's weak anchors.
"""

import copy
import hashlib
import json
import time

import numpy as np

from .. import config as config_mod
from .. import constants as C
from ..decoding import contact
from ..io import submission
from ..io.broadcast import bar_data
from ..paths import OUTPUTS, ensure
from ..validation import sanity
from ..validation.regions import region_chunks, region_of_frames
from . import adapt, m0


def fold_configs(config="main"):
    """[[(name, cfg) per fold] per seed set]."""
    cfg = config_mod.load(config)
    v = cfg["validation"]
    n = int(v["region_folds"])
    seeds = [int(s) for s in (v.get("region_seeds") or [cfg["seed"]])]
    sets = []
    for j, s in enumerate(seeds):
        row = []
        for k in range(n):
            fc = config_mod.load(f"{config}_rf{k}")
            assert list(fc["validation"].get("region_cv") or []) == [k, n], \
                f"configs/{config}_rf{k}.yaml needs validation.region_cv: [{k}, {n}]"
            if j > 0 or s != int(fc["seed"]):
                fc = copy.deepcopy(fc)
                fc["seed"] = s
                fc["name"] = f"{fc['name']}_s{s}"
            row.append((fc["name"], fc))
        sets.append(row)
    return sets


def _entropy(p):
    q = np.clip(p, 1e-9, 1.0)
    return -(q * np.log(q)).sum(1)


def temperature_match(p_ens, members, mask, lo=0.3, hi=1.0, iters=40):
    """tau in [lo, hi] (p ** (1 / tau), renormalised) so that mean entropy(p_ens) on mask equals the
    members' mean entropy (bisection; deterministic)."""
    target = float(np.mean([_entropy(m[mask]).mean() for m in members]))

    def sharpen(p, tau):
        q = np.clip(p, 1e-9, 1.0) ** (1.0 / tau)
        return q / q.sum(1, keepdims=True)

    if _entropy(p_ens[mask]).mean() <= target:
        return p_ens, 1.0
    a, b = lo, hi
    for _ in range(iters):
        mid = (a + b) / 2
        if _entropy(sharpen(p_ens[mask], mid)).mean() > target:
            b = mid
        else:
            a = mid
    tau = round((a + b) / 2, 6)
    return np.round(sharpen(p_ens, tau), 6), tau


def stitch(cfg, fold_sets, out_name):
    """fold_sets: [[fold name per region] per seed set] of finished runs -> outputs/<out_name>.
    Returns (labels, summary dict)."""
    region = region_of_frames(cfg, C.N_FRAMES)
    graded = np.zeros(C.N_FRAMES, bool)
    for a, b in C.HALVES.values():
        graded[a:b] = True
    members, bnds = [], []
    for names in fold_sets:
        p = np.load(OUTPUTS / names[0] / "probs.npy").copy()
        b = np.load(OUTPUTS / names[0] / "bar_bnd_adapted.npy").copy()
        for k in range(1, len(names)):
            m = region == k
            p[m] = np.load(OUTPUTS / names[k] / "probs.npy")[m]
            b[m] = np.load(OUTPUTS / names[k] / "bar_bnd_adapted.npy")[m]
        members.append(p)
        bnds.append(b)
    tau = 1.0
    if len(members) == 1:
        p, bnd = members[0], bnds[0]
    else:
        p, tau = temperature_match(np.mean(members, axis=0), members, graded)
        bnd = np.round(np.percentile(np.stack(bnds), 80, axis=0), 6)
    dcfg = adapt.blend_cfg(cfg)
    lab = m0.decode_match(p, dcfg, bnd, regime=contact.regime_for(bar_data(cfg), dcfg))
    out = ensure(OUTPUTS / out_name)
    np.save(out / "probs.npy", p)
    np.save(out / "bar_bnd_adapted.npy", bnd)
    sub = submission.write(out / "submission.csv", lab)
    reason = submission.validate(sub)
    sha = hashlib.sha256(sub.read_bytes()).hexdigest()
    st = sanity.stats(lab)
    summary = {"config": cfg["name"], "hand_labels": "none", "region_folds": fold_sets,
               "regions": region_chunks(cfg), "contact_gate": bool(contact.params(dcfg).get("gate")),
               "ensemble_temperature": tau, "boundary_pooling": "p80" if len(members) > 1 else "single",
               "official_format_check": reason or "OK", "sha256": sha, "dead_share": st["dead_share"],
               "n_possessions": st["n_possessions"], "home_share": st["home_share"],
               "note": "region-fold out-of-fold submission: every frame from the fold that held its region out"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return lab, summary


def train_fold(name, fc, bootstrap=0):
    m0.run(cfg=fc, bootstrap=bootstrap, verbose=False, compare_baselines=False)
    adapt.run(config=name, name=name, cfg=fc)


def run(config="main", bootstrap=0):
    t0 = time.time()
    cfg = config_mod.load(config)
    sets = fold_configs(config)
    for j, row in enumerate(sets):
        for k, (name, fc) in enumerate(row):
            print(f"[seed set {j + 1}/{len(sets)}, region fold {k}/{len(row)}] {name} ({time.time() - t0:.0f}s)", flush=True)
            train_fold(name, fc, bootstrap)
    lab, s = stitch(cfg, [[n for n, _ in row] for row in sets], cfg["name"])
    print(f"[main] region-fold submission -> outputs/{cfg['name']}/submission.csv  sha256 {s['sha256'][:16]}  "
          f"format {s['official_format_check']}  poss {s['n_possessions']} dead {s['dead_share']:.3f}  "
          f"tau {s['ensemble_temperature']}  ({time.time() - t0:.0f}s)", flush=True)
    return lab, s["sha256"]
