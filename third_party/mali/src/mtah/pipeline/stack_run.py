"""Run the calibration / stacking family on a finished adaptation run (autonomous loop).

run(run_dir, net) evaluates, with evalkit (DFL LOMO nested + BAR-ATH + public diagnostic):
  pool_raw_w*     log-pool of the uncalibrated probabilities (reference)
  pool_cal_w*     vector-scaled log-pool
  meta_gt35/gt11  meta-refiner (logistic regression) with Metrica-aligned / training labels
  *_em            + EM prior-shift per match
Family finalist = best DFL spellF1 (35-aligned) among candidates whose BAR full-match statistics
lie inside the reference bands (if none: best DFL overall, flagged).
"""

import json

import numpy as np

from .. import config as config_mod
from .. import constants as C
from ..paths import OUTPUTS, ensure
from ..validation.splits import cache_tag
from . import autolog, evalkit
from .stack import Stacker, nested_dfl


def items(cfg, run_dir, net, target, extra_cfgs=()):
    ctx = evalkit.contexts(cfg)
    z = np.load(OUTPUTS / run_dir / f"dfl_lomo_net_{net}.npz")
    out = {}
    for m, d in ctx["dfl"].items():
        pb = np.load(OUTPUTS / "decoder" / cfg["name"] / f"lomo_{m}{cache_tag(cfg)}.npy")
        y = d["gt35"] if target == "gt35" else np.where(d["prep"]["data"].period > 0, d["prep"]["aligned"], C.X)
        ex = [np.load(OUTPUTS / "decoder" / e / f"lomo_{m}{cache_tag(config_mod.load(e))}.npy") for e in extra_cfgs]
        out[m] = (pb, z[m], d["vis"], np.asarray(y, np.int8)) + ((ex,) if ex else ())
    return out


def run(run_dir, net="adapted", config="main", pool_ws=(0.25, 0.5, 0.75), extra_cfgs=(), tag=""):
    cfg = config_mod.load(config)
    ctx = evalkit.contexts(cfg)
    bar_pb = np.load(OUTPUTS / f"{cfg['name']}_lgbm" / "probs.npy")
    bar_pn = np.load(OUTPUTS / run_dir / f"bar_net_{net}.npy")
    bar_vis = ctx["bar"]["vis"]
    bar_ex = [np.load(OUTPUTS / f"{config_mod.load(e)['name']}_lgbm" / "probs.npy") for e in extra_cfgs]
    variants = []
    for w in (pool_ws if not extra_cfgs else ()):
        variants.append((f"pool_raw_w{w}", dict(mode="pool", w=w, calibrate=False), "gt11"))
        variants.append((f"pool_cal_w{w}", dict(mode="pool", w=w, calibrate=True), "gt11"))
    for tgt in ("gt35", "gt11"):
        variants.append((f"meta_{tgt}", dict(mode="meta"), tgt))
        variants.append((f"meta_{tgt}_em", dict(mode="meta", em=True), tgt))
    rows, probs_keep = [], {}
    item_cache = {}
    for name, kw, tgt in variants:
        if tgt not in item_cache:
            item_cache[tgt] = items(cfg, run_dir, net, tgt, extra_cfgs)
        it = item_cache[tgt]
        dfl = nested_dfl(it, lambda: Stacker(**kw))
        st = Stacker(**kw).fit(list(it.values()))
        bar = st.predict(bar_pb, bar_pn, bar_vis, bar_ex)
        r = evalkit.evaluate(cfg, bar, dfl)
        r.pop("labels")
        name = name + tag
        row = {"variant": name, "net": net, "run": run_dir, **r}
        rows.append(row)
        probs_keep[name] = (bar, dfl)
        autolog.log(f"C {run_dir}/{net} {name:<16} DFL gt35 {r['dfl_spell_gt35']:.4f} gt11 {r['dfl_spell_gt11']:.4f} "
                    f"macro {r['dfl_macro']:.4f} | BAR poss {r['bar_poss']} D {r['bar_dead']:.3f} home {r['bar_home']:.3f} "
                    f"bands {r['bar_in_bands']} | public {r['pub_spell']:.4f}/{r['pub_macro']:.4f}")
        autolog.record({"run": f"auto_C_{run_dir}_{net}_{name}", "family": "stack", "spell_f1": r["pub_spell"],
                        "macro_f1": r["pub_macro"], "sanity_n_possessions": r["bar_poss"],
                        "sanity_dead_share": r["bar_dead"], "sanity_home_share": r["bar_home"],
                        "dfl_spell_gt35": r["dfl_spell_gt35"], "dfl_spell_gt11": r["dfl_spell_gt11"]})
    ok = [r for r in rows if r["bar_in_bands"]]
    pool = ok or rows
    best = max(pool, key=lambda r: (r["dfl_spell_gt35"], r["dfl_macro"]))
    best["finalist_in_bands"] = bool(ok)
    folder = ensure(OUTPUTS / f"stack_{run_dir}_{net}{tag}")
    bar, dfl = probs_keep[best["variant"]]
    np.save(folder / "probs.npy", bar)
    np.savez_compressed(folder / "dfl_lomo_blend.npz", **dfl)
    (folder / "summary.json").write_text(json.dumps({"rows": rows, "finalist": best}, indent=1, default=float),
                                         encoding="utf-8")
    autolog.log(f"C finalist {run_dir}/{net}: {best['variant']} (bands ok: {bool(ok)}) DFL gt35 "
                f"{best['dfl_spell_gt35']:.4f} -> {folder}")
    return rows, best
