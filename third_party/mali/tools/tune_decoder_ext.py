r"""External-data-only decoder tuning (organisers' Nov 2026 clarification: parameters MUST be tuned on external data).

Fixed probabilities of a finished `adapt` run; nothing is retrained. Coordinate descent over the decoder
parameters on a TUNING set of external matches, then a report on a disjoint VALIDATION set:
  tune      : DFL core matches (joint.matches), both corruptions (v1 LOMO + v2b eval LOMO)
  validation: the booster-unseen DFL matches (v1 + v2b) and the Metrica open games (never trained on)
Objective = 0.45 * v2b + 0.35 * Metrica-like + 0.20 * v1 analogue on the tuning set: here 0.6 * v2b + 0.4 * v1
(Metrica stays a pure validation referee). The provided game is decoded ONLY for a label-free report;
its statistics never enter the objective or the choice. No ground truth of the provided game is read.

  .venv\Scripts\python.exe tools/tune_decoder_ext.py --run main_ext --out outputs/decoder_ext_main_ext.json
"""

import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.io.broadcast import bar_data  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.labels.masking import gap_lengths  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine, m0, metrica_src  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

# (path in cfg["decoder"], candidate values); the current value is always added.
GRID = [
    (("emission_scale",), [1.25, 1.5, 1.75, 2.0, 2.5]),
    (("penalty_turnover",), [8.0, 11.0, 14.0, 18.0]),
    (("boundary_gamma",), [0.0, 1.0, 2.0, 3.0]),
    (("duration", "possession", "median"), [150, 200, 250, 320]),
    (("duration", "possession", "sigma"), [0.6, 0.85, 1.1]),
    (("duration", "dead", "median"), [250, 350, 450]),
    (("emission_weights", 2), [1.0, 1.15, 1.3]),
    (("penalty_stoppage",), [6.0, 10.0, 14.0]),
    (("penalty_restart",), [3.0, 6.0, 9.0]),
    (("contact", "free_penalty"), [18.0, 24.0, 30.0]),
    (("contact", "contact_penalty"), [6.0, 8.5, 11.0]),
]


def get(d, path):
    for k in path:
        d = d[k]
    return d


def put(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {}) if isinstance(k, str) else d[k]
    if isinstance(path[-1], int):
        d[path[-1]] = value
    else:
        d[path[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="finished adapt run under outputs/ (all 7 DFL folds saved)")
    ap.add_argument("--config", help="config name (default: the run's summary config)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-alpha", action="store_true",
                    help="5 Nov: also tune alpha_duration (weight of the duration prior; never tuned on external data)")
    ap.add_argument("--tune-blend", action="store_true",
                    help="also tune the booster share of the log-pool blend (w_boost) on the same tuning set")
    args = ap.parse_args()
    t0 = time.time()
    run = OUTPUTS / args.run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(args.config or summary["config"])
    ac, _ = adapt.params(cfg)
    base = adapt.blend_cfg(cfg, ac)
    base["decoder"].setdefault("contact", {})
    base["decoder"]["emission_weights"] = list(base["decoder"]["emission_weights"])
    for path, _ in GRID:                       # make every key explicit (contact defaults)
        if path[0] == "contact":
            base["decoder"]["contact"].setdefault(path[1], contact.params(base)[path[1]])
    lengths = gap_lengths(bar_data(cfg))
    ecfg = dfl_referee.eval_cfg(cfg)
    core = list(cfg["joint"]["matches"])
    w0 = float(ac["w_boost"])
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    v1, v1b = np.load(run / "dfl_lomo_blend.npz"), np.load(run / "dfl_lomo_bnd.npz")
    v2, v2b = np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz")
    if (run / "metrica_eval_blend.npz").is_file():
        met, metb = np.load(run / "metrica_eval_blend.npz"), np.load(run / "metrica_eval_bnd.npz")
    else:                                      # Metrica trained on: its leave-one-game-out folds are in the LOMO files
        met, metb = None, None
    missing = [m for m in core if m not in v1.files or m not in v2.files]
    if missing:
        raise SystemExit(f"run lacks LOMO probabilities of core matches {missing}; run adapt without eval_matches")
    refs = []                                  # (name, group, split, probs, bnd, runs, gt, corrupted)
    for kind, blend, bnd, c in (("v1", v1, v1b, cfg), ("v2b", v2, v2b, ecfg)):
        for m in blend.files:
            if m.startswith("metrica_"):
                continue
            prep = dfl_referee.prepare(m, c, lengths)
            gt = np.where(prep["data"].period > 0,
                          suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                          C.X).astype(np.int8)
            refs.append((m, kind, "tune" if m in core else "val", blend[m], bnd[m], prep["runs"], gt, prep["corrupted"]))
    met_src = (met, metb) if met is not None else (v1, v1b)
    for m in [f for f in met_src[0].files if f.startswith("metrica_")]:
        prep = metrica_src.prepare(int(m.split("_")[1]), cfg)
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        refs.append((m, "metrica", "val", met_src[0][m], met_src[1][m], prep["runs"], gt, prep["data"]))
    bar_probs, bar_bnd = np.load(run / "probs.npy"), np.load(run / "bar_bnd_adapted.npy")
    pools = {}                                 # name/kind -> (booster probs, net probs) for re-blending
    if args.tune_blend:
        extra = list(ac["extra_matches"])
        boost = {"v1": {**decoder_tune.lomo_probs(cfg, folder), **decoder_tune.unseen_probs(cfg, extra, folder)},
                 "v2b": decoder_tune.eval_probs(cfg, folder, tuple(extra))}
        for m, kind, *_ in refs:
            if kind == "metrica":
                boost.setdefault("metrica", {})[m] = decoder_tune.unseen_probs(
                    cfg, [m], folder, preps=[metrica_src.prepare(int(m.split("_")[1]), cfg)])[m]
        for i, (m, kind, sp, pr, *_r) in enumerate(refs):
            pb = boost[kind][m]
            # the net's probabilities are recovered exactly (up to the 6-digit rounding) from the saved blend
            lp = (np.log(np.clip(pr, 1e-6, 1)) - w0 * np.log(np.clip(pb, 1e-6, 1))) / (1 - w0)
            lp -= lp.max(1, keepdims=True)
            pn = np.exp(lp)
            pools[i] = (pb, pn / pn.sum(1, keepdims=True))
        bar_boost = np.load(OUTPUTS / f"{ac.get('booster_source') or cfg['name']}_lgbm" / "probs.npy")
        bar_net = np.load(run / "bar_net_adapted.npy")
        assert np.allclose(log_pool(bar_boost, bar_net, w0), bar_probs, atol=1e-6), "BAR blend does not reproduce"
        base["w_boost"] = w0

    def probs_for(i, pr, c):
        w = c.get("w_boost", w0)
        return pr if (i not in pools or w == w0) else log_pool(*pools[i], w)
    bar = bar_data(cfg)
    print(f"refs: {[(r[0], r[1], r[2]) for r in refs]} ({time.time() - t0:.0f}s)", flush=True)

    def score(c, split):
        out = {}
        for i, (m, kind, sp, pr, bd, runs, gt, corr) in enumerate(refs):
            if sp != split:
                continue
            pred = engine.decode_runs(probs_for(i, pr, c), runs, c, boundary=bd, regime=contact.regime_for(corr, c))
            out.setdefault(kind, {})[m] = pm.evaluate(gt, pred)["spell_f1_mean"]
        return {k: float(np.mean(list(v.values()))) for k, v in out.items()}, out

    def objective(c):
        s, _ = score(c, "tune")
        return 0.6 * s["v2b"] + 0.4 * s["v1"]

    def bar_report(c):
        w = c.get("w_boost", w0)
        bp = bar_probs if w == w0 else log_pool(bar_boost, bar_net, w)
        lab = m0.decode_match(bp, c, bar_bnd, regime=contact.regime_for(bar, c))
        return {k: v for k, v in sanity.stats(lab).items() if k in ("n_possessions", "dead_share", "home_share")}

    cur = copy.deepcopy(base)
    best = objective(cur)
    trace = [{"step": "start", "objective": best}]
    print(f"start objective {best:.4f}", flush=True)
    for p in range(args.passes):
        changed = False
        grid = ([((None, "w_boost"), [0.0, 0.1, 0.25, 0.4])] if args.tune_blend else []) + GRID             + ([(("alpha_duration",), [2.0, 3.0, 4.0, 5.0])] if args.with_alpha else [])
        for path, values in grid:
            tgt, path = (cur, path[1:]) if path[0] is None else (cur["decoder"], path)
            here = get(tgt, path)
            cands = sorted(set(values) | {here})
            results = {}
            for v in cands:
                c = copy.deepcopy(cur)
                put(c if tgt is cur else c["decoder"], path, v)
                results[v] = objective(c)
            v_best = max(cands, key=lambda v: (round(results[v], 6), -abs(cands.index(v) - cands.index(here))))
            if results[v_best] > best + 1e-4:          # move only on a real gain on the tuning set
                put(tgt, path, v_best)
                best, changed = results[v_best], True
            trace.append({"pass": p, "param": "/".join(map(str, path)), "values": {str(k): round(x, 4) for k, x in results.items()},
                          "chosen": get(tgt, path), "objective": best})
            print(f"pass {p} {'/'.join(map(str, path)):<32} -> {get(tgt, path)}  obj {best:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if not changed:
            break
    s0_t, _ = score(base, "tune")
    s1_t, _ = score(cur, "tune")
    s0_v, per0 = score(base, "val")
    s1_v, per1 = score(cur, "val")
    comp = lambda s: 0.45 * s["v2b"] + 0.35 * s["metrica"] + 0.20 * s["v1"]  # noqa: E731
    res = {"run": args.run, "config": cfg["name"], "objective": "0.6 v2b + 0.4 v1 on DFL core (tune split)",
           "tune": {"before": s0_t, "after": s1_t}, "validation": {"before": s0_v, "after": s1_v,
           "composite_before": comp(s0_v), "composite_after": comp(s1_v),
           "per_match_before": per0, "per_match_after": per1},
           "decoder_before": base["decoder"], "decoder_after": cur["decoder"],
           "w_boost_before": w0, "w_boost_after": cur.get("w_boost", w0),
           "bar_label_free_report_only": {"before": bar_report(base), "after": bar_report(cur)},
           "trace": trace, "seconds": round(time.time() - t0)}
    Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("tune", "validation")}, indent=1, default=float)[:1500])
    print("bar (report only):", res["bar_label_free_report_only"])
    print("DECODER_TUNE_COMPLETE")


if __name__ == "__main__":
    main()
