r"""Fixed-probability ensembles of finished external-data-only `adapt` runs (5 Nov 2026).

Nothing is retrained or tuned: every member's saved, blended (booster + net) probabilities are combined
with EQUAL weights in log space (geometric mean, renormalised; boundary probabilities averaged) and
decoded with one fixed decoder config. Scored on the screen referees (3 booster-unseen DFL matches,
v1 + v2b corruption, and the 3 Metrica games, each out of sample for every member) and checked with the
4 gates against the reference run decoded by the same code. The provided game is only decoded for the
label-free plausibility gate; no ground truth of it is read.

  .venv\Scripts\python.exe tools/ensemble_ext.py --reference ev_ext_met \
      --combos ev_ext_met+ev_ext_met_sc ev_ext_met+ev_ext_met_sc+ev_ext_met_sc20 --out outputs/ensemble_ext.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.io.broadcast import bar_data  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.labels.masking import gap_lengths  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, dfl_referee, engine, m0, metrica_src  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402


def member_views(run):
    """{(kind, match): (probs, bnd)} of one run; Metrica from its LOMO folds or its evaluation file."""
    d = OUTPUTS / run
    out = {}
    v1, v1b = np.load(d / "dfl_lomo_blend.npz"), np.load(d / "dfl_lomo_bnd.npz")
    v2, v2b = np.load(d / "dfl_lomo_eval_blend.npz"), np.load(d / "dfl_lomo_eval_bnd.npz")
    for m in v1.files:
        out[("metrica" if m.startswith("metrica_") else "v1", m)] = (v1[m], v1b[m])
    for m in v2.files:
        out[("v2b", m)] = (v2[m], v2b[m])
    if (d / "metrica_eval_blend.npz").is_file():
        me, meb = np.load(d / "metrica_eval_blend.npz"), np.load(d / "metrica_eval_bnd.npz")
        for m in me.files:
            out[("metrica", m)] = (me[m], meb[m])
    out[("bar", "bar")] = (np.load(d / "probs.npy"), np.load(d / "bar_bnd_adapted.npy"))
    return out


def combine(parts):
    lp = np.mean([np.log(np.clip(p, 1e-6, 1.0)) for p, _ in parts], axis=0)
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    p = np.round(p / p.sum(1, keepdims=True), 6)
    return p, np.mean([b for _, b in parts], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--combos", nargs="+", required=True, help="members joined with '+'")
    ap.add_argument("--decoder-config", default=None, help="decoder config (default: the reference run's config)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    import run_day_screen as screen
    ref_summary = json.loads((OUTPUTS / args.reference / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(args.decoder_config or ref_summary["config"])
    ac, _ = adapt.params(cfg)
    dcfg = adapt.blend_cfg(cfg, ac)
    ecfg = dfl_referee.eval_cfg(config_mod.load(ref_summary["config"]))
    lengths = gap_lengths(bar_data(cfg))
    runs = sorted({m for c in [args.reference] + args.combos for m in c.split("+")})
    views = {r: member_views(r) for r in runs}
    keys = sorted(set.intersection(*[set(v) for v in views.values()]))
    refs = {}
    for kind, m in keys:
        if kind == "bar":
            continue
        if kind == "metrica":
            prep = metrica_src.prepare(int(m.split("_")[1]), cfg)
            gt, corr = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8), prep["data"]
        else:
            prep = dfl_referee.prepare(m, cfg if kind == "v1" else ecfg, lengths)
            gt = np.where(prep["data"].period > 0,
                          suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                          C.X).astype(np.int8)
            corr = prep["corrupted"]
        refs[(kind, m)] = (prep["runs"], gt, corr)
    bar = bar_data(cfg)
    print(f"members {runs}; referees {sorted(refs)} ({time.time() - t0:.0f}s)", flush=True)

    def scores(combo):
        members = combo.split("+")
        per = {"v1": {}, "v2b": {}, "metrica": {}}
        for key, (rruns, gt, corr) in refs.items():
            probs, bnd = combine([views[r][key] for r in members])
            pred = engine.decode_runs(probs, rruns, dcfg, boundary=bnd, regime=contact.regime_for(corr, dcfg))
            per[key[0]][key[1]] = round(pm.evaluate(gt, pred)["spell_f1_mean"], 4)
        probs, bnd = combine([views[r][("bar", "bar")] for r in members])
        lab = m0.decode_match(probs, dcfg, bnd, regime=contact.regime_for(bar, dcfg))
        st = sanity.stats(lab)
        mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
        return {"dfl_v1": mean(per["v1"]), "dfl_v2b": mean(per["v2b"]), "metrica": mean(per["metrica"]),
                "dfl_v1_per_match": per["v1"], "dfl_v2b_per_match": per["v2b"], "metrica_per_match": per["metrica"],
                "bar": {k: st[k] for k in ("n_possessions", "dead_share", "home_share")}}

    ref = scores(args.reference)
    print(f"reference {args.reference}: v1 {ref['dfl_v1']:.4f} v2b {ref['dfl_v2b']:.4f} met {ref['metrica']:.4f} "
          f"bar {ref['bar']} ({time.time() - t0:.0f}s)", flush=True)
    res = {"reference": args.reference, "decoder_config": cfg["name"], "rule": "equal-weight log-pool, mean boundary",
           "reference_scores": ref, "combos": {}}
    for combo in args.combos:
        cand = scores(combo)
        g = screen.four_gates({"x": {"reference": ref, "candidate": cand}})
        res["combos"][combo] = {"scores": cand, "four_gates": g}
        print(f"{combo}: v1 {cand['dfl_v1']:.4f} v2b {cand['dfl_v2b']:.4f} met {cand['metrica']:.4f} | "
              f"d comp {g['delta_composite']:+.4f} pos {g['positive_matches']}/6 gates {g['gates']} bar {cand['bar']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("ENSEMBLE_COMPLETE")


if __name__ == "__main__":
    main()
