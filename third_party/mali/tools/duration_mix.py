r"""Short-possession duration component in the decoder (no training; external data only).

tools/short_spell_diag.py: ~78 % of missed DFL possessions are < 5 s, and in ~35 % of the missed < 2 s ones
both booster and net already point to the right team -- the decoder's single log-normal duration prior
(median 150 frames) suppresses them. Here every H / A segment may instead use a short log-normal component
(decoder.contested_duration with the contested mask = every frame: a two-component duration mixture, Viterbi
max). Nothing else changes: the official blend (saved nets + Metrica booster, w 0.25, main_ext_met_dt decoder).

Reference = the same decode without the component. Gates: tools/run_day_screen.four_gates over both net seed
sets (each set reported apart as well). The provided game is decoded only for the label-free gate.

  .venv\Scripts\python.exe tools/duration_mix.py --runs ev_ext_met ev_ext_met_s2
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import config as config_mod  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, engine, m0  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

import run_day_screen as screen  # noqa: E402
from short_spell_diag import build_refs  # noqa: E402

GRID = [None] + [dict(median=m, sigma=s) for m in (25, 40, 60) for s in (0.4, 0.6)]


def scores(refs, dcfg, w, bar):
    per = {"v1": {}, "v2b": {}, "metrica": {}}
    for view, m, prep, gt, pn, pb, b in refs:
        pr = log_pool(pb, pn, w)
        pred = engine.decode_runs(pr, prep["runs"], dcfg, boundary=b, contested=np.ones(len(pr), bool),
                                  regime=contact.regime_for(prep["corrupted"], dcfg))
        per[view][m] = pm.evaluate(gt, pred)["spell_f1_mean"]
    probs, bnd, data = bar
    lab = m0.decode_match(probs, dcfg, bnd, contested=np.ones(len(probs), bool),
                          regime=contact.regime_for(data, dcfg))
    st = sanity.stats(lab)
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    return {"dfl_v1": mean(per["v1"]), "dfl_v2b": mean(per["v2b"]), "metrica": mean(per["metrica"]),
            "dfl_v1_per_match": per["v1"], "dfl_v2b_per_match": per["v2b"], "metrica_per_match": per["metrica"],
            "bar": {k: st[k] for k in ("n_possessions", "dead_share", "home_share")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["ev_ext_met", "ev_ext_met_s2"])
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    table = {}          # run -> [scores per grid point]
    for run_name in args.runs:
        refs, dcfg, w, run, new = build_refs(run_name, args.decoder_config)
        cfg = config_mod.load(json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"])
        bar = (log_pool(new["BAR"], np.load(run / "bar_net_adapted.npy"), w), np.load(run / "bar_bnd_adapted.npy"),
               adapt.bar_context(cfg)["data"])
        table[run_name] = []
        for spec in GRID:
            c = copy.deepcopy(dcfg)
            if spec:
                c["decoder"]["contested_duration"] = spec
            else:
                c["decoder"].pop("contested_duration", None)
            s = scores(refs, c, w, bar)
            table[run_name].append(s)
            print(f"{run_name:<14} {str(spec):<32} v1 {s['dfl_v1']:.4f} v2b {s['dfl_v2b']:.4f} "
                  f"Metrica {s['metrica']:.4f} | BAR {s['bar']['n_possessions']} D {s['bar']['dead_share']:.3f} "
                  f"home {s['bar']['home_share']:.3f}", flush=True)
    res = {"runs": args.runs, "grid": GRID, "table": table, "gates": []}
    for i, spec in enumerate(GRID[1:], 1):
        both = screen.four_gates({r: {"reference": table[r][0], "candidate": table[r][i]} for r in args.runs})
        each = {r: screen.four_gates({r: {"reference": table[r][0], "candidate": table[r][i]}})["pass"]
                for r in args.runs}
        res["gates"].append({"spec": spec, "both": both, "each_pass": each})
        d = both["delta"]
        print(f"{str(spec):<32} d_comp {both['delta_composite']:+.4f} v1 {d['dfl_v1']:+.4f} v2b {d['dfl_v2b']:+.4f} "
              f"Met {d['metrica']:+.4f} pos {both['positive_matches']}/6 gates {both['gates']} each {each}", flush=True)
    (OUTPUTS / "duration_mix.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
