r"""External-data check of the decoder constants that tools/tune_decoder_ext.py does not tune (no training).

The official decoder (main_ext_met_dt) was tuned on external data for emission scale, penalties, boundary
gamma, contact gate and the possession / dead duration medians. A few constants kept their early values:
the dead-ball emission weight w_D (emission_weights[2] = 1.15) and the dead duration sigma (0.7). Here each is
varied alone around the official value, on the official blend (saved nets + Metrica booster, w 0.25) of both
net seed sets; the 4 gates are computed against the official value. The provided game is decoded only for
the label-free gate.

  .venv\Scripts\python.exe tools/decoder_point_check.py --runs ev_ext_met ev_ext_met_s2
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
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402

import run_day_screen as screen  # noqa: E402
from duration_mix import scores  # noqa: E402
from short_spell_diag import build_refs  # noqa: E402


def set_wd(c, v):
    c["decoder"]["emission_weights"] = list(c["decoder"]["emission_weights"][:2]) + [v]


def set_dead_sigma(c, v):
    c["decoder"]["duration"]["dead"]["sigma"] = v


CHECKS = [("w_D", set_wd, 1.15, [1.0, 1.3]), ("dead_sigma", set_dead_sigma, 0.7, [0.5, 0.9])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["ev_ext_met", "ev_ext_met_s2"])
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    table = {}
    for run_name in args.runs:
        refs, dcfg, w, run, new = build_refs(run_name, args.decoder_config)
        assert abs(dcfg["decoder"]["emission_weights"][2] - 1.15) < 1e-9
        assert abs(dcfg["decoder"]["duration"]["dead"]["sigma"] - 0.7) < 1e-9
        cfg = config_mod.load(json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"])
        bar = (log_pool(new["BAR"], np.load(run / "bar_net_adapted.npy"), w), np.load(run / "bar_bnd_adapted.npy"),
               adapt.bar_context(cfg)["data"])
        table[run_name] = {"official": scores(refs, dcfg, w, bar)}
        for name, setter, _, values in CHECKS:
            for v in values:
                c = copy.deepcopy(dcfg)
                setter(c, v)
                s = scores(refs, c, w, bar)
                table[run_name][f"{name}={v}"] = s
                print(f"{run_name:<14} {name}={v:<5} v1 {s['dfl_v1']:.4f} v2b {s['dfl_v2b']:.4f} "
                      f"Metrica {s['metrica']:.4f} | BAR {s['bar']['n_possessions']} D {s['bar']['dead_share']:.3f}",
                      flush=True)
        o = table[run_name]["official"]
        print(f"{run_name:<14} official   v1 {o['dfl_v1']:.4f} v2b {o['dfl_v2b']:.4f} Metrica {o['metrica']:.4f}",
              flush=True)
    res = {"runs": args.runs, "table": table, "gates": {}}
    for key in [k for k in table[args.runs[0]] if k != "official"]:
        g = screen.four_gates({r: {"reference": table[r]["official"], "candidate": table[r][key]} for r in args.runs})
        res["gates"][key] = g
        d = g["delta"]
        print(f"{key:<16} d_comp {g['delta_composite']:+.4f} v1 {d['dfl_v1']:+.4f} v2b {d['dfl_v2b']:+.4f} "
              f"Met {d['metrica']:+.4f} pos {g['positive_matches']}/6 pass {g['pass']}", flush=True)
    (OUTPUTS / "decoder_point_check.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
