r"""Score a finished external run on HELD-OUT PFF FC World Cup games (11 Nov; extra view, not one of the 4 gates).

The run's final nets (outputs/<run>/nets) and its net-input booster predict PFF games the run never trained
on: group-stage games g[1::3] of the sorted id list (training configs use g[::3]). View: the BAR-like PFF copy
of pipeline/skillcorner_src (dead ball filled and hidden with the corruption's dead fraction, player
degradation of the evaluation corruption). Referee labels: PFF game events with short returns suppressed
(35 frames), i.e. DFL-like semantics. Decoder: --decoder-config. Blend booster = the run's own (DFL-only).

  .venv\Scripts\python.exe tools/eval_pff.py --run ev_ext_met_bl [--n 12]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.io import pff  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine, skillcorner_src  # noqa: E402
from mtah.pipeline.hybrid import log_pool, net_boundary, net_probs, step_view  # noqa: E402
from mtah.validation.official import pm  # noqa: E402


def holdout(n):
    g = [x for x in pff.games() if x.startswith("38")]
    return g[1::3][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    import bar_from_nets
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, p = adapt.params(cfg)
    w, stride = float(ac["w_boost"]), int(p["stride"])
    dt = config_mod.load(args.decoder_config)
    dcfg = dict(adapt.blend_cfg(cfg, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    games = holdout(args.n)
    trained = {str(g) for g in ac.get("skillcorner_games") or ()}
    assert not trained & {f"pff_{g}" for g in games}, "held-out games overlap the run's training games"
    rc = dfl_referee.referee_cfg(dfl_referee.eval_cfg(cfg))
    vcfg = {k: v for k, v in cfg.items() if k != "metrica_dead_drift"}      # one fixed view for every run
    preps = [skillcorner_src.prepare_corrupted(f"pff_{g}", vcfg, rc) for g in games]
    names = [f"skillcorner_pff_{g}_v2b" for g in games]
    boost = decoder_tune.unseen_probs(cfg, names, folder, preps=preps)
    nets, columns = bar_from_nets.load_nets(run / "nets")
    models = [n for _, n in nets]
    res = {"run": args.run, "games": games, "rows": []}
    for g, nm, prep in zip(games, names, preps):
        assert list(prep["feats"].columns) == columns
        pb = boost[nm]
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        steps = step_view(prep["feats"], columns, pb, None, prep["runs"], stride)
        pn = net_probs(models, steps, len(pb), pb, stride, None)
        bnd = net_boundary(models, steps, len(pb), stride)
        reg = contact.regime_for(prep["corrupted"], dcfg)
        row = {"game": g}
        for ww in (0.0, w):
            pred = engine.decode_runs(log_pool(pb, pn, ww), prep["runs"], dcfg, boundary=bnd, regime=reg)
            ev = pm.evaluate(gt, pred)
            row[f"spell_w{ww:g}"] = round(ev["spell_f1_mean"], 4)
            row[f"macro_w{ww:g}"] = round(ev["macro_f1"], 4)
            row[f"Drec_w{ww:g}"] = round(ev["per_class"]["D"]["recall"], 3)
        res["rows"].append(row)
        print(row, flush=True)
    keys = [k for k in res["rows"][0] if k != "game"]
    res["mean"] = {k: round(float(np.mean([r[k] for r in res["rows"]])), 4) for k in keys}
    print("mean", res["mean"])
    (run / "eval_pff.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
