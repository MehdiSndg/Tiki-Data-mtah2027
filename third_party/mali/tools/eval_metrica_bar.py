r"""Score a finished external run on the BAR-LIKE view of its held-out Metrica games (12 Nov; extra view, not a gate).

The raw Metrica track never shows the dead ball and always shows the live one, so the raw-track referee rewards
"ball hidden = dead". Here every Metrica fold net (outputs/<run>/fold_nets/metrica_<g>) predicts its held-out
game on the copy whose dead ball is filled in, wanders at the target's out-of-bounds ball speeds (x 1.5) and is
masked like a DFL track (dead 0.58 / live 0.32), with the evaluation corruption of the players. Labels:
ball-status labels. The same view for every run.

  .venv\Scripts\python.exe tools/eval_metrica_bar.py --run ev_ext_met_blfd2
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    import bar_from_nets
    from mtah import cloud
    from mtah.pipeline import metrica_src
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, p = adapt.params(cfg)
    w, stride = float(ac["w_boost"]), int(p["stride"])
    dt = config_mod.load(args.decoder_config)
    dcfg = dict(adapt.blend_cfg(cfg, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    lengths = cloud.bar_lengths(cfg)
    rc = dfl_referee.referee_cfg(dfl_referee.eval_cfg(cfg))
    vcfg = dict(cfg, metrica_labels="ball", metrica_dead_fill=True, metrica_dead_drift=1.5)
    res = {"run": args.run, "rows": []}
    for g in (1, 2, 3):
        prep = metrica_src.prepare_corrupted(g, vcfg, rc, 0.0, lengths)
        nm = f"metrica_{g}_v2bfd1.5"
        pb = decoder_tune.unseen_probs(cfg, [nm], folder, preps=[prep])[nm]
        nets, columns = bar_from_nets.load_nets(run / "fold_nets" / f"metrica_{g}")
        models = [n for _, n in nets]
        assert list(prep["feats"].columns) == columns
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
    (run / "eval_metrica_bar.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
