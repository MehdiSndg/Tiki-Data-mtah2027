"""Mali's BAR-ATH components rebuilt by us (14 Nov), exactly as his tools/team_export.py writes components.npz for his
official submission, but from our rebuild: nets trained on Colab with his recipe (cl_main_ext_met, no BAR data, public
GT never read: MTAH_NO_PUBLIC=1), his DFL-only booster rebuilt locally (ev_ext_mb), his tuned decoder
(main_ext_met_dt) for the submission. Lets the final build reproduce every input from code.

    MTAH_DATA_DIR=... /home/mehdi/venvs/mali/bin/python scripts/mali_components.py <barca dir> <out dir>
    MTAH_DATA_DIR=... /home/mehdi/venvs/mali/bin/python scripts/mali_components.py <barca dir> <out dir> external

external: the rebuilt run's held-out predictions on the 7 DFL matches (his BAR-like referee) and Metrica 1-3, written
and scored exactly as his team_export.py does for his delivered external/ folder (so the two can be compared).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(barca, out, what="barath"):
    barca, out = Path(barca), Path(out)
    sys.path.insert(0, str(barca / "src"))
    sys.path.insert(0, str(barca / "tools"))
    from mtah import config as config_mod
    from mtah.paths import OUTPUTS
    from mtah.pipeline import adapt
    from mtah.pipeline.hybrid import log_pool
    from team_export import barath, external
    run = OUTPUTS / "cl_main_ext_met"
    cfg = config_mod.load(json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"])
    w = float(adapt.params(cfg)[0]["w_boost"])
    net, bnd = np.load(run / "bar_net_adapted.npy"), np.load(run / "bar_bnd_adapted.npy")
    mb = np.load(OUTPUTS / "ev_ext_mb_lgbm" / "probs.npy")
    if what == "barath":
        agree = barath(out, OUTPUTS / "cl_main_ext_met_dt_mb" / "submission.csv", log_pool(mb, net, w), bnd)
        print("written", out, "w_boost", w, "argmax agreement with the decoded labels {:.4f}".format(agree))
        return
    from mtah import cloud, constants as C
    if what == "candb_metrica":                # 14 Nov: his cand B on Metrica 1-3, which his export never built; composed on the
        from mtah.pipeline import metrica_src     # two runs' blends (both hold the same ev_ext booster at 0.25: an approximation)
        from mtah.pipeline.hybrid import compose_dead_team
        bc = config_mod.load(json.loads((OUTPUTS / "cl_candB" / "summary.json").read_text(encoding="utf-8"))["config"])
        ac = adapt.params(bc)[0]
        dt = config_mod.load("main_ext_met_dt")
        dcfg = dict(adapt.blend_cfg(bc, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
        new, ref = np.load(OUTPUTS / "cl_candB" / "dfl_lomo_blend.npz"), np.load(OUTPUTS / "cl_ev_ext_met_r1" / "dfl_lomo_blend.npz")
        nb = np.load(OUTPUTS / "cl_candB" / "dfl_lomo_bnd.npz")
        res = {}
        for m in [k for k in new.files if k.startswith("metrica_")]:
            prep = metrica_src.prepare(int(m.split("_")[1]), dict(bc, metrica_labels="ball"))
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            pb = compose_dead_team(0.5 * ref[m].astype(np.float64) + 0.5 * new[m].astype(np.float64), new[m])
            res[m] = external(out / "{}.npz".format(m), pb, prep, gt, nb[m], dcfg)
            print("cand B", m, res[m], flush=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        return
    from mtah.labels.align import suppress_short_returns
    from mtah.pipeline import dfl_referee, metrica_src
    ac = adapt.params(cfg)[0]
    dt = config_mod.load("main_ext_met_dt")
    dcfg = dict(adapt.blend_cfg(cfg, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    lengths, ecfg = cloud.bar_lengths(cfg), dfl_referee.eval_cfg(cfg)
    blend, ebnd = np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz")
    res = {}
    for m in blend.files:
        prep = dfl_referee.prepare(m, ecfg, lengths)
        gt = np.where(prep["data"].period > 0, suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True), C.X).astype(np.int8)
        res["dfl_" + m] = external(out / "dfl_{}.npz".format(m), blend[m], prep, gt, ebnd[m], dcfg)
        print(m, res["dfl_" + m], flush=True)
    vblend, vbnd = np.load(run / "dfl_lomo_blend.npz"), np.load(run / "dfl_lomo_bnd.npz")
    gcfg = dict(cfg, metrica_labels="ball")
    for m in [k for k in vblend.files if k.startswith("metrica_")]:
        prep = metrica_src.prepare(int(m.split("_")[1]), gcfg)
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        res[m] = external(out / "{}.npz".format(m), vblend[m], prep, gt, vbnd[m], dcfg)
        print(m, res[m], flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
