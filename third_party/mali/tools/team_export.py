r"""Team hand-over package (team/PROTOCOL.md of the team repo, 12 Nov): model outputs only, no provided data.

barath/    submission.csv, probs.npz (blend before the decoder), components.npz
external/  one npz per held-out external match: p_home / p_away / p_dead (float16), valid, labels, fps, half,
           frame (the data set's own frame numbers) + pred (our decoder's labels, so the score can be recomputed)

  .venv\Scripts\python.exe tools/team_export.py --out <team folder>
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, dfl_referee, engine, metrica_src  # noqa: E402
from mtah.pipeline.hybrid import compose_dead_team, log_pool  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def read_submission(path):
    lab = [ln.strip().split(",")[1] for ln in Path(path).read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    return np.array([CODES[x] for x in lab], np.int8)


def norm(p):
    p = np.nan_to_num(np.asarray(p, np.float64), nan=1.0 / 3)
    return p / p.sum(1, keepdims=True)


def barath(folder, sub_path, probs, bnd):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sub_path, folder / "submission.csv")
    lab = read_submission(sub_path)
    p = norm(probs)
    assert len(p) == len(lab)
    np.savez_compressed(folder / "probs.npz", p_home=p[:, 0].astype(np.float16), p_away=p[:, 1].astype(np.float16),
                        p_dead=p[:, 2].astype(np.float16), valid=lab != C.X)
    alive = np.maximum(p[:, 0] + p[:, 1], 1e-9)
    np.savez_compressed(folder / "components.npz", p_dead=p[:, 2].astype(np.float16),
                        p_home_given_alive=(p[:, 0] / alive).astype(np.float16),
                        p_boundary=np.nan_to_num(np.asarray(bnd, np.float64)).astype(np.float16), valid=lab != C.X)
    return float(np.mean(p.argmax(1)[lab != C.X] == lab[lab != C.X]))


def external(path, probs, prep, gt, bnd, dcfg):
    data = prep["data"]
    pred = np.asarray(engine.decode_runs(probs, prep["runs"], dcfg, boundary=bnd,
                                         regime=contact.regime_for(prep["corrupted"], dcfg)), np.int8)
    p = norm(probs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p_home=p[:, 0].astype(np.float16), p_away=p[:, 1].astype(np.float16),
                        p_dead=p[:, 2].astype(np.float16), valid=np.asarray(data.period) > 0,
                        labels=np.asarray(gt, np.int8), fps=float(data.fps), half=np.asarray(data.period, np.int8),
                        frame=np.asarray(data.frame, np.int32), pred=pred)
    ev = pm.evaluate(gt, pred)
    return {"spell_f1": round(ev["spell_f1_mean"], 4), "macro_f1": round(ev["macro_f1"], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    res = {}

    # ---- official submission f4080c4c: main_ext_met nets + ev_ext_mb booster + main_ext_met_dt decoder
    run = OUTPUTS / "main_ext_met"
    cfg = config_mod.load(json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"])
    ac, _ = adapt.params(cfg)
    w = float(ac["w_boost"])
    dt = config_mod.load("main_ext_met_dt")
    dcfg = dict(adapt.blend_cfg(cfg, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    net, bnd = np.load(run / "bar_net_adapted.npy"), np.load(run / "bar_bnd_adapted.npy")
    mb = np.load(OUTPUTS / "ev_ext_mb_lgbm" / "probs.npy")
    res["barath_official_argmax_agreement"] = barath(out / "barath", OUTPUTS / "main" / "submission.csv",
                                                     log_pool(mb, net, w), bnd)

    # ---- candidate B (not submitted): team split from the PFF-trained nets, P(D) = mean of the two recipes
    pff_net = np.load(OUTPUTS / "ev_ext_met_blfd2_pff" / "bar_net_adapted.npy").astype(np.float64)
    mix = compose_dead_team(0.5 * net.astype(np.float64) + 0.5 * pff_net, pff_net)
    res["barath_candB_argmax_agreement"] = barath(
        out / "barath" / "cand_mix_pff", OUTPUTS / "cand_mix_pff_s1" / "submission.csv", log_pool(mb, mix, w),
        np.load(OUTPUTS / "ev_ext_met_blfd2_pff" / "bar_bnd_adapted.npy"))

    # ---- external, held-out: 7 DFL matches (leave one match out), BAR-like corrupted view; nets + DFL-only booster
    lengths = cloud.bar_lengths(cfg)
    ecfg = dfl_referee.eval_cfg(cfg)
    blend, ebnd = np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz")
    res["external"] = {}
    for m in blend.files:
        prep = dfl_referee.prepare(m, ecfg, lengths)
        gt = np.where(prep["data"].period > 0,
                      suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True), C.X).astype(np.int8)
        res["external"][f"dfl_{m}"] = external(out / "external" / f"dfl_{m}.npz", blend[m], prep, gt, ebnd[m], dcfg)
        print(m, res["external"][f"dfl_{m}"], flush=True)
    # ---- Metrica sample games 1-3 (each held out of the nets; the DFL-only booster never saw Metrica), raw track
    vblend, vbnd = np.load(run / "dfl_lomo_blend.npz"), np.load(run / "dfl_lomo_bnd.npz")
    gcfg = dict(cfg, metrica_labels="ball")
    for m in [k for k in vblend.files if k.startswith("metrica_")]:
        prep = metrica_src.prepare(int(m.split("_")[1]), gcfg)
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        res["external"][m] = external(out / "external" / f"{m}.npz", vblend[m], prep, gt, vbnd[m], dcfg)
        print(m, res["external"][m], flush=True)
    # ---- candidate B on the 3 DFL matches its fold nets held out (same seed set: ev_ext_met_r1 + ev_ext_met_blfd2_pff)
    from mtah.pipeline import decoder_tune

    def fold_net(name):
        r = OUTPUTS / name
        c = config_mod.load(json.loads((r / "summary.json").read_text(encoding="utf-8"))["config"])
        a, _ = adapt.params(c)
        z = np.load(r / "dfl_lomo_eval_blend.npz")
        old = decoder_tune.eval_probs(c, OUTPUTS / "decoder" / (a.get("booster_source") or c["name"]), tuple(z.files))
        w0, o = float(a["w_boost"]), {}
        for m in z.files:
            lp = (np.log(np.clip(z[m], 1e-6, 1)) - w0 * np.log(np.clip(old[m], 1e-6, 1))) / (1 - w0)
            q = np.exp(lp - lp.max(1, keepdims=True))
            o[m] = (q / q.sum(1, keepdims=True), old[m])
        return o, np.load(r / "dfl_lomo_eval_bnd.npz")

    ref, _ = fold_net("ev_ext_met_r1")
    new, nbnd = fold_net("ev_ext_met_blfd2_pff")
    res["external_candB"] = {}
    for m in new:
        prep = dfl_referee.prepare(m, ecfg, lengths)
        gt = np.where(prep["data"].period > 0,
                      suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True), C.X).astype(np.int8)
        pn = compose_dead_team(0.5 * ref[m][0] + 0.5 * new[m][0], new[m][0])
        res["external_candB"][f"dfl_{m}"] = external(out / "external" / "cand_mix_pff" / f"dfl_{m}.npz",
                                                     log_pool(new[m][1], pn, w), prep, gt, nbnd[m], dcfg)
        print("cand B", m, res["external_candB"][f"dfl_{m}"], flush=True)
    (out / "export_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
