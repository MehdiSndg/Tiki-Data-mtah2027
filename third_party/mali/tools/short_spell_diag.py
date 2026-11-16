r"""Where is the evidence for the short possessions the official stack misses? (diagnostic, no training)

External data only (DFL-3 held-out matches under v1 and v2b corruption, Metrica 1-3 with nested booster).
The net probabilities are the run's saved ones, the booster the cached Metrica booster of tools/booster_met.py
(the official blend: w_boost 0.25, main_ext_met_dt decoder). For every ground-truth H/A spell:

  found       a predicted spell of the same team with temporal IoU >= 0.5
  evidence    mean P(team | alive) inside the spell for booster, net and blend (> 0.5 = points to the right team)
  ball        share of the spell's frames with the ball visible on the (corrupted) track

Prints only aggregates per duration bucket; writes outputs/short_spell_diag_<run>.json.

  .venv\Scripts\python.exe tools/short_spell_diag.py --run ev_ext_met
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine, metrica_src  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402

BUCKETS = ((0, 50, "<2s"), (50, 125, "2-5s"), (125, 250, "5-10s"), (250, 10 ** 9, ">10s"))


def spells(lab):
    """[(start, end, team)] runs of H (0) / A (1)."""
    out, i, n = [], 0, len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        if lab[i] in (0, 1):
            out.append((i, j + 1, int(lab[i])))
        i = j + 1
    return out


def best_iou(s, pred_by_team):
    a, b, t = s
    best = 0.0
    for c, d in pred_by_team[t]:
        if d <= a:
            continue
        if c >= b:
            break
        inter = min(b, d) - max(a, c)
        best = max(best, inter / (max(b, d) - min(a, c)))
    return best


def team_p(p, a, b, t):
    q = p[a:b, :2]
    return float(np.mean(q[:, t] / np.clip(q.sum(1), 1e-9, None)))


def build_refs(run_name, decoder_config="main_ext_met_dt"):
    """(refs, dcfg, w, run_dir, new booster dict): refs = [(view, match, prep, gt, net, booster, boundary)]."""
    run = OUTPUTS / run_name
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, _ = adapt.params(cfg)
    w = float(ac["w_boost"])
    dt = config_mod.load(decoder_config)
    dcfg = dict(adapt.blend_cfg(cfg, ac), decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    lengths = cloud.bar_lengths(cfg)
    ecfg = dfl_referee.eval_cfg(cfg)
    extra = [m for m in ac["eval_matches"] if not m.startswith("metrica_")]
    cache = OUTPUTS / "booster_met" / f"{run_name}.npz"
    if not cache.is_file():
        raise SystemExit(f"run tools/booster_met.py --run {run_name} first (needs {cache})")
    z = np.load(cache)
    new = {k: z[k] for k in z.files}
    old_v2b = decoder_tune.eval_probs(cfg, folder, tuple(extra))
    net_v1, bnd_v1 = np.load(run / "dfl_lomo_net_adapted.npz"), np.load(run / "dfl_lomo_bnd.npz")
    blend_v2b, bnd_v2b = np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz")

    def net_from_blend(pr, pb):
        lp = (np.log(np.clip(pr, 1e-6, 1)) - w * np.log(np.clip(pb, 1e-6, 1))) / (1 - w)
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def gt_dfl(prep):
        return np.where(prep["data"].period > 0,
                        suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                        C.X).astype(np.int8)

    refs = []
    for m in net_v1.files:
        if m.startswith("metrica_"):
            prep = metrica_src.prepare(int(m.split("_")[1]), cfg)
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            refs.append(("metrica", m, prep, gt, net_v1[m], new[m], bnd_v1[m]))
        else:
            prep = dfl_referee.prepare(m, cfg, lengths)
            refs.append(("v1", m, prep, gt_dfl(prep), net_v1[m], new[m], bnd_v1[m]))
    for m in blend_v2b.files:
        prep = dfl_referee.prepare(m, ecfg, lengths)
        refs.append(("v2b", m, prep, gt_dfl(prep), net_from_blend(blend_v2b[m], old_v2b[m]), new[m + "_v2b"],
                     bnd_v2b[m]))
    return refs, dcfg, w, run, new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    refs, dcfg, w, _, _ = build_refs(args.run, args.decoder_config)

    rows = []           # one per GT spell
    pred_rows = []      # one per predicted spell
    for view, m, prep, gt, pn, pb, bnd in refs:
        pr = log_pool(pb, pn, w)
        pred = np.asarray(engine.decode_runs(pr, prep["runs"], dcfg, boundary=bnd,
                                             regime=contact.regime_for(prep["corrupted"], dcfg)))
        vis = ~np.isnan(prep["corrupted"].ball[:, 0])
        gs, ps = spells(gt), spells(pred)
        pbt = {t: [(a, b) for a, b, tt in ps if tt == t] for t in (0, 1)}
        gbt = {t: [(a, b) for a, b, tt in gs if tt == t] for t in (0, 1)}
        for s in gs:
            a, b, t = s
            rows.append(dict(view=view, match=m, dur=b - a, found=best_iou(s, pbt) >= 0.5,
                             boost=team_p(pb, a, b, t), net=team_p(pn, a, b, t), blend=team_p(pr, a, b, t),
                             alive_blend=float(np.mean(1 - pr[a:b, 2])), ball=float(np.mean(vis[a:b])),
                             pred_same=float(np.mean(pred[a:b] == t))))
        for s in ps:
            pred_rows.append(dict(view=view, dur=s[1] - s[0], correct=best_iou(s, gbt) >= 0.5))

    def agg(sel, prs):
        out = {}
        for lo, hi, name in BUCKETS:
            r = [x for x in sel if lo <= x["dur"] < hi]
            p = [x for x in prs if lo <= x["dur"] < hi]
            if not r:
                continue
            miss = [x for x in r if not x["found"]]
            d = dict(n_gt=len(r), recall=round(np.mean([x["found"] for x in r]), 3),
                     n_pred=len(p), precision=round(np.mean([x["correct"] for x in p]), 3) if p else None)
            for grp, xs in (("found", [x for x in r if x["found"]]), ("missed", miss)):
                if xs:
                    d[grp] = {k: round(float(np.mean([x[k] for x in xs])), 3)
                              for k in ("boost", "net", "blend", "alive_blend", "ball", "pred_same")}
            if miss:
                d["missed_evidence"] = {
                    "boost>0.5": round(np.mean([x["boost"] > 0.5 for x in miss]), 3),
                    "net>0.5": round(np.mean([x["net"] > 0.5 for x in miss]), 3),
                    "boost>0.5_net<=0.5": round(np.mean([x["boost"] > 0.5 >= x["net"] for x in miss]), 3),
                    "neither": round(np.mean([max(x["boost"], x["net"]) <= 0.5 for x in miss]), 3),
                    "ball_visible>0.5": round(np.mean([x["ball"] > 0.5 for x in miss]), 3),
                    "alive<0.5 (decoded dead)": round(np.mean([x["alive_blend"] < 0.5 for x in miss]), 3)}
            out[name] = d
        return out

    res = {"run": args.run, "w_boost": w, "decoder_config": args.decoder_config,
           "missed_share_of_all_misses": {}, "views": {}}
    for view in ("v1", "v2b", "metrica"):
        sel = [x for x in rows if x["view"] == view]
        res["views"][view] = agg(sel, [x for x in pred_rows if x["view"] == view])
        miss = [x for x in sel if not x["found"]]
        res["missed_share_of_all_misses"][view] = {
            name: round(np.mean([lo <= x["dur"] < hi for x in miss]), 3) for lo, hi, name in BUCKETS}
    out = OUTPUTS / f"short_spell_diag_{args.run}.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
