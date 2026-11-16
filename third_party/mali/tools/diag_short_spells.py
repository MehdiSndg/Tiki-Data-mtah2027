r"""CPU diagnostics on a finished external run (Rapor 10, Q1-1 and Q1-3; no training).

Q1-1  For true possession spells of 1-4 s on the external referees (DFL-3 v1 / v2b, Metrica LOGO): how
      many the decoded output misses, and inside missed vs found spells the mean probability of the
      true team from the booster, from the net and from the blend, plus the boundary score at the
      spell start. Says whether the evidence is absent (both low), dropped by the net (booster high,
      net low) or dropped by the decoder (blend high, still missed).
Q1-3  H<->A transition lag: for true team changes, frames from the true change to (a) the first frame
      where the blend's argmax team flips, (b) the decoded change. v1 vs v2b on the same matches.

Only aggregate numbers are printed / written (external games only).

  .venv\Scripts\python.exe tools/diag_short_spells.py --run ev_ext_met --out outputs/diag_short_spells_ev_ext_met.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine, metrica_src  # noqa: E402

SHORT = (25, 100)          # 1-4 s at 25 fps


def spells(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] in (C.H, C.A):
                out.append((s, i, int(lab[s])))
            s = i
    return out


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    ac, _ = adapt.params(cfg)
    w = float(ac["w_boost"])
    dcfg = adapt.blend_cfg(config_mod.load(args.decoder_config), ac)
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    lengths = cloud.bar_lengths(cfg)
    ecfg = dfl_referee.eval_cfg(cfg)
    extra = tuple(ac["extra_matches"])
    views = {
        "v1": (np.load(run / "dfl_lomo_blend.npz"), np.load(run / "dfl_lomo_bnd.npz"),
               {**decoder_tune.lomo_probs(cfg, folder), **decoder_tune.unseen_probs(cfg, list(extra), folder)}, cfg),
        "v2b": (np.load(run / "dfl_lomo_eval_blend.npz"), np.load(run / "dfl_lomo_eval_bnd.npz"),
                decoder_tune.eval_probs(cfg, folder, extra), ecfg),
    }
    res = {"run": args.run, "decoder": args.decoder_config, "short_frames": SHORT, "views": {}}
    for view, (blend, bnd, boost, vcfg) in views.items():
        agg = {"missed": [], "found": []}
        lags = {"emission": [], "decoded": [], "never_flips": 0, "n": 0}
        names = [m for m in blend.files if view == "v1" or not m.startswith("metrica_")]
        for m in names:
            if m.startswith("metrica_"):
                g = int(m.split("_")[1])
                prep = metrica_src.prepare(g, cfg)
                gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
                pb = decoder_tune.unseen_probs(cfg, [m], folder, preps=[prep])[m]
                corr = prep["data"]
                kind = "metrica"
            else:
                prep = dfl_referee.prepare(m, vcfg, lengths)
                gt = np.where(prep["data"].period > 0,
                              suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                              C.X).astype(np.int8)
                pb = boost[m]
                corr = prep["corrupted"]
                kind = "dfl"
            pr = blend[m]
            # net probabilities recovered exactly from the log-pool (w = booster share)
            lp = (np.log(np.clip(pr, 1e-6, 1)) - w * np.log(np.clip(pb, 1e-6, 1))) / (1 - w)
            pn = np.exp(lp - lp.max(1, keepdims=True))
            pn /= pn.sum(1, keepdims=True)
            pred = engine.decode_runs(pr, prep["runs"], dcfg, boundary=bnd[m], regime=contact.regime_for(corr, dcfg))
            ps = spells(pred)
            b = bnd[m]
            for s, e, t in spells(gt):
                if not SHORT[0] <= e - s <= SHORT[1]:
                    continue
                hit = any(tt == t and iou((s, e), (a, z)) >= 0.5 for a, z, tt in ps)
                row = {"src": kind, "match": m, "booster": float(pb[s:e, t].mean()), "net": float(pn[s:e, t].mean()),
                       "blend": float(pr[s:e, t].mean()),
                       "bnd_start": float(b[max(0, s - 5):s + 6].max()) if len(b) == len(pr) else None,
                       "ball_visible": float(np.mean(~np.isnan(corr.ball[s:e, 0])))}
                agg["found" if hit else "missed"].append(row)
            # transition lag on true H<->A changes
            for i in np.flatnonzero((gt[1:] != gt[:-1]) & np.isin(gt[1:], (C.H, C.A)) & np.isin(gt[:-1], (C.H, C.A))):
                t1 = i + 1
                new = gt[t1]
                lags["n"] += 1
                win = slice(max(0, t1 - 75), min(len(gt), t1 + 76))
                am = np.argmax(pr[win, :2], axis=1)
                idx = np.flatnonzero(am == new)
                dec = np.flatnonzero(pred[win] == new)
                if len(idx) == 0 or len(dec) == 0:
                    lags["never_flips"] += 1
                    continue
                base = t1 - win.start
                lags["emission"].append(int(idx[np.argmin(np.abs(idx - base))] - base))
                lags["decoded"].append(int(dec[np.argmin(np.abs(dec - base))] - base))

        def summ_rows(rows):
            if not rows:
                return {"n": 0}
            a = {k: np.array([r[k] for r in rows if r[k] is not None], float) for k in ("booster", "net", "blend", "bnd_start", "ball_visible")}
            return {"n": len(rows), **{f"mean_{k}": round(float(v.mean()), 3) for k, v in a.items() if len(v)},
                    "share_booster_ge_0.5_net_lt_0.5": round(float(np.mean((a["booster"] >= 0.5) & (a["net"] < 0.5))), 3),
                    "share_both_lt_0.5": round(float(np.mean((a["booster"] < 0.5) & (a["net"] < 0.5))), 3),
                    "share_blend_ge_0.5": round(float(np.mean(a["blend"] >= 0.5)), 3)}
        out = {}
        for src in ("dfl", "metrica"):
            mis = [r for r in agg["missed"] if r["src"] == src]
            fnd = [r for r in agg["found"] if r["src"] == src]
            if mis or fnd:
                out[src] = {"recall_1_4s": round(len(fnd) / max(1, len(mis) + len(fnd)), 3),
                            "missed": summ_rows(mis), "found": summ_rows(fnd)}
        em, de = np.array(lags["emission"]), np.array(lags["decoded"])
        out["transition_lag_frames"] = {"n": lags["n"], "never_flips_within_3s": lags["never_flips"],
                                        "emission_median": float(np.median(em)) if len(em) else None,
                                        "decoded_median": float(np.median(de)) if len(de) else None,
                                        "emission_abs_median": float(np.median(np.abs(em))) if len(em) else None,
                                        "decoded_abs_median": float(np.median(np.abs(de))) if len(de) else None}
        res["views"][view] = out
        print(view, json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
