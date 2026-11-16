r"""Boundary timing audit and two-seed-set union on the official blend (no training; external data only).

EVALUATION.md: a perfect prediction that is 8 frames late scores 0.89, 1 s late 0.65 -- a systematic boundary
offset costs more than any model change we test. Here, on the external referees (DFL-3 under v1 / v2b
corruption, Metrica 1-3; saved nets + Metrica booster, w 0.25, main_ext_met_dt decoder):

  timing   signed offset (predicted - true, frames) of every true boundary to the nearest predicted boundary of
           the same kind (team change, possession start after dead, possession end into dead) within +-50
           frames; and spellF1 when the whole prediction is shifted by k frames (k > 0 = later).
  union    the net probabilities (and boundary heads) of both seed sets averaged (arithmetic / geometric)
           before the blend; 4 gates against each single set. The provided game is decoded only for the
           label-free gate.

  .venv\Scripts\python.exe tools/timing_union.py
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
from mtah.decoding import contact  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, engine, m0  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

import run_day_screen as screen  # noqa: E402
from short_spell_diag import build_refs  # noqa: E402

SHIFTS = (-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12)
KINDS = ("change", "start", "end")
VIEW = {"v1": "dfl_v1", "v2b": "dfl_v2b", "metrica": "metrica"}


def boundaries(lab):
    """{kind: frames} of label changes: H<->A, D->team (start), team->D (end). X neighbours are skipped."""
    lab = np.asarray(lab)
    i = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    a, b = lab[i - 1], lab[i]
    team = lambda x: (x == 0) | (x == 1)  # noqa: E731
    return {"change": i[team(a) & team(b)], "start": i[(a == 2) & team(b)], "end": i[team(a) & (b == 2)]}


def offsets(gt, pred, window=50):
    g, p = boundaries(gt), boundaries(pred)
    out = {}
    for k in KINDS:
        if len(p[k]) == 0:
            out[k] = (np.array([]), len(g[k]))
            continue
        j = np.clip(np.searchsorted(p[k], g[k]), 1, len(p[k]) - 1) if len(p[k]) > 1 else np.zeros(len(g[k]), int)
        cand = np.stack([p[k][np.maximum(j - 1, 0)], p[k][j]], 1) - g[k][:, None]
        d = cand[np.arange(len(cand)), np.abs(cand).argmin(1)]
        out[k] = (d[np.abs(d) <= window], len(g[k]))
    return out


def shifted(pred, runs, k):
    """Prediction delayed by k frames inside every run (k < 0 = earlier); edges repeat."""
    out = np.asarray(pred).copy()
    for a, b in runs:
        idx = np.clip(np.arange(a, b) - k, a, b - 1)
        out[a:b] = pred[idx]
    return out


def decode(ref, net, bnd, dcfg, w):
    view, m, prep, gt, _, pb, _ = ref
    return np.asarray(engine.decode_runs(log_pool(pb, net, w), prep["runs"], dcfg, boundary=bnd,
                                         regime=contact.regime_for(prep["corrupted"], dcfg)))


def summarise(refs, preds, bar_stats):
    per = {v: {} for v in VIEW}
    for (view, m, prep, gt, *_), pred in zip(refs, preds):
        per[view][m] = pm.evaluate(gt, pred)["spell_f1_mean"]
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    return {"dfl_v1": mean(per["v1"]), "dfl_v2b": mean(per["v2b"]), "metrica": mean(per["metrica"]),
            "dfl_v1_per_match": per["v1"], "dfl_v2b_per_match": per["v2b"], "metrica_per_match": per["metrica"],
            "bar": bar_stats}


def bar_decode(run, cfg, new, net, bnd, dcfg, w):
    lab = m0.decode_match(log_pool(new["BAR"], net, w), dcfg, bnd,
                          regime=contact.regime_for(adapt.bar_context(cfg)["data"], dcfg))
    st = sanity.stats(lab)
    return {k: st[k] for k in ("n_possessions", "dead_share", "home_share")}


def geo(a, b):
    lp = 0.5 * (np.log(np.clip(a, 1e-6, 1)) + np.log(np.clip(b, 1e-6, 1)))
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs=2, default=["ev_ext_met", "ev_ext_met_s2"])
    ap.add_argument("--decoder-config", default="main_ext_met_dt")
    args = ap.parse_args()
    built = {r: build_refs(r, args.decoder_config) for r in args.runs}
    res = {"runs": args.runs, "timing": {}, "shift": {}, "single": {}, "union": {}}
    preds, bars, nets = {}, {}, {}
    for r, (refs, dcfg, w, run, new) in built.items():
        cfg = config_mod.load(json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"])
        nets[r] = (np.load(run / "bar_net_adapted.npy"), np.load(run / "bar_bnd_adapted.npy"), cfg, run, new)
        preds[r] = [decode(ref, ref[4], ref[6], dcfg, w) for ref in refs]
        bars[r] = bar_decode(run, cfg, new, nets[r][0], nets[r][1], dcfg, w)
        res["single"][r] = s = summarise(refs, preds[r], bars[r])
        print(f"{r:<14} single  v1 {s['dfl_v1']:.4f} v2b {s['dfl_v2b']:.4f} Metrica {s['metrica']:.4f} | BAR {bars[r]}",
              flush=True)
        # timing
        acc = {v: {k: [[], 0] for k in KINDS} for v in VIEW}
        for ref, pred in zip(refs, preds[r]):
            for k, (d, n) in offsets(ref[3], pred).items():
                acc[ref[0]][k][0].extend(d.tolist())
                acc[ref[0]][k][1] += n
        res["timing"][r] = {}
        for v in VIEW:
            res["timing"][r][v] = {}
            for k in KINDS:
                d, n = np.array(acc[v][k][0]), acc[v][k][1]
                t = dict(n_true=n, matched=round(len(d) / max(n, 1), 3), median=float(np.median(d)),
                         mean=round(float(d.mean()), 2), p25=float(np.percentile(d, 25)),
                         p75=float(np.percentile(d, 75)), abs_median=float(np.median(np.abs(d))))
                res["timing"][r][v][k] = t
                print(f"{r:<14} {v:<8} {k:<7} n {n:>4} matched {t['matched']:.2f} offset median {t['median']:+.0f} "
                      f"mean {t['mean']:+.1f} IQR [{t['p25']:+.0f}, {t['p75']:+.0f}] |offset| median {t['abs_median']:.0f}",
                      flush=True)
        res["shift"][r] = {}
        for k in SHIFTS:
            s = summarise(refs, [shifted(p, ref[2]["runs"], k) for ref, p in zip(refs, preds[r])], bars[r])
            res["shift"][r][k] = s
            print(f"{r:<14} shift {k:+3d}  v1 {s['dfl_v1']:.4f} v2b {s['dfl_v2b']:.4f} Metrica {s['metrica']:.4f}",
                  flush=True)
    for k in SHIFTS:
        if k == 0:
            continue
        g = screen.four_gates({r: {"reference": res["shift"][r][0], "candidate": res["shift"][r][k]} for r in args.runs})
        d = g["delta"]
        print(f"shift {k:+3d} d_comp {g['delta_composite']:+.4f} v1 {d['dfl_v1']:+.4f} v2b {d['dfl_v2b']:+.4f} "
              f"Met {d['metrica']:+.4f} pos {g['positive_matches']}/6 pass {g['pass']}", flush=True)
        res["shift"][f"gate_{k}"] = g
    # union of both seed sets
    r1, r2 = args.runs
    refs1, dcfg, w, _, _ = built[r1]
    refs2 = built[r2][0]
    assert [(a[0], a[1]) for a in refs1] == [(a[0], a[1]) for a in refs2]
    for name, comb in (("arith", lambda a, b: 0.5 * (a + b)), ("geo", geo)):
        up = [decode(a, comb(a[4], b[4]), 0.5 * (a[6] + b[6]), dcfg, w) for a, b in zip(refs1, refs2)]
        n1, n2 = nets[r1], nets[r2]
        bar = bar_decode(n1[3], n1[2], n1[4], comb(n1[0], n2[0]), 0.5 * (n1[1] + n2[1]), dcfg, w)
        s = summarise(refs1, up, bar)
        g = screen.four_gates({r: {"reference": res["single"][r], "candidate": s} for r in args.runs})
        res["union"][name] = {"scores": s, "gates": g}
        d = g["delta"]
        print(f"union {name:<6} v1 {s['dfl_v1']:.4f} v2b {s['dfl_v2b']:.4f} Metrica {s['metrica']:.4f} | BAR {bar}\n"
              f"   vs mean of singles: d_comp {g['delta_composite']:+.4f} v1 {d['dfl_v1']:+.4f} v2b {d['dfl_v2b']:+.4f} "
              f"Met {d['metrica']:+.4f} pos {g['positive_matches']}/6 gates {g['gates']}", flush=True)
    (OUTPUTS / "timing_union.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
