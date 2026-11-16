"""Decoder calibration (Gün 2 / Adım 4a-b): HSMM duration priors + post-processing on FIXED
probabilities.

Objective = mean official spellF1 on the DFL leave-one-match-out probabilities of the given
config (whole matches, aligned GT). The public window is never used to select: its score and
the BAR-ATH full-match statistics are only reported next to each candidate.
"""

import copy
import itertools
import json
import time

import numpy as np

from .. import cloud
from .. import config as config_mod
from .. import constants as C
from ..decoding import post
from ..features import build_features
from ..io.broadcast import bar_data
from ..io.tags import load_tags
from ..labels.masking import gap_lengths
from ..paths import OUTPUTS, ensure
from ..validation import sanity
from ..validation.scoring import score_blocks
from ..validation.splits import labelled_blocks, public_blocks
from . import dfl_referee, engine, joint, m0

GRID = dict(possession_median=[150, 200, 250], possession_sigma=[0.85, 1.1],
            penalty_turnover=[8.0, 14.0], dead_median=[350, 500], d_weight=[1.0, 1.15, 1.3])


def lomo_probs(cfg, folder):
    """{match: probs} of the config's LOMO folds (cached in `folder`)."""
    matches = joint.available(cfg["joint"]["matches"])
    from ..validation.splits import cache_tag
    tag = cache_tag(cfg)
    paths = {m: folder / f"lomo_{m}{tag}.npy" for m in matches}
    if all(p.is_file() for p in paths.values()):
        return {m: np.load(p) for m, p in paths.items()}
    # train in a scratch run folder: never overwrite outputs/<name>_lgbm (the booster stage that
    # other runs read) with a model fitted under a different training mask
    res = m0.run(cfg=dict(cfg, name=f"{cfg['name']}_aux"), bootstrap=0, verbose=False, compare_baselines=False)
    lengths = cloud.bar_lengths(cfg)
    rows = joint.evaluate_lomo(cfg, matches, res["bar_source"], res["columns"], lengths, return_probs=True)
    for r in rows:
        np.save(paths[r["match"]], r["probs"])
    return {r["match"]: r["probs"] for r in rows}


def eval_probs(cfg, folder, extra=()):
    """Gün 7 (Ek Q.1): {match: booster probs} on the EVALUATION-corrupted DFL tracks
    (dfl_referee_eval): LOMO folds for the joint matches, the full model for `extra` (unseen)
    matches. The models are the ones lomo_probs / unseen_probs train (same config, same seeds);
    only the predicted features differ. Cached as <folder>/{lomo,unseen}_<m><tag><eval tag>.npy."""
    from ..validation.splits import cache_tag
    ecfg = dfl_referee.eval_cfg(cfg)
    assert ecfg is not None, "eval_probs needs dfl_referee_eval"
    tag = cache_tag(cfg) + dfl_referee.eval_tag(cfg)
    core = joint.available(cfg["joint"]["matches"])
    paths = {**{m: folder / f"lomo_{m}{tag}.npy" for m in core},
             **{m: folder / f"unseen_{m}{tag}.npy" for m in extra}}
    if all(p.is_file() for p in paths.values()):
        return {m: np.load(p) for m, p in paths.items()}
    res = m0.run(cfg=dict(cfg, name=f"{cfg['name']}_aux"), bootstrap=0, verbose=False, compare_baselines=False)
    lengths = cloud.bar_lengths(cfg)
    out = {}
    for r in joint.evaluate_lomo(cfg, core, res["bar_source"], res["columns"], lengths, return_probs=True, ecfg=ecfg):
        out[r["match"]] = r["probs_eval"]
    if extra:
        assert not set(extra) & set(cfg["joint"]["matches"]), "unseen matches must not be training matches"
        sources = joint.training_sources(cfg, res["bar_source"], cfg["joint"]["matches"], lengths)
        preps = [dfl_referee.prepare(m, ecfg, lengths) for m in extra]
        per_target, _, _ = engine.fit_predict_sources(sources, [p["feats"] for p in preps], cfg, res["columns"])
        for m, per_seed in zip(extra, per_target):
            out[m] = engine.ensemble(per_seed)
    for m, p in out.items():
        np.save(paths[m], p)
    return out


def with_decoder(cfg, pm, ps, pt, dm, dw):
    c = copy.deepcopy(cfg)
    d = c["decoder"]
    d["duration"]["possession"]["median"] = pm
    d["duration"]["possession"]["sigma"] = ps
    d["penalty_turnover"] = pt
    d["duration"]["dead"]["median"] = dm
    d["emission_weights"] = [1.0, 1.0, dw]
    return c


def evaluate(c, dfl, bar_probs, bar_ctx, blocks):
    spells, macros = [], []
    for m, (probs, prep, kick) in dfl.items():
        pred = engine.decode_runs(probs, prep["runs"], c)
        pred = post.apply(pred, prep["runs"], c, kick, prep["corrupted"].ball, prep["data"].pitch)
        s = dfl_referee.score(pred, prep, None)
        spells.append(s["spell_f1"])
        macros.append(s["macro_f1"])
    lab = m0.decode_match(bar_probs, c)
    lab = post.apply(lab, list(C.HALVES.values()), c, bar_ctx["kick"], bar_ctx["ball"], bar_ctx["pitch"])
    pub = score_blocks(lab, blocks)["ALL"]
    st = sanity.stats(lab)
    return {"dfl_spell": float(np.mean(spells)), "dfl_macro": float(np.mean(macros)),
            "dfl_spell_per_match": [round(x, 4) for x in spells],
            "bar_poss": st["n_possessions"], "bar_dead": st["dead_share"], "bar_home": st["home_share"],
            "pub_spell_diag": pub["spell_f1_mean"], "pub_macro_diag": pub["macro_f1"]}


def run(config="main", grid=GRID):
    t0 = time.time()
    cfg = config_mod.load(config)
    folder = ensure(OUTPUTS / "decoder" / config)
    probs = lomo_probs(cfg, folder)
    lengths = cloud.bar_lengths(cfg)
    dfl = {}
    for m, p in probs.items():
        prep = dfl_referee.prepare(m, cfg, lengths)
        kfe = build_features(prep["corrupted"], prep["cfg"], None, groups=["kin"], cache_key=prep["key"])
        dfl[m] = (p, prep, kfe["kin_kick"].to_numpy() > 0)
    bar = bar_data(cfg)
    bk = build_features(bar, cfg, load_tags(), groups=["kin"], cache_key=bar.name)
    bar_ctx = {"kick": bk["kin_kick"].to_numpy() > 0, "ball": bar.ball, "pitch": bar.pitch}
    bar_probs = np.load(OUTPUTS / f"{cfg['name']}_lgbm" / "probs.npy")
    blocks = public_blocks()               # 'public' = organisers' window only
    print(f"LOMO probs for {list(probs)} ready ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    base = evaluate(cfg, dfl, bar_probs, bar_ctx, blocks)
    rows.append({"name": "current", **base})
    print(f"current decoder: {base}", flush=True)
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        c = with_decoder(cfg, *vals)
        r = evaluate(c, dfl, bar_probs, bar_ctx, blocks)
        rows.append({"name": "grid", **dict(zip(keys, vals)), **r})
    grid_rows = [r for r in rows if r["name"] == "grid"]
    best = max(grid_rows, key=lambda r: (r["dfl_spell"], r["dfl_macro"]))
    print(f"best grid (by DFL LOMO spellF1): {best}  ({time.time() - t0:.0f}s)", flush=True)

    best_cfg = with_decoder(cfg, *[best[k] for k in keys])
    post_rows = []
    for close, snap in ((False, False), (True, False), (False, True), (True, True)):
        for dmax in ((125,) if close else (0,)):
            c = copy.deepcopy(best_cfg)
            c["post"] = {"close_short_dead": close, "short_dead_max_frames": dmax, "snap": snap,
                         "snap_before": 20, "snap_after": 15}
            r = evaluate(c, dfl, bar_probs, bar_ctx, blocks)
            post_rows.append({"close_short_dead": close, "snap": snap, **r})
            print(f"post close={close} snap={snap}: {r}", flush=True)
    out = {"config": config, "current": base, "best_grid": best, "post": post_rows,
           "grid_top10": sorted(grid_rows, key=lambda r: -r["dfl_spell"])[:10]}
    (folder / "decoder_tune.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {folder / 'decoder_tune.json'} ({time.time() - t0:.0f}s)")
    return out


def sweep_probs(config, run_dir, turnover=(14.0, 18.0, 22.0, 26.0, 32.0, 40.0),
                poss_min=(12, 25, 50, 75), poss_median=(250, 350), poss_band=(246, 260)):
    """Decoder sweep on the FIXED probabilities of a finished run (outputs/<run_dir>):
    probs.npy (BAR-ATH) + dfl_lomo_blend.npz (DFL leave-one-match-out). Selection (rule 3):
    BAR full-match possessions inside `poss_band` and dead share inside the config's bands ->
    best DFL LOMO spellF1 on the 35-aligned GT. The public score is reported, never used."""
    from ..labels.align import suppress_short_returns
    from ..validation.official import pm
    from .ablation import _in_bands, official_reference, reference_bands
    cfg = config_mod.load(config)
    run = OUTPUTS / run_dir
    bar_probs = np.load(run / "probs.npy")
    dfl_npz = np.load(run / "dfl_lomo_blend.npz")
    lengths = cloud.bar_lengths(cfg)
    preps, gt35, kicks = {}, {}, {}
    for m in dfl_npz.files:
        prep = dfl_referee.prepare(m, cfg, lengths)
        preps[m] = prep
        gt35[m] = np.where(prep["data"].period > 0,
                           suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                           C.X).astype(np.int8)
        kfe = build_features(prep["corrupted"], prep["cfg"], None, groups=["kin"], cache_key=prep["key"])
        kicks[m] = kfe["kin_kick"].to_numpy() > 0
    bar = bar_data(cfg)
    bk = build_features(bar, cfg, load_tags(), groups=["kin"], cache_key=bar.name)
    bar_kick = bk["kin_kick"].to_numpy() > 0
    blocks = public_blocks()               # 'public' = organisers' window only
    bands = reference_bands(cfg["acceptance"], None, official_reference(cfg)[1])

    def one(c, snap=False):
        sp35, sp11, mac = [], [], []
        for m in dfl_npz.files:
            pred = engine.decode_runs(dfl_npz[m], preps[m]["runs"], c)
            if snap:
                pred = post.snap_boundaries(pred, preps[m]["runs"], kicks[m],
                                            post.oob_entries(preps[m]["corrupted"].ball, preps[m]["data"].pitch))
            s = dfl_referee.score(pred, preps[m], None)
            sp11.append(s["spell_f1"])
            mac.append(s["macro_f1"])
            sp35.append(pm.evaluate(gt35[m], pred)["spell_f1_mean"])
        lab = m0.decode_match(bar_probs, c)
        if snap:
            lab = post.snap_boundaries(lab, list(C.HALVES.values()), bar_kick, post.oob_entries(bar.ball, bar.pitch))
        pub = score_blocks(lab, blocks)["ALL"]
        st = sanity.stats(lab)
        ok, _ = _in_bands(st, bands)
        return {"dfl_spell_gt35": float(np.mean(sp35)), "dfl_spell_gt11": float(np.mean(sp11)),
                "dfl_macro": float(np.mean(mac)), "bar_poss": st["n_possessions"], "bar_dead": st["dead_share"],
                "bar_home": st["home_share"], "in_bands": ok, "pub_spell": pub["spell_f1_mean"],
                "pub_macro": pub["macro_f1"]}

    rows = []
    for pt, pmin, pmed in itertools.product(turnover, poss_min, poss_median):
        c = copy.deepcopy(cfg)
        c["decoder"]["penalty_turnover"] = pt
        c["decoder"]["duration"]["possession"]["min"] = pmin
        c["decoder"]["duration"]["possession"]["median"] = pmed
        rows.append({"penalty_turnover": pt, "poss_min": pmin, "poss_median": pmed, **one(c)})
    ok_rows = [r for r in rows if poss_band[0] <= r["bar_poss"] <= poss_band[1] and r["in_bands"]]
    pool = ok_rows or rows
    best = max(pool, key=lambda r: (r["dfl_spell_gt35"], r["dfl_macro"]))
    c = copy.deepcopy(cfg)
    c["decoder"]["penalty_turnover"] = best["penalty_turnover"]
    c["decoder"]["duration"]["possession"]["min"] = best["poss_min"]
    c["decoder"]["duration"]["possession"]["median"] = best["poss_median"]
    snapped = {**{k: best[k] for k in ("penalty_turnover", "poss_min", "poss_median")}, "snap": True, **one(c, snap=True)}
    out = {"run": run_dir, "n_grid": len(rows), "n_in_band": len(ok_rows), "best": best,
           "best_snapped": snapped, "grid": rows,
           "current": [r for r in rows if r["penalty_turnover"] == cfg["decoder"]["penalty_turnover"]
                       and r["poss_min"] == cfg["decoder"]["duration"]["possession"]["min"]
                       and r["poss_median"] == cfg["decoder"]["duration"]["possession"]["median"]]}
    (run / "decoder_sweep.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def draw_probs(cfg, folder, alt_cfg, extra=()):
    """{match: booster probs} on the DFL tracks corrupted by `alt_cfg` (another corruption draw):
    LOMO folds for the joint matches, the full model for `extra` (unseen) matches -- as eval_probs,
    but for any corruption config. Cached as <folder>/{lomo,unseen}_<m><tag>_d<digest>.npy."""
    from ..validation.splits import cache_tag
    tag = cache_tag(cfg) + "_d" + config_mod.digest(dfl_referee.referee_cfg(alt_cfg))[:8]
    core = joint.available(cfg["joint"]["matches"])
    paths = {**{m: folder / f"lomo_{m}{tag}.npy" for m in core},
             **{m: folder / f"unseen_{m}{tag}.npy" for m in extra}}
    if all(p.is_file() for p in paths.values()):
        return {m: np.load(p) for m, p in paths.items()}
    res = m0.run(cfg=dict(cfg, name=f"{cfg['name']}_aux"), bootstrap=0, verbose=False, compare_baselines=False)
    lengths = cloud.bar_lengths(cfg)
    out = {}
    for r in joint.evaluate_lomo(cfg, core, res["bar_source"], res["columns"], lengths, return_probs=True, ecfg=alt_cfg):
        out[r["match"]] = r["probs_eval"]
    if extra:
        assert not set(extra) & set(cfg["joint"]["matches"]), "unseen matches must not be training matches"
        sources = joint.training_sources(cfg, res["bar_source"], cfg["joint"]["matches"], lengths)
        preps = [dfl_referee.prepare(m, alt_cfg, lengths) for m in extra]
        per_target, _, _ = engine.fit_predict_sources(sources, [p["feats"] for p in preps], cfg, res["columns"])
        for m, per_seed in zip(extra, per_target):
            out[m] = engine.ensemble(per_seed)
    for m, pr in out.items():
        np.save(paths[m], pr)
    return out


def unseen_probs(cfg, matches, folder, preps=None):
    """{match: probs} of the config's full model (BAR-ATH + its own DFL matches) on DFL matches it
    never trained on -- genuinely out-of-sample, so they can join the LOMO probabilities as
    training inputs of a downstream model. Cached as <folder>/unseen_<m>.npy. `preps` (same
    order as matches) lets other sources (e.g. Metrica open games) reuse it."""
    from ..validation.splits import cache_tag
    tag = cache_tag(cfg)
    paths = {m: folder / f"unseen_{m}{tag}.npy" for m in matches}
    if all(p.is_file() for p in paths.values()):
        return {m: np.load(p) for m, p in paths.items()}
    assert not set(matches) & set(cfg["joint"]["matches"]), "unseen matches must not be training matches"
    # train in a scratch run folder: never overwrite outputs/<name>_lgbm (the booster stage that
    # other runs read) with a model fitted under a different training mask
    res = m0.run(cfg=dict(cfg, name=f"{cfg['name']}_aux"), bootstrap=0, verbose=False, compare_baselines=False)
    lengths = cloud.bar_lengths(cfg)
    sources = joint.training_sources(cfg, res["bar_source"], cfg["joint"]["matches"], lengths,
                                     exclude_metrica=joint.metrica_games_in(matches))
    if preps is None:
        preps = [dfl_referee.prepare(m, cfg, lengths) for m in matches]
    per_target, _, _ = engine.fit_predict_sources(sources, [p["feats"] for p in preps], cfg, res["columns"])
    out = {}
    for m, per_seed in zip(matches, per_target):
        out[m] = engine.ensemble(per_seed)
        np.save(paths[m], out[m])
    return out


def _spells(codes, lo, hi):
    """Maximal H / A runs of codes[lo:hi] as (start, stop, team)."""
    c = np.asarray(codes[lo:hi])
    out, i = [], 0
    while i < len(c):
        j = i
        while j < len(c) and c[j] == c[i]:
            j += 1
        if c[i] in (C.H, C.A):
            out.append((lo + i, lo + j, int(c[i])))
        i = j
    return out


def short_recall(gt, pred, lo, hi, max_frames=75, iou=0.5):
    """(recalled, total): GT possessions <= max_frames inside [lo, hi) whose best same-team
    predicted possession reaches temporal IoU >= iou (a diagnostic, not the metric)."""
    ps = _spells(pred, lo, hi)
    hit = tot = 0
    for s, e, k in _spells(gt, lo, hi):
        if e - s > max_frames:
            continue
        tot += 1
        best = 0.0
        for a, b, kk in ps:
            if kk != k or b <= s or a >= e:
                continue
            best = max(best, (min(b, e) - max(a, s)) / (max(b, e) - min(a, s)))
        hit += best >= iou
    return hit, tot


def joint_sweep(config, run_dir, betas=(0.8, 1.0, 1.25, 1.5), turnovers=(8.0, 10.0, 11.0, 12.0, 14.0),
                gammas=(0.0, 0.5, 1.0), poss_band=(240, 290), dead_band=(0.43, 0.46), log=print):
    """Gün 3 / Adım 1: emission scale (beta) x turnover penalty x boundary bonus (gamma) on the
    FIXED ensemble probabilities of a finished run. Referees: DFL LOMO (35-aligned GT; core 4 and
    all 7 matches), Metrica open games the run never trained on (metrica_eval_*.npz, event GT)
    and full-match plausibility (possessions, dead share). No BAR-ATH label is read: hand labels
    are retired and the public window is not a tuning set (upstream rules)."""
    from ..labels.align import suppress_short_returns
    from ..validation.official import pm
    from . import metrica_src
    from .adapt import blend_cfg
    t0 = time.time()
    cfg = blend_cfg(config_mod.load(config))     # 'current' = the config's blend decoder
    run = OUTPUTS / run_dir

    def opt(fname):
        f = run / fname
        return (np.load(f) if f.is_file() else None)

    bar_probs = np.load(run / "probs.npy")
    bar_bnd, dfl, dfl_b = opt("bar_bnd_adapted.npy"), np.load(run / "dfl_lomo_blend.npz"), opt("dfl_lomo_bnd.npz")
    met, met_b = opt("metrica_eval_blend.npz"), opt("metrica_eval_bnd.npz")
    if bar_bnd is None and any(g != 0.0 for g in gammas):
        log("  no boundary probabilities in this run -> gamma fixed to 0")
        gammas = (0.0,)
    lengths = cloud.bar_lengths(cfg)
    refs = {}                                        # name -> (probs, bnd, runs, gt, group)

    def metrica_ref(m, probs, bnd):
        prep = metrica_src.prepare(int(m.split("_")[1]), cfg)
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        refs[m] = (probs, bnd, prep["runs"], gt, "metrica")

    for m in dfl.files:
        bnd = None if dfl_b is None else dfl_b[m]
        if m.startswith("metrica_"):
            metrica_ref(m, dfl[m], bnd)
            continue
        prep = dfl_referee.prepare(m, cfg, lengths)
        gt = np.where(prep["data"].period > 0,
                      suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                      C.X).astype(np.int8)
        refs[m] = (dfl[m], bnd, prep["runs"], gt, "dfl_core" if m in cfg["joint"]["matches"] else "dfl_extra")
    if met is not None:
        for m in met.files:
            metrica_ref(m, met[m], None if met_b is None else met_b[m])
    log(f"  referees: {sorted(refs)} | boundary probs "
        f"{'yes' if bar_bnd is not None else 'no'} ({time.time() - t0:.0f}s)")

    def gmean(sp, pred):
        vals = [v[0] for v in sp.values() if pred(v[2])]
        return float(np.mean(vals)) if vals else float("nan")

    def one(c):
        sp = {}
        for m, (pr, bd, runs, gt, grp) in refs.items():
            r = pm.evaluate(gt, engine.decode_runs(pr, runs, c, boundary=bd))
            sp[m] = (r["spell_f1_mean"], r["macro_f1"], grp)
        lab = m0.decode_match(bar_probs, c, bar_bnd)
        st = sanity.stats(lab)
        row = {"dfl_core": gmean(sp, lambda g: g == "dfl_core"), "dfl_all": gmean(sp, lambda g: g.startswith("dfl")),
               "metrica": gmean(sp, lambda g: g == "metrica"),
               "poss": st["n_possessions"], "dead": st["dead_share"], "home": st["home_share"],
               "per_match": {m: round(v[0], 4) for m, v in sp.items()}}
        row["in_band"] = bool(poss_band[0] <= row["poss"] <= poss_band[1]
                              and dead_band[0] <= row["dead"] <= dead_band[1])
        used = [row["dfl_core"]] + ([] if np.isnan(row["metrica"]) else [row["metrica"]])
        row["combined"] = float(np.mean(used))
        return row, lab

    rows, labs = [], {}
    for beta, pt, gamma in itertools.product(betas, turnovers, gammas):
        c = copy.deepcopy(cfg)
        c["decoder"]["emission_scale"] = beta
        c["decoder"]["penalty_turnover"] = pt
        c["decoder"]["boundary_gamma"] = gamma
        row, lab = one(c)
        row = {"beta": beta, "turnover": pt, "gamma": gamma, **row}
        rows.append(row)
        labs[(beta, pt, gamma)] = lab
        log(f"  b {beta:<4} t {pt:<4} g {gamma:<3} | DFL core {row['dfl_core']:.4f} all {row['dfl_all']:.4f} "
            f"| Met {row['metrica']:.4f} | poss {row['poss']} D {row['dead']:.3f} "
            f"| comb {row['combined']:.4f}{'' if row['in_band'] else '  (out of band)'} [{time.time() - t0:.0f}s]")
    dc = cfg["decoder"]
    cur_key = (float(dc.get("emission_scale", 1.0)), float(dc["penalty_turnover"]), float(dc.get("boundary_gamma", 0.0)))
    cur = next((r for r in rows if (r["beta"], r["turnover"], r["gamma"]) == cur_key), None)
    pool = [r for r in rows if r["in_band"]] or rows
    best = max(pool, key=lambda r: (r["combined"], r["dfl_core"]))
    bkey = (best["beta"], best["turnover"], best["gamma"])
    paired = None
    if cur is not None and bkey != cur_key:
        paired = {"matches_better": int(sum(best["per_match"][m] > cur["per_match"][m] for m in best["per_match"])),
                  "matches_worse": int(sum(best["per_match"][m] < cur["per_match"][m] for m in best["per_match"])),
                  "matches_total": len(best["per_match"])}
    out = {"run": run_dir, "grid": rows, "current": cur, "best": best, "paired": paired,
           "bands": {"poss": poss_band, "dead": dead_band}, "seconds": round(time.time() - t0)}
    (run / "decoder_joint_sweep.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log(f"-> {run / 'decoder_joint_sweep.json'} ({time.time() - t0:.0f}s)")
    return out
