"""BAR-ATH end-to-end run (M0 and every later variant: feature groups come from the config).

tracking -> feature groups (cached) -> weak labels + ball-masked copies (built once) ->
n-seed two-stage LightGBM (only the models are refitted per seed) -> seed-averaged probabilities
-> 3-state ED-HSMM per graded half (+ forced dead before the 2nd-half kick-off)
-> submission.csv -> official checks, validation-block scores, full-match sanity gates.
Each seed is also decoded on its own, so the run reports its own seed spread.

Validation discipline: training uses algorithmic weak labels only; no BAR-ATH label (hand or
public) is ever fitted. The held-out blocks of the config (region fold k, or the public window)
are excluded together with a `validation.buffer_frames` margin. The public score printed here
is a format / sanity diagnostic, never a selection criterion.
"""

import datetime as dt
import json
import time

import numpy as np
import pandas as pd

from .. import config as config_mod
from .. import constants as C
from ..features import build_features
from ..io import submission
from ..io.broadcast import bar_data
from ..io.tags import load_tags
from ..labels import weak
from ..labels.masking import gap_lengths
from ..models.rules import rule_probs
from ..models.two_stage import to_probs
from ..paths import BASELINE_RESULTS, OUTPUTS, ensure
from ..validation import sanity
from ..validation.official import pm
from ..validation.scoring import report, score_blocks
from ..validation.splits import (labelled_blocks, merged_gt, public_blocks, training_blocks, training_mask,
                                 trains_on_blocks)
from . import engine


def _log(msg, t0, verbose=True):
    if verbose:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def joint_columns(cfg, feats):
    """Columns of the joint model: all BAR-ATH columns (source-specific ones are NaN on other
    sources) unless joint.drop_source_specific removes them."""
    if not cfg.get("joint", {}).get("drop_source_specific", False):
        return feats.columns
    from ..features import common_groups
    return [c for g in common_groups(feats.attrs["groups"]) for c in feats.attrs["group_columns"][g]]


def bar_source(cfg, data, feats, anchors, team_ok, train_ok, tags, lengths):
    return engine.TrainSource("barath", data, feats, anchors, team_ok, train_ok, data.name, lengths,
                              tags, int(cfg["lgbm"]["row_stride"]),
                              d_share=cfg["weak_labels"].get("d_weight_share"))


def decode_match(probs, cfg, boundary=None, contested=None, regime=None):
    """Full-length labels: decoded H/A/D on graded halves, X elsewhere (never scored)."""
    return engine.decode_runs(probs, list(C.HALVES.values()), cfg, C.FORCED_DEAD, C.N_FRAMES, boundary, contested, regime)


def baseline_scores(blocks):
    out = {}
    for name in ("baseline_smooth50", "baseline_smooth125"):
        table = pm.read_label_csv(str(BASELINE_RESULTS / f"{name}.csv"))
        out[name] = score_blocks(pm.label_codes(table), blocks)["ALL"]
    return out


def log_experiment(summary):
    """Append one row to outputs/experiments.csv (columns are the union over all runs)."""
    path = OUTPUTS / "experiments.csv"
    row = pd.DataFrame([summary])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False, lineterminator="\n")


def run(emission="lgbm", cfg_name="m0", bootstrap=1000, cfg=None, verbose=True, compare_baselines=True):
    """Train, decode, write the submission and score it. Returns labels, scores and seed spread."""
    t0 = time.time()
    cfg = cfg if cfg is not None else config_mod.load(cfg_name)
    from .. import cloud
    if cloud.enabled():
        # no provided-game data in cloud mode: callers (decoder_tune) only need the feature columns; the
        # booster folds are trained on DFL by joint.evaluate_lomo / unseen_probs
        cloud.require_external(cfg)
        assert emission == "lgbm"
        return {"bar_source": None, "columns": cloud.external_columns(cfg, cloud.bar_lengths(cfg)), "cloud": True}
    groups = list(cfg["features"]["groups"])
    n_seeds = int(cfg["lgbm"].get("n_seeds", 1))
    run_name = f"{cfg['name']}_{emission}"
    out_dir = ensure(OUTPUTS / run_name)

    data = bar_data(cfg)
    tags = load_tags()
    feats = build_features(data, cfg, tags, cache_key=data.name)
    _log(f"features {feats.shape} groups {groups}", t0, verbose)

    blocks = labelled_blocks()
    gt_val = merged_gt(blocks)
    train_ok = training_mask(training_blocks(cfg, blocks), cfg["validation"]["buffer_frames"]) & (data.period > 0)
    info = {}
    anchors, team_ok = weak.anchors(feats, data.period, cfg, C.FORCED_DEAD, data=data, info=info)
    if verbose:
        print(f"  validation blocks {[(b.name, b.start, b.stop) for b, _ in blocks]}; "
              f"training-eligible frames {int(train_ok.sum())}")
        print(f"  anchor sources (frames): {info}")
        print("  weak-label anchors on the public window (diagnostic only):")
        print(weak.precision_report(anchors, gt_val))

    imp, per_seed = None, None
    if emission == "lgbm":
        lengths = gap_lengths(data)
        bar = bar_source(cfg, data, feats, anchors, team_ok, train_ok, tags, lengths)
        sources = [bar]
        external = bool(cfg.get("joint", {}).get("external_only"))
        if cfg.get("joint", {}).get("matches"):
            from . import joint
            sources = joint.training_sources(cfg, bar, cfg["joint"]["matches"], lengths)
        if external:                    # never train on the provided game; BAR is only predicted
            from ..features import common_groups
            assert all(s.name != "barath" for s in sources) and cfg["joint"].get("matches")
            bar = None
            columns = [c for g in common_groups(feats.attrs["groups"]) for c in feats.attrs["group_columns"][g]]
        else:
            columns = joint_columns(cfg, feats) if len(sources) > 1 else feats.columns
        _log(f"training sources {[s.name for s in sources]}; fitting {n_seeds} seed(s) "
             f"(masking per seed: {bool(cfg['masking'].get('per_seed', False))})", t0, verbose)
        (per_seed,), imp, counts = engine.fit_predict_sources(sources, [feats], cfg, columns)
        _log(f"training rows per source (last seed): {counts}", t0, verbose)
        probs = engine.ensemble(per_seed)
        if verbose:
            print("  top features  alive:", ", ".join(n for n, _ in imp["alive"][:8]))
            print("                team :", ", ".join(n for n, _ in imp["team"][:8]))
    elif emission == "rules":
        probs = to_probs(*rule_probs(feats))
    else:
        raise ValueError(f"unknown emission {emission}")
    np.save(out_dir / "probs.npy", probs)

    labels = decode_match(probs, cfg)
    sub_path = submission.write(out_dir / "submission.csv", labels)
    reason = submission.validate(sub_path)
    _log(f"submission {sub_path}  official format check: {'OK' if reason is None else reason}", t0, verbose)

    pub_blocks = public_blocks()          # 'public' = the organisers' window only (hand blocks apart)
    results = score_blocks(labels, pub_blocks, bootstrap=bootstrap)
    stats, ok, text = sanity.check(labels)

    seed_rows = []
    if per_seed is not None and len(per_seed) > 1:
        for i, (pa, ph) in enumerate(per_seed):
            lab_i = decode_match(to_probs(pa, ph), cfg)
            r_i = score_blocks(lab_i, pub_blocks)["ALL"]
            fams = [c.get("model", {}).get("family", "lgbm") for c in engine.family_cfgs(cfg)]
            seed_rows.append({"seed": engine.model_seed(cfg, i // len(fams)), "family": fams[i % len(fams)], "spell_f1": r_i["spell_f1_mean"],
                              "macro_f1": r_i["macro_f1"], **sanity.stats(lab_i)})

    if verbose:
        print()
        print(report(results, f"VALIDATION ({run_name}; {n_seeds}-seed ensemble; official metric, out-of-sample)"))
        if seed_rows:
            sp = [s["spell_f1"] for s in seed_rows]
            print(f"  single-seed spellF1 range {min(sp):.4f}..{max(sp):.4f} (n={len(sp)})")
        if compare_baselines:
            print(report(baseline_scores(pub_blocks), "reference baselines on the same blocks"))
        print(text)

    r = results["ALL"]
    summary = {
        "time": dt.datetime.now().isoformat(timespec="seconds"), "run": run_name,
        "groups": "+".join(groups), "n_features": feats.shape[1], "n_seeds": n_seeds,
        "config_hash": config_mod.digest(cfg), "emission": emission,
        "joint": "+".join(cfg.get("joint", {}).get("matches", [])) or None,
        "weak_d": "+".join(k for k in ("d_tempo", "d_static_ball", "d_tags_tempo")
                           if cfg["weak_labels"].get(k)) or None,
        "mask_per_seed": bool(cfg["masking"].get("per_seed", False)),
        "trains_on_public_frames": trains_on_blocks(cfg),
        "spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
        "boot_low": r.get("spell_bootstrap_low"), "boot_high": r.get("spell_bootstrap_high"),
        "n_val_spells": r["n_gt_spells"], "n_pred_val_spells": r["n_pred_spells"],
        "d_precision": r["per_class"]["D"]["precision"], "d_recall": r["per_class"]["D"]["recall"],
        **{f"sanity_{k}": v for k, v in stats.items()}, "sanity_ok": ok,
    }
    if seed_rows:
        for key in ("spell_f1", "macro_f1", "dead_share", "n_possessions", "home_share"):
            vals = np.array([s[key] for s in seed_rows], float)
            summary[f"seeds_{key}_mean"] = float(vals.mean())
            summary[f"seeds_{key}_std"] = float(vals.std(ddof=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "seeds.json").write_text(json.dumps(seed_rows, indent=2), encoding="utf-8")
    log_experiment(summary)
    _log(f"done -> {out_dir}", t0, verbose)
    return {"summary": summary, "labels": labels, "probs": probs, "results": results,
            "stats": stats, "gt": gt_val, "importance": imp, "seeds": seed_rows,
            "bar_source": bar if emission == "lgbm" else None,
            "columns": list(columns) if emission == "lgbm" else None, "anchor_info": info}
