"""BAR-ATH domain adaptation of the temporal net (Gün 2): fine-tune the DFL-trained TCN+BiGRU on
BAR-ATH weak anchors, then re-blend with the boosting probabilities.

Leakage discipline
  * BAR-ATH input probabilities for training are TIME-BLOCK OUT-OF-FOLD: the config's booster is
    refitted K times (graded halves split into K contiguous blocks), each time without the
    block +- buffer, and predicts only that block. The final (all-data) probabilities are used
    only at inference, exactly as on DFL (LOMO in, full model out).
  * BAR anchors honour the training mask (region fold k, or the public window when held out)
    +- 1,125 frames never enters training.
  * DFL LOMO of the adapted net: fold m fine-tunes the models trained without m on the other DFL
    matches + BAR anchors; m is never seen.
Balancing: every training window carries a weight; BAR windows together get `bar_share` of the
known-step weight (the DFL real labels the rest), and no boundary loss (sparse anchors carry no
reliable change targets). Fine-tuning: lr 1e-4, `ft_epochs` epochs, same seeds.
"""

import datetime as dt
import json
import time

import numpy as np

from .. import config as config_mod
from .. import constants as C
from ..features import build_features, common_groups
from ..io import submission
from ..io.broadcast import bar_data
from ..io.tags import load_tags
from ..labels import weak
from ..labels.masking import gap_lengths
from ..paths import OUTPUTS, ensure
from ..validation import sanity
from ..validation.compare import paired_spell_bootstrap
from ..validation.official import pm
from ..validation.scoring import score_blocks
from ..validation.splits import (labelled_blocks, merged_gt, public_blocks, training_blocks, training_mask,
                                 trains_on_blocks)
from ..decoding import contact
from . import decoder_tune, dfl_referee, engine, joint, m0
from .ablation import _in_bands, official_reference, reference_bands
from .hybrid import (FINE_DIM, TOKEN_SHAPE, fine_view, log_pool, net_boundary, net_probs, player_tokens, robust_scale,
                     step_view, with_tokens)
from ..labels.align import controlled_possession, suppress_short_returns

DEFAULTS = dict(k_folds=4, bar_share=0.3, ft_epochs=8, ft_lr=1e-4, min_votes=3, w_boost=0.25,
                d_share=None, bar_team=True, dfl_target="gt11", extra_matches=(), metrica_games=(),
                metrica_eval=(), decoder=None, short_team=None, eval_matches=(), booster_source=None,
                segment_diagnostics=False, external_team_copies=None, tta_swap=None, bar_swap_anchors=False, metrica_corrupt=False,
                metrica_ball_mask=0.0, draw_overrides=None, metrica_raw_train=True, metrica_repeat=1,
                skillcorner_games=(), skillcorner_stride=1, skillcorner_team=1.0, extra_draws=(),
                ctrl=dict(max_frames=100, min_frames=19, before_dead=False, dribble_frames=25))
# dfl_target 'ctrl' (Gün 3): DFL labels turned into Metrica-like CONTROLLED possession by
# labels.align.controlled_possession with the `ctrl` parameters, calibrated on Metrica's
# event-derived statistics (DFL 7 matches: 351 possessions / alive hour, median 6.1 s; Metrica
# 343, 5.75 s; gt35 423, 4.8 s). LOMO is still scored on gt35 (standard) and also on ctrl.
# decoder (Gün 3, user decision 23 Oct): overrides of cfg['decoder'] used ONLY to decode the
# blended booster + net probabilities (LOMO scores, BAR submission, seeds, Metrica eval), e.g.
# {emission_scale: 1.75, boundary_gamma: 2.0}; the boundary bonus reads the net's boundary head.
# The booster stage (outputs/<name>_lgbm) keeps the base decoder.
# metrica_eval (Gün 3): Metrica open games used ONLY as a referee: the final models (all DFL
# matches + BAR anchors, never Metrica) predict them -> outputs/<name>/metrica_eval_{blend,bnd}.npz.
# metrica_games: Metrica open sample games (1, 2, 3) added as training sources AND as extra
# leave-one-out folds (a Metrica-semantics referee); see pipeline/metrica_src.py.
# extra_matches: further DFL matches the config's booster NEVER trained on; their booster input
# is the full model's (out-of-sample) prediction (decoder_tune.unseen_probs). They add training
# data for the net and folds to its leave-one-match-out evaluation.
# dfl_target (autonomous loop): the DFL labels the net is TRAINED on -- "gt11" (training
# alignment, <= 11-frame returns merged) or "gt35" (Metrica-like: <= 35-frame returns merged
# unless the short spell shows control). The boosting LOMO inputs are unchanged.
# bar_team (autonomous loop, 22 Oct): weight lambda of the TEAM part of the BAR windows' loss
# (True = 1: full 3-class loss; False = 0: dead vs alive only; a float in between mixes them).
# lambda 1 -> 293 possessions (sparse team anchors teach fast flips), lambda 0 -> 152 (team
# dynamics collapse on BAR): a GT-free possession knob.
# tta_swap (Rapor 9 Sıra 1, 1 Kasım): test-time team-exchange average of the adapted net, before the
# decoder. "report": outputs unchanged, the averaged scores go to summary["tta"] (an exactly paired,
# seed-noise-free comparison: the same models predict both views). "apply": the averaged
# probabilities / boundaries become the outputs (probs.npy, submission, npz), summary["no_tta"]
# keeps the plain ones. The booster is not exchanged (its input to the net is permuted, as in training).
# bar_swap_anchors (Rapor 9 Sıra 1b, 1 Kasım): the BAR weak-anchor windows also enter fine-tuning as their
# team-exchanged twins (features recomputed, OOF booster input permuted, H/A anchors swapped). The BAR
# share of the loss is unchanged (the twins split it); tested against the same seeds without twins.
# d_share (user decision, 22 Oct): weight of the D anchor steps inside the BAR windows, fixed to
# the reference anchor ratio 0.35 -- the net-side twin of weak_labels.d_weight_share. Without it
# the 82%-D anchor set shifted the prior (full-match dead share 0.49, outside the bands).


def bar_context(cfg):
    data = bar_data(cfg)
    tags = load_tags()
    feats = build_features(data, cfg, tags, cache_key=data.name)
    blocks = labelled_blocks()
    train_ok = training_mask(training_blocks(cfg, blocks), cfg["validation"]["buffer_frames"]) & (data.period > 0)
    anchors, team_ok = weak.anchors(feats, data.period, cfg, C.FORCED_DEAD, data=data)
    return dict(data=data, tags=tags, feats=feats, blocks=blocks, train_ok=train_ok,
                anchors=anchors, team_ok=team_ok)


def time_blocks(k):
    """K contiguous blocks over the graded halves (k/2 per half)."""
    out = []
    per = max(1, k // 2)
    for a, b in C.HALVES.values():
        edges = np.linspace(a, b, per + 1).astype(int)
        out += list(zip(edges[:-1], edges[1:]))
    return out


def bar_oof_probs(cfg, ctx, folder, k):
    from ..validation.splits import cache_tag
    path = folder / f"bar_oof_k{k}{cache_tag(cfg)}.npy"
    if path.is_file():
        return np.load(path)
    data, feats = ctx["data"], ctx["feats"]
    lengths = gap_lengths(data)
    dfl = joint.dfl_sources(cfg, cfg["joint"]["matches"], lengths) if cfg.get("joint", {}).get("matches") else []
    columns = m0.joint_columns(cfg, feats) if dfl else feats.columns
    buf = int(cfg["validation"]["buffer_frames"])
    source = (cfg.get("adapt") or {}).get("booster_source") or cfg["name"]
    oof = np.load(OUTPUTS / f"{source}_lgbm" / "probs.npy").copy()     # non-graded frames
    for a, b in time_blocks(k):
        excl = np.zeros(data.n_frames, bool)
        excl[max(0, a - buf):b + buf] = True
        bar = m0.bar_source(cfg, data, feats, ctx["anchors"], ctx["team_ok"], ctx["train_ok"] & ~excl,
                            ctx["tags"], lengths)
        if dfl:
            bar.share = joint.joint_cfg(cfg)["bar_share"]
        (per_seed,), _, _ = engine.fit_predict_sources([bar] + dfl, [feats], cfg, columns)
        oof[a:b] = engine.ensemble(per_seed)[a:b]
        print(f"    OOF block [{a}, {b}) done", flush=True)
    np.save(path, oof)
    return oof


def anchor_steps(feats, columns, probs, anchors, runs, stride, min_votes):
    """Like hybrid.step_view, but a step is labelled when >= min_votes of its frames carry the
    same anchor (sparse weak labels instead of dense truth)."""
    steps = step_view(feats, columns, probs, None, runs, stride)
    out = []
    for x, _, _, (a, b) in steps:
        n = len(x)
        blk = anchors[a:a + n * stride].reshape(n, stride)
        votes = np.stack([(blk == c).sum(1) for c in (C.H, C.A, C.D)], axis=1)
        y = votes.argmax(1).astype(np.int64)
        known = votes.max(1) >= min_votes
        out.append((x, y, known, (a, b)))
    return out


def params(cfg, over=None):
    """(adapt params, seqnet params): defaults < config < command-line overrides. A key may
    appear at several levels (e.g. dfl_target in configs/main.yaml and --dfl-target): the
    later level wins (dict(**a, **b) would raise on the duplicate)."""
    from ..models import seqnet
    ac = {**DEFAULTS, **cfg.get("adapt", {}), **(over or {})}
    p = {**seqnet.DEFAULTS, **cfg.get("seqnet", {}), **(ac.get("seqnet") or {})}
    return ac, p


def network_seeds(cfg, ac, p):
    """Change only network seeds; keep booster identity/cache and its seed untouched."""
    start = int(cfg["seed"]) + int(ac.get("net_seed_offset", 0))
    return [start + 1000 * i for i in range(int(p["n_seeds"]))]


def dfl_target(name, prep, gt35, gtc):
    """The DFL labels the net trains on: gt11 (training alignment, prep['aligned']), gt35,
    ctrl, or any gtN (Gün 3: A-B-A returns <= N frames merged unless B shows control, as gt35)."""
    if name == "gt35":
        return gt35
    if name == "ctrl":
        return gtc
    if name == "gt11":
        return prep["aligned"]
    if name.startswith("gt") and name[2:].isdigit():
        d = prep["data"]
        return np.where(d.period > 0, suppress_short_returns(d.labels, prep["runs"], int(name[2:]), d, True),
                        C.X).astype(np.int8)
    raise ValueError(f"unknown dfl_target {name!r}")


def blend_cfg(cfg, ac=None):
    """cfg with the blend-decoder overrides (adapt.decoder) applied."""
    import copy
    ac = ac if ac is not None else params(cfg)[0]
    c = copy.deepcopy(cfg)
    c["decoder"].update(ac.get("decoder") or {})
    return c


def touch_targets(steps, labels, stride):
    """Per-step touch-level targets for DFL step items: raw DFL labels at the step centre,
    H -> 0, A -> 1, D -> 2 (team_touch 'None'), X -> -1 (unknown)."""
    out = []
    for _, y, _, (a, _) in steps:
        n = len(y)
        lab = np.asarray(labels[a:a + n * stride]).reshape(n, stride)[:, stride // 2]
        out.append(np.where(lab == C.X, -1, lab).astype(np.int64))
    return out


def save_nets(folder, models, seeds, p, n_in, columns):
    """The final nets (one per seed) with what is needed to rebuild them: lets the provided game be
    predicted on another machine than the one that trained (cloud mode, tools/bar_from_nets.py)."""
    import torch
    ensure(folder)
    for sd, mdl in zip(seeds, models):
        if not isinstance(mdl, torch.nn.Module):    # test doubles
            continue
        torch.save({"seed": int(sd), "n_in": int(n_in), "p": dict(p), "columns": list(columns),
                    "state": {k: v.detach().cpu() for k, v in mdl.state_dict().items()}}, folder / f"net_{sd}.pt")


def mixed(dfl_steps, bar_steps, share, d_share=None, bar_team=True):
    """Training list with weights: BAR windows together get `share` of the known steps' weight;
    DFL windows keep weight 1 and their boundary loss. With d_share, BAR steps get per-step
    weights so that D anchors carry exactly d_share of the BAR weight (H/A the rest)."""
    n_dfl = sum(int(k.sum()) for _, _, k, _ in dfl_steps)
    n_bar = sum(int(k.sum()) for _, _, k, _ in bar_steps)
    w_bar = share / (1 - share) * n_dfl / max(n_bar, 1)
    lam = 1.0 if bar_team is True else (0.0 if bar_team is False else float(bar_team))
    team_flag = () if lam == 1.0 else (1.0,)
    dfl = [(x, y, k, 1.0, True) + team_flag for x, y, k, _ in dfl_steps]
    bar_flag = () if lam == 1.0 else (lam,)
    if d_share is None:
        return dfl + [(x, y, k, w_bar, False) + bar_flag for x, y, k, _ in bar_steps], w_bar
    n_d = sum(int((k & (y == C.D)).sum()) for _, y, k, _ in bar_steps)
    n_l = n_bar - n_d
    cw_d = d_share * n_bar / max(n_d, 1)            # mean step weight over BAR stays 1
    cw_l = (1 - d_share) * n_bar / max(n_l, 1)
    bar = [(x, y, k, (w_bar * np.where(y == C.D, cw_d, cw_l)).astype(np.float32), False) + bar_flag
           for x, y, k, _ in bar_steps]
    return dfl + bar, w_bar


def run(config="main", name="hybrid_adapted", over=None, cfg=None):
    """cfg: an already-built config dict (e.g. a region fold with another seed); `config` then
    only names its cache folder outputs/decoder/<config>."""
    from ..models import seqnet
    t0 = time.time()
    cfg = cfg if cfg is not None else config_mod.load(config)
    ac, p = params(cfg, over)
    dcfg = blend_cfg(cfg, ac)                # decoder for the blended probabilities
    split = p.get("heads", "joint") == "split"
    stride = int(p["stride"])
    use_tok = bool(p.get("use_tokens"))
    from ..models.player_set import token_view, SET_SHAPE
    tokens = lambda data, runs: token_view(data, cfg, runs, stride, p)
    if use_tok:
        p["token_shape"] = SET_SHAPE if p.get("token_pool") == "deepsets_team" else TOKEN_SHAPE
    use_fine = bool(p.get("fine"))           # Gün 4 / Sıra 3: 25 fps fine branch
    team_copies = ac.get("external_team_copies")
    if team_copies not in (None, "repeat", "swap"):
        raise ValueError("external_team_copies must be repeat, swap, or null")
    if team_copies and (split or use_fine or ac.get("short_team")):
        raise ValueError("Team-copy screening supports dense DFL joint-head training only")
    tta = ac.get("tta_swap")
    if tta not in (None, "report", "apply"):
        raise ValueError("tta_swap must be report, apply, or null")
    if tta and (split or use_fine or use_tok):
        raise ValueError("Test-time team exchange supports the dense joint-head net only")
    if tta:
        from ..augmentation import SWAP, swapped_steps, tta_boundary, tta_probs
    assert not (use_fine and use_tok), "fine branch and player tokens are not combined"
    if use_fine:
        p["fine_dim"] = FINE_DIM
    fv = (lambda feats, probs, labels, steps: fine_view(feats, probs, labels, steps, stride)) if use_fine \
        else (lambda feats, probs, labels, steps: None)
    out_dir = ensure(OUTPUTS / name)
    booster_source = ac.get("booster_source")
    if booster_source:
        from ..validation.splits import cache_tag
        if cache_tag(cfg) != cache_tag(config_mod.load(booster_source)):
            raise ValueError("booster_source must have exactly the same booster configuration")
    folder = ensure(OUTPUTS / "decoder" / (booster_source or config))
    from .. import cloud
    cloud_mode = cloud.enabled()
    if cloud_mode:
        # 5 Nov: rented GPU, no provided-game data at all (src/mtah/cloud.py); the provided game is
        # predicted locally from the saved nets (tools/bar_from_nets.py)
        cloud.require_external(cfg)
        p_boost, ctx = None, None
        lengths = cloud.bar_lengths(cfg)
        columns = cloud.external_columns(cfg, lengths)
    else:
        p_boost = np.load(OUTPUTS / f"{booster_source or cfg['name']}_lgbm" / "probs.npy")
        ctx = bar_context(cfg)
        lengths = gap_lengths(ctx["data"])
        groups = common_groups(cfg["features"]["groups"])
        columns = [c for g in groups for c in ctx["feats"].attrs["group_columns"][g]]

    # external_only (organisers, Nov 2026): no training on the provided game. No BAR OOF booster, no BAR
    # anchor windows, no fine-tuning: the "adapted" models ARE the DFL-trained models.
    external = bool(cfg.get("joint", {}).get("external_only"))
    if external and (ac.get("bar_swap_anchors") or use_tok or use_fine or split):
        raise ValueError("external_only supports the dense joint-head net without BAR anchor options")
    if external:
        print(f"[{time.time() - t0:5.0f}s] external_only: no BAR anchors, no BAR fine-tuning", flush=True)
        oof = None
        bar_anchor = None if cloud_mode else np.full(len(p_boost), C.X, np.int8)
        bar_steps = []
    else:
        print(f"[{time.time() - t0:5.0f}s] BAR time-block OOF probabilities (k={ac['k_folds']})", flush=True)
        oof = bar_oof_probs(cfg, ctx, folder, int(ac["k_folds"]))
        bar_anchor = np.where(ctx["train_ok"], ctx["anchors"], C.X)
        bar_steps = anchor_steps(ctx["feats"], columns, oof, bar_anchor, list(C.HALVES.values()), stride,
                                 int(ac["min_votes"]))
    if use_tok:
        bar_tokens = tokens(ctx["data"], list(C.HALVES.values()))
        bar_steps = with_tokens(bar_steps, bar_tokens)
    bar_fine = fv(ctx["feats"], oof, bar_anchor, bar_steps) if bar_steps else []
    if ac.get("bar_swap_anchors"):
        if use_tok or use_fine or split:
            raise ValueError("bar_swap_anchors supports the dense joint-head net only")
        from ..augmentation import SWAP, swap_labels, swapped_features
        runs_bar = list(C.HALVES.values())
        _, feats_sw = swapped_features(dict(corrupted=ctx["data"], key=ctx["data"].name, feats=ctx["feats"]), cfg, columns)
        bar_steps = bar_steps + anchor_steps(feats_sw, columns, oof[:, SWAP], swap_labels(bar_anchor), runs_bar,
                                             stride, int(ac["min_votes"]))
        print(f"[{time.time() - t0:5.0f}s] BAR anchors + team-exchanged twins: {len(bar_steps)} windows", flush=True)
    n_known = sum(int(k.sum()) for _, _, k, _ in bar_steps)
    if bar_steps:
        cls = np.concatenate([y[k] for _, y, k, _ in bar_steps])
        print(f"[{time.time() - t0:5.0f}s] BAR anchor steps {n_known} (H {np.mean(cls == C.H):.2f} "
              f"A {np.mean(cls == C.A):.2f} D {np.mean(cls == C.D):.2f})", flush=True)

    lomo = decoder_tune.lomo_probs(cfg, folder)
    if ac["extra_matches"]:
        lomo.update(decoder_tune.unseen_probs(cfg, list(ac["extra_matches"]), folder))
    matches = list(lomo)
    preps, dsteps, gt35, dtouch, gtc, dfine, targets = {}, {}, {}, {}, {}, {}, {}
    for m in matches:
        prep = dfl_referee.prepare(m, cfg, lengths)
        preps[m] = prep
        assert list(prep["feats"].columns) == columns
        gt35[m] = np.where(prep["data"].period > 0,
                           suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                           C.X).astype(np.int8)
        gtc[m] = np.where(prep["data"].period > 0,
                          controlled_possession(prep["data"].labels, prep["runs"], prep["data"], **ac["ctrl"]),
                          C.X).astype(np.int8)
        target = dfl_target(ac["dfl_target"], prep, gt35[m], gtc[m])
        dsteps[m] = step_view(prep["feats"], columns, lomo[m], target, prep["runs"], stride)
        targets[m] = target
        if use_tok:
            dsteps[m] = with_tokens(dsteps[m], tokens(prep["corrupted"], prep["runs"]))
        dtouch[m] = touch_targets(dsteps[m], prep["data"].labels, stride) if split else [None] * len(dsteps[m])
        dfine[m] = fv(prep["feats"], lomo[m], target, dsteps[m])
    for g in ac["metrica_games"]:
        # Metrica open games: raw Metrica tracking + event-derived labels (Metrica semantics);
        # the booster never saw them, so its prediction is out-of-sample (unseen_probs)
        from . import metrica_src
        m = f"metrica_{g}"
        prep = metrica_src.prepare(g, cfg)
        assert list(prep["feats"].columns) == columns
        preps[m] = prep
        targets[m] = prep["aligned"]
        lomo[m] = decoder_tune.unseen_probs(cfg, [m], folder, preps=[prep])[m]
        gt35[m] = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        dsteps[m] = step_view(prep["feats"], columns, lomo[m], prep["aligned"], prep["runs"], stride)
        if use_tok:
            dsteps[m] = with_tokens(dsteps[m], tokens(prep["data"], prep["runs"]))
        dtouch[m] = [None] * len(dsteps[m])          # Metrica events = controlled semantics only
        dfine[m] = fv(prep["feats"], lomo[m], prep["aligned"], dsteps[m])
        matches.append(m)
    # Kasım 2026 (external data only): SkillCorner open games (broadcast tracking, phase-of-play
    # labels) as TRAINING data only -- never a fold, never a referee (pipeline/skillcorner_src.py).
    if ac["skillcorner_games"]:
        from . import skillcorner_src
        sc_names = [skillcorner_src.name(g, cfg) for g in ac["skillcorner_games"]]
        sc_preps = [skillcorner_src.prepare(g, cfg) for g in ac["skillcorner_games"]]
        sc_probs = decoder_tune.unseen_probs(cfg, sc_names, folder, preps=sc_preps)
        for m, prep in zip(sc_names, sc_preps):
            assert list(prep["feats"].columns) == columns
            preps[m], targets[m], lomo[m] = prep, prep["aligned"], sc_probs[m]
            gt35[m] = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            dsteps[m] = step_view(prep["feats"], columns, lomo[m], prep["aligned"], prep["runs"], stride)
            dtouch[m] = [None] * len(dsteps[m])
            dfine[m] = fv(prep["feats"], lomo[m], prep["aligned"], dsteps[m])
            matches.append(m)
        print(f"[{time.time() - t0:5.0f}s] SkillCorner training games: {len(sc_names)} "
              f"({sum(len(dsteps[m]) for m in sc_names)} windows)", flush=True)

    fold_matches = list(ac.get("eval_matches") or [m for m in matches if not m.startswith("skillcorner_")])
    if len(set(fold_matches)) != len(fold_matches) or not set(fold_matches) <= set(matches):
        raise ValueError("eval_matches must be distinct matches present in the net's data")

    # Gün 7 (Ek Q.1): evaluation-only DFL corruption (dfl_referee_eval). Training is untouched; the
    # fold models below also predict the held-out DFL match corrupted this way.
    ecfg = dfl_referee.eval_cfg(cfg)
    lomo_e, preps_e, dsteps_e, ad_e, bnd_e, tokens_e = {}, {}, {}, {}, {}, {}
    if ecfg is not None:
        assert not use_fine, "dfl_referee_eval is not wired for fine branch"
        lomo_e = decoder_tune.eval_probs(cfg, folder, tuple(ac["extra_matches"]))
        for m in lomo_e:
            preps_e[m] = dfl_referee.prepare(m, ecfg, lengths)
            assert list(preps_e[m]["feats"].columns) == columns
            dsteps_e[m] = step_view(preps_e[m]["feats"], columns, lomo_e[m], None, preps_e[m]["runs"], stride)
            if use_tok:
                tokens_e[m] = tokens(preps_e[m]["corrupted"], preps_e[m]["runs"])
                dsteps_e[m] = with_tokens(dsteps_e[m], tokens_e[m])
        print(f"[{time.time() - t0:5.0f}s] evaluation corruption {cfg['dfl_referee_eval']} on {sorted(lomo_e)}", flush=True)
    # Gün 7 (Ek Q.4): dfl_mix_eval -- the net also trains on the evaluation-corrupted copy of every DFL
    # match (same targets), next to the training-corrupted one.
    dsteps_mix = {}
    if cfg.get("dfl_mix_eval"):
        assert ecfg is not None and not split, "dfl_mix_eval needs dfl_referee_eval and no split heads"
        dsteps_mix = {m: step_view(preps_e[m]["feats"], columns, lomo_e[m], targets[m], preps_e[m]["runs"], stride)
                      for m in lomo_e}
        if use_tok:
            dsteps_mix = {m: with_tokens(s, tokens_e[m]) for m, s in dsteps_mix.items()}
        # Kasım 2026 (external data only): Metrica training games also get a BAR-like corrupted copy
        # (evaluation corruption without ball masking), next to the raw one -- like the DFL mix.
        if ac.get("metrica_corrupt") and ac["metrica_games"]:
            from . import metrica_src
            assert not use_tok and not use_fine, "metrica_corrupt supports the dense net only"
            for g in ac["metrica_games"]:
                m = f"metrica_{g}"
                bm = float(ac.get("metrica_ball_mask") or 0.0)      # 11 Nov: BAR-like live ball masking
                pc = metrica_src.prepare_corrupted(g, cfg, dfl_referee.referee_cfg(ecfg), bm, lengths)
                assert list(pc["feats"].columns) == columns
                preps_e[m] = pc
                nm = m + ("_v2bf" if metrica_src.dead_fill(cfg) else "_v2bm" if bm > 0 else "_v2b")   # own booster-probability cache
                nm += f"d{metrica_src.dead_drift(cfg)}" if metrica_src.dead_drift(cfg) else ""
                lomo_e[m] = decoder_tune.unseen_probs(cfg, [nm], folder, preps=[pc])[nm]
                dsteps_mix[m] = step_view(pc["feats"], columns, lomo_e[m], targets[m], pc["runs"], stride)
            print(f"[{time.time() - t0:5.0f}s] Metrica corrupted training copies: {ac['metrica_games']}", flush=True)
        if ac["skillcorner_games"]:
            from . import skillcorner_src
            assert not use_tok and not use_fine, "SkillCorner copies support the dense net only"
            sc_c = {skillcorner_src.name(g, cfg): skillcorner_src.prepare_corrupted(g, cfg, dfl_referee.referee_cfg(ecfg))
                    for g in ac["skillcorner_games"]}
            sc_pe = decoder_tune.unseen_probs(cfg, [m + "_v2b" for m in sc_c], folder, preps=list(sc_c.values()))
            for m, pc in sc_c.items():
                assert list(pc["feats"].columns) == columns
                preps_e[m], lomo_e[m] = pc, sc_pe[m + "_v2b"]
                dsteps_mix[m] = step_view(pc["feats"], columns, lomo_e[m], targets[m], pc["runs"], stride)
            print(f"[{time.time() - t0:5.0f}s] SkillCorner corrupted training copies: {len(sc_c)}", flush=True)
        print(f"[{time.time() - t0:5.0f}s] mixed-corruption training copies: {sorted(dsteps_mix)}", flush=True)

    # 5 Nov: dfl_referee_train (v3_dr) -> DFL scoring copies never train; Metrica raw keeps training
    dr_mode = bool(cfg.get("dfl_referee_train"))
    train_ms = [m for m in matches if m.startswith(("metrica_", "skillcorner_"))] if dr_mode else matches
    if dr_mode and cfg.get("dfl_mix_eval"):
        raise ValueError("dfl_referee_train needs dfl_mix_eval off (the v2b copies are referees only)")
    copies = {}
    if team_copies:
        from ..augmentation import swapped_steps
        for m in train_ms:
            if team_copies == "repeat":
                copies[m] = list(dsteps[m]) + list(dsteps_mix.get(m, []))
            else:
                tk = {"token_params": p} if use_tok else {}
                copies[m] = swapped_steps(preps[m], cfg, columns, lomo[m], targets[m], stride, **tk)
                if m in dsteps_mix:
                    copies[m] += swapped_steps(preps_e[m], cfg, columns, lomo_e[m], targets[m], stride, **tk)
        print(f"  external team copies: {team_copies}; both corruptions; BAR anchors unchanged", flush=True)
    k = int(ac["skillcorner_stride"])
    if k > 1:
        # cap SkillCorner's weight in the mix: keep every k-th training window of each SkillCorner game
        # (raw, corrupted and team-exchanged views alike); DFL / Metrica untouched
        for m in [m for m in matches if m.startswith("skillcorner_")]:
            dsteps[m], dtouch[m] = dsteps[m][::k], dtouch[m][::k]
            if dfine[m] is not None:
                dfine[m] = dfine[m][::k]
            if m in dsteps_mix:
                dsteps_mix[m] = dsteps_mix[m][::k]
            if m in copies:
                copies[m] = copies[m][::k]
        print(f"  SkillCorner windows thinned: every {k}-th kept", flush=True)
    sc_lam = float(ac["skillcorner_team"])
    if not 0.0 <= sc_lam <= 1.0:
        raise ValueError("skillcorner_team must be in [0, 1]")
    sc_ids = set()
    if sc_lam < 1.0 and ac["skillcorner_games"]:
        # 5 Kasım: phase-of-play team labels are coarse (no short touches) -> SkillCorner windows keep
        # (1 - lambda) of a dead-vs-alive loss and lambda of the 3-class loss; with lambda < 1 their
        # boundary targets (which include coarse H/A edges) are off, and at lambda 0 the H<->A twins
        # carry nothing new, so they are dropped. DFL / Metrica windows are unchanged.
        assert external and not split and not use_fine, "skillcorner_team < 1 supports the external dense net only"
        for m in [m for m in matches if m.startswith("skillcorner_")]:
            if sc_lam == 0.0:
                copies.pop(m, None)
            sc_ids.update(id(s) for s in list(dsteps[m]) + list(dsteps_mix.get(m, [])) + list(copies.get(m, [])))
        print(f"  SkillCorner team weight {sc_lam:g} ({len(sc_ids)} sequences; boundary loss off)", flush=True)

    def net_items(steps):
        if not sc_ids:
            return [(x, y, k) for x, y, k, _ in steps]
        return [(x, y, k, 1.0, False, sc_lam) if id(s) in sc_ids else (x, y, k, 1.0, True, 1.0)
                for s in steps for x, y, k, _ in [s]]

    from ..augmentation import training_steps
    # 5 Kasım: more corruption draws of the external matches (a teammate's finding: three degradation
    # draws per match helped their nets). For each extra seed: the DFL training corruption, the DFL
    # evaluation (v2b) corruption and the Metrica v2b copy, re-drawn with that corruption_seed, each with
    # booster probabilities from a booster that never saw the match, plus H<->A twins. Training only:
    # the referees stay the seed-7 copies; a fold excludes every draw of its held-out match.
    draws = {}
    if ac["extra_draws"] or dr_mode:
        from . import metrica_src
        from ..augmentation import swapped_steps
        assert external and not split and not use_fine and not use_tok and not ac.get("short_team"),             "extra_draws / dfl_referee_train support the external dense net only"
        mix = bool(cfg.get("dfl_mix_eval"))
        if not dr_mode:
            assert ecfg is not None and mix, "extra_draws needs dfl_referee_eval + dfl_mix_eval"
        dfl_ms = [m for m in matches if not m.startswith(("metrica_", "skillcorner_"))]
        unseen = tuple(m for m in dfl_ms if m not in cfg["joint"]["matches"])
        seed0 = int(dfl_referee.referee_cfg(cfg)["corruption_seed"])

        def add(m, prep, bp):
            st = step_view(prep["feats"], columns, bp, targets[m], prep["runs"], stride)
            draws.setdefault(m, []).extend(st)
            if team_copies == "swap":
                draws[m] += swapped_steps(prep, cfg, columns, bp, targets[m], stride)
            elif team_copies == "repeat":
                draws[m] += list(st)
        # dr_mode (5 Nov, v3_dr): EVERY training copy is a domain-randomised draw (the primary seed too);
        # the seed-7 v1 / v2b copies only score. Otherwise extra seeds re-draw v1 and v2b.
        for sd in ([seed0] if dr_mode else []) + [int(x) for x in ac["extra_draws"]]:
            if not dr_mode and sd in (seed0, int(dfl_referee.referee_cfg(ecfg)["corruption_seed"])):
                raise ValueError(f"extra draw seed {sd} repeats an existing corruption seed")
            tcfg = dfl_referee.train_cfg(cfg, seed=sd)
            alts = [tcfg]
            ecfg_s = None
            if mix and ecfg is not None:
                ecfg_s = dfl_referee.train_cfg(ecfg, seed=sd) if not dr_mode else None
                alts += [ecfg_s] if ecfg_s is not None else []
            if ac.get("draw_overrides"):
                # 11 Nov: the extra DFL draws carry corruption overrides the scoring copies do not have, e.g.
                # {dead_drift: 1.5}: the visible dead ball wanders at the target's out-of-bounds ball speeds
                assert not dr_mode, "draw_overrides is for extra_draws"
                ov = dict(ac["draw_overrides"])
                only_eval = ov.pop("eval_copy_only", False)
                if ov.get("dead_drift"):
                    ov["dead_drift_speeds"] = [round(float(x), 3) for x in metrica_src.bar_dead_speeds(cfg)]
                for c in (tcfg, ecfg_s):
                    if c is not None:
                        c["dfl_referee"] = dict(c["dfl_referee"], **ov)
                if only_eval and ecfg_s is not None:                # one draw per seed: v2b + overrides
                    alts = [ecfg_s]
            for c in alts:
                bp = decoder_tune.draw_probs(cfg, folder, c, unseen)
                for m in dfl_ms:
                    pr = dfl_referee.prepare(m, c, lengths)
                    assert list(pr["feats"].columns) == columns
                    add(m, pr, bp[m])
            if ac.get("metrica_corrupt"):
                rc_m = dfl_referee.referee_cfg(tcfg if dr_mode else ecfg_s)
                rc_m = {k: v for k, v in rc_m.items() if not k.startswith("dead_drift")}   # Metrica drifts via metrica_dead_drift
                for g in ac["metrica_games"]:
                    m = f"metrica_{g}"
                    bm = float(ac.get("metrica_ball_mask") or 0.0)
                    nm = f"{m}_dr_s{sd}" if dr_mode else f"{m}_v2b{'f' if metrica_src.dead_fill(cfg) else 'm' if bm > 0 else ''}_s{sd}"
                    nm += f"d{metrica_src.dead_drift(cfg)}" if metrica_src.dead_drift(cfg) else ""
                    pc = metrica_src.prepare_corrupted(g, cfg, rc_m, bm, lengths)
                    assert list(pc["feats"].columns) == columns
                    add(m, pc, decoder_tune.unseen_probs(cfg, [nm], folder, preps=[pc])[nm])
            print(f"[{time.time() - t0:5.0f}s] {'v3_dr' if dr_mode else 'extra'} corruption draw seed {sd}: "
                  f"{sum(len(v) for v in draws.values())} draw sequences so far", flush=True)

    if not ac.get("metrica_raw_train", True):
        # 11 Nov: the RAW Metrica track never shows the dead ball and always shows the live one, so "visible =
        # live" holds without exception there. Train on the BAR-like copies only; the raw track still scores.
        assert ac.get("metrica_corrupt") and not dr_mode, "metrica_raw_train: false needs metrica_corrupt"
        dsteps_train = {m: ([] if m.startswith("metrica_") else v) for m, v in dsteps.items()}
        for m in [m for m in copies if m.startswith("metrica_")]:
            copies[m] = copies[m][len(copies[m]) - len(dsteps_mix[m]):] if team_copies != "repeat" else list(dsteps_mix[m])
        print("  raw Metrica windows (and their team twins) excluded from training", flush=True)
    else:
        dsteps_train = dsteps

    mrep = int(ac.get("metrica_repeat", 1))
    if mrep > 1:
        # 12 Nov: Metrica is the only source with the target's possession rule (same provider); its BAR-like
        # training windows (corrupted copies, twins, extra draws) are repeated to weigh more in the mix
        for m in [m for m in matches if m.startswith("metrica_")]:
            for d in (dsteps_mix, copies, draws):
                if m in d:
                    d[m] = list(d[m]) * mrep
        print(f"  Metrica training windows repeated x{mrep}", flush=True)

    def with_draws(steps, exclude=None):
        return steps + [s for o in draws if o != exclude for s in draws[o]]
    team_w = None
    if ac.get("short_team"):
        from ..labels.short_team import step_weights
        team_w = {m: step_weights(dsteps[m], targets[m], preps[m]["runs"], preps[m]["data"].fps,
                                 stride, **ac["short_team"]) for m in matches}
        print(f"  short-team CE weighting: {ac['short_team']} (external dense labels only)", flush=True)

    def training_team_weights(exclude=None):
        if team_w is None:
            return None
        return ([w for m in matches if m != exclude for w in team_w[m]]
                + [w for m in dsteps_mix if m != exclude for w in team_w[m]])

    def score_dfl_eval(w, probs_e=None, bnds_e=None):
        probs_e = ad_e if probs_e is None else probs_e
        bnds_e = bnd_e if bnds_e is None else bnds_e
        sp, core, mac = [], [], []
        diag = {}
        for m in probs_e:
            pe = preps_e[m]
            pred = engine.decode_runs(log_pool(lomo_e[m], probs_e[m], w), pe["runs"], dcfg, boundary=bnds_e[m],
                                      regime=contact.regime_for(pe["corrupted"], dcfg))
            r = pm.evaluate(gt35[m], pred)
            if ac["segment_diagnostics"]:
                from ..validation.possession import diagnostics
                diag[m] = diagnostics(gt35[m], pred, pe["runs"], pe["data"].fps)
            sp.append(r["spell_f1_mean"])
            mac.append(r["macro_f1"])
            if m in cfg["joint"]["matches"]:
                core.append(r["spell_f1_mean"])
        return {"w_boost": w, "spell_gt35": float(np.mean(sp)), **({"segments": diag} if diag else {}),
                "spell_gt35_core": float(np.mean(core)) if core else float("nan"),
                "macro": float(np.mean(mac)), "per_match_gt35": dict(zip(probs_e, [round(x, 4) for x in sp]))}

    def score_dfl(probs_by_match, w, bnd_by_match=None, c=None):
        sp11, sp35, mac, spc = [], [], [], []
        diag = {}
        c = dcfg if c is None else c
        for m in fold_matches:
            pred = engine.decode_runs(log_pool(lomo[m], probs_by_match[m], w), preps[m]["runs"], c,
                                      boundary=None if bnd_by_match is None else bnd_by_match[m],
                                      regime=contact.regime_for(preps[m].get("corrupted", preps[m]["data"]), c))
            s = dfl_referee.score(pred, preps[m], None)
            if ac["segment_diagnostics"]:
                from ..validation.possession import diagnostics
                diag[m] = diagnostics(gt35[m], pred, preps[m]["runs"], preps[m]["data"].fps)
            sp11.append(s["spell_f1"])
            mac.append(s["macro_f1"])
            sp35.append(pm.evaluate(gt35[m], pred)["spell_f1_mean"])
            spc.append(pm.evaluate(gtc[m], pred)["spell_f1_mean"] if m in gtc else float("nan"))
        core = [i for i, m in enumerate(fold_matches) if m in cfg["joint"]["matches"]]
        dfl = [i for i, m in enumerate(fold_matches) if not m.startswith("metrica_")]
        met = [i for i, m in enumerate(fold_matches) if m.startswith("metrica_")]
        mean = lambda xs, ii: float(np.mean([xs[i] for i in ii])) if ii else float("nan")  # noqa: E731
        return {"spell_gt11": mean(sp11, dfl), "spell_gt35": mean(sp35, dfl),
                **({"segments": diag} if diag else {}),
                "spell_gt35_core": mean(sp35, core), "spell_metrica": mean(sp35, met),
                "spell_ctrl": mean(spc, dfl), "spell_ctrl_core": mean(spc, core),
                "macro_metrica": mean(mac, met),
                "macro": mean(mac, dfl), "per_match_gt35": dict(zip(fold_matches, [round(x, 4) for x in sp35]))}

    seeds = network_seeds(cfg, ac, p)
    base_lomo, ad_lomo, ad_bnd, base_bnd = {}, {}, {}, {}
    t_lomo, t_bnd, t_e, t_bnd_e = {}, {}, {}, {}
    if tta:
        sw = {m: swapped_steps(preps[m], cfg, columns, lomo[m], None, stride) for m in fold_matches}
        sw_e = {m: swapped_steps(preps_e[m], cfg, columns, lomo_e[m], None, stride) for m in fold_matches if m in dsteps_e}
        print(f"[{time.time() - t0:5.0f}s] test-time team exchange ({tta}) on {len(sw)} + {len(sw_e)} held-out views", flush=True)
    for m in fold_matches:
        dfl_train = with_draws(training_steps(train_ms, dsteps_train, dsteps_mix, copies, exclude=m), m)
        t_dfl = [t for o in matches if o != m for t in dtouch[o]] if split else None
        f_dfl = [f for o in matches if o != m for f in dfine[o]] if use_fine else None
        tw_dfl = training_team_weights(m)
        base = [seqnet.train(net_items(dfl_train), p, sd, touch=t_dfl, fine=f_dfl,
                             team_weights=tw_dfl) for sd in seeds]
        if external:
            adapted, w_bar = base, 0.0
        else:
            train_mix, w_bar = mixed(dfl_train, bar_steps, ac["bar_share"], ac["d_share"], ac["bar_team"])
            t_mix = t_dfl + [None] * len(bar_steps) if split else None
            f_mix = f_dfl + bar_fine if use_fine else None
            tw_mix = None if tw_dfl is None else tw_dfl + [np.ones(len(s[0]), np.float32) for s in bar_steps]
            adapted = [seqnet.train(train_mix, p, sd, init=b, epochs=ac["ft_epochs"], lr=ac["ft_lr"], touch=t_mix,
                                    fine=f_mix, team_weights=tw_mix)
                       for sd, b in zip(seeds, base)]
        base_lomo[m] = net_probs(base, dsteps[m], len(lomo[m]), lomo[m], stride, dfine[m])
        base_bnd[m] = net_boundary(base, dsteps[m], len(lomo[m]), stride)
        ad_lomo[m] = net_probs(adapted, dsteps[m], len(lomo[m]), lomo[m], stride, dfine[m])
        ad_bnd[m] = net_boundary(adapted, dsteps[m], len(lomo[m]), stride)
        if m in dsteps_e:
            ad_e[m] = net_probs(adapted, dsteps_e[m], len(lomo_e[m]), lomo_e[m], stride, None)
            bnd_e[m] = net_boundary(adapted, dsteps_e[m], len(lomo_e[m]), stride)
        if tta:
            t_lomo[m] = tta_probs(ad_lomo[m], net_probs(adapted, sw[m], len(lomo[m]), lomo[m][:, SWAP], stride, None))
            t_bnd[m] = tta_boundary(ad_bnd[m], net_boundary(adapted, sw[m], len(lomo[m]), stride))
            if m in sw_e:
                t_e[m] = tta_probs(ad_e[m], net_probs(adapted, sw_e[m], len(lomo_e[m]), lomo_e[m][:, SWAP], stride, None))
                t_bnd_e[m] = tta_boundary(bnd_e[m], net_boundary(adapted, sw_e[m], len(lomo_e[m]), stride))
        # 5 Nov: the fold nets are kept so a held-out match can be re-scored later under any view (e.g. the clean
        # DFL track, tools/eval_clean.py) without retraining
        save_nets(out_dir / "fold_nets" / m, adapted, seeds, p, dfl_train[0][0].shape[-1], columns)
        print(f"[{time.time() - t0:5.0f}s] fold {m}: base + adapted (BAR window weight {w_bar:.2f})", flush=True)
    dfl_rows = []
    for label, probs, bnds in (("base", base_lomo, base_bnd), ("adapted", ad_lomo, ad_bnd)):
        for w in (0.0, ac["w_boost"]):
            r = {"net": label, "w_boost": w, **score_dfl(probs, w, bnds)}
            dfl_rows.append(r)
            print(f"  DFL LOMO {label:<8} w_boost {w:.2f}: spell {r['spell_gt11']:.4f} "
                  f"(gt35 {r['spell_gt35']:.4f}; core {r['spell_gt35_core']:.4f}, n={len(fold_matches)}) "
                  f"macro {r['macro']:.4f} ctrl-GT {r['spell_ctrl']:.4f} (core {r['spell_ctrl_core']:.4f})"
                  + (f" | Metrica LOMO spell {r['spell_metrica']:.4f} macro {r['macro_metrica']:.4f}"
                     if ac["metrica_games"] else ""), flush=True)
    r = {"net": "none (booster)", "w_boost": 1.0, **score_dfl(lomo, 1.0, None, cfg)}
    dfl_rows.append(r)
    eval_rows = []
    if ad_e:
        for w in (0.0, ac["w_boost"]):
            r = score_dfl_eval(w)
            eval_rows.append(r)
            print(f"  DFL LOMO EVAL-CORRUPTION adapted w_boost {w:.2f}: gt35 {r['spell_gt35']:.4f} "
                  f"(core {r['spell_gt35_core']:.4f}, n={len(ad_e)}) macro {r['macro']:.4f}", flush=True)
    alt = {}                                  # the other view of tta_swap ("tta" or "no_tta")
    if tta:
        t_rows = [{"net": "adapted", "w_boost": w, **score_dfl(t_lomo, w, t_bnd)} for w in (0.0, ac["w_boost"])]
        t_eval = [score_dfl_eval(w, t_e, t_bnd_e) for w in (0.0, ac["w_boost"])] if t_e else []
        for r in t_rows:
            print(f"  TTA DFL LOMO w_boost {r['w_boost']:.2f}: gt35 {r['spell_gt35']:.4f} macro {r['macro']:.4f}", flush=True)
        for r in t_eval:
            print(f"  TTA EVAL-CORRUPTION w_boost {r['w_boost']:.2f}: gt35 {r['spell_gt35']:.4f} macro {r['macro']:.4f}",
                  flush=True)
        if tta == "apply":                    # outputs carry the averaged view; keep the plain one aside
            alt = {"dfl_lomo": [r for r in dfl_rows if r["net"] == "adapted"], "dfl_lomo_eval": eval_rows}
            dfl_rows = [r for r in dfl_rows if r["net"] != "adapted"] + t_rows
            eval_rows = t_eval
            ad_lomo, ad_bnd, ad_e, bnd_e = t_lomo, t_bnd, t_e, t_bnd_e
        else:
            alt = {"dfl_lomo": t_rows, "dfl_lomo_eval": t_eval}
    if ad_e:
        np.savez_compressed(out_dir / "dfl_lomo_eval_blend.npz",
                            **{m: log_pool(lomo_e[m], ad_e[m], ac["w_boost"]) for m in ad_e})
        np.savez_compressed(out_dir / "dfl_lomo_eval_bnd.npz", **bnd_e)
    # blended LOMO probabilities of the adapted net (for decoder tuning on fixed probabilities)
    np.savez_compressed(out_dir / "dfl_lomo_blend.npz",
                        **{m: log_pool(lomo[m], ad_lomo[m], ac["w_boost"]) for m in ad_lomo})
    # raw (unblended) net probabilities for calibration / stacking (pipeline/stack.py)
    np.savez_compressed(out_dir / "dfl_lomo_net_adapted.npz", **ad_lomo)
    np.savez_compressed(out_dir / "dfl_lomo_net_base.npz", **base_lomo)
    # boundary-head probabilities of the adapted net (decoder boundary bonus, Gün 3)
    np.savez_compressed(out_dir / "dfl_lomo_bnd.npz", **ad_bnd)
    if tta == "report":                       # averaged views for fixed-probability decoder checks
        np.savez_compressed(out_dir / "dfl_lomo_blend_tta.npz",
                            **{m: log_pool(lomo[m], t_lomo[m], ac["w_boost"]) for m in t_lomo})
        np.savez_compressed(out_dir / "dfl_lomo_bnd_tta.npz", **t_bnd)
        if t_e:
            np.savez_compressed(out_dir / "dfl_lomo_eval_blend_tta.npz",
                                **{m: log_pool(lomo_e[m], t_e[m], ac["w_boost"]) for m in t_e})
            np.savez_compressed(out_dir / "dfl_lomo_eval_bnd_tta.npz", **t_bnd_e)

    # ---- BAR: all DFL matches + BAR anchors ---------------------------------------------------
    dfl_all = with_draws(training_steps(train_ms, dsteps_train, dsteps_mix, copies))
    t_all = [t for o in matches for t in dtouch[o]] if split else None
    f_all = [f for o in matches for f in dfine[o]] if use_fine else None
    tw_all = training_team_weights()
    base = [seqnet.train(net_items(dfl_all), p, sd, touch=t_all, fine=f_all,
                         team_weights=tw_all) for sd in seeds]
    if external:
        adapted, w_bar = base, 0.0
    else:
        train_mix, w_bar = mixed(dfl_all, bar_steps, ac["bar_share"], ac["d_share"], ac["bar_team"])
        t_mix = t_all + [None] * len(bar_steps) if split else None
        f_mix = f_all + bar_fine if use_fine else None
        tw_mix = None if tw_all is None else tw_all + [np.ones(len(s[0]), np.float32) for s in bar_steps]
        adapted = [seqnet.train(train_mix, p, sd, init=b, epochs=ac["ft_epochs"], lr=ac["ft_lr"], touch=t_mix,
                                fine=f_mix, team_weights=tw_mix)
                   for sd, b in zip(seeds, base)]
    save_nets(out_dir / "nets", adapted, seeds, p, dfl_all[0][0].shape[-1], columns)
    bar_rows, alt_bar = [], []
    if not cloud_mode:
        inf_steps = step_view(ctx["feats"], columns, p_boost, None, list(C.HALVES.values()), stride)
        inf_fine = fv(ctx["feats"], p_boost, None, inf_steps)
        if use_tok:
            inf_steps = with_tokens(inf_steps, bar_tokens)
        blocks = public_blocks()               # 'public' score = organisers' window only
        gt = merged_gt(blocks)
        ref_labels, ref_stats, ref_path = official_reference(cfg, exclude=name)
        bands = reference_bands(cfg["acceptance"], None, ref_stats)
        print(f"  reference for P(better) and bands: {ref_path}", flush=True)
        if tta:
            bar_prep = dict(corrupted=ctx["data"], key=ctx["data"].name, feats=ctx["feats"], runs=list(C.HALVES.values()))
            sw_bar = swapped_steps(bar_prep, cfg, columns, p_boost, None, stride)
        views = [("base", base, False), ("adapted", adapted, tta == "apply")]
        if tta:
            views.append(("adapted", adapted, tta == "report"))        # the secondary view, scored only
        for i, (label, models, averaged) in enumerate(views):
            primary = i < 2
            p_net = net_probs(models, inf_steps, len(p_boost), p_boost, stride, inf_fine)
            bnd = net_boundary(models, inf_steps, len(p_boost), stride)
            if averaged:
                p_net = tta_probs(p_net, net_probs(models, sw_bar, len(p_boost), p_boost[:, SWAP], stride, None))
                bnd = tta_boundary(bnd, net_boundary(models, sw_bar, len(p_boost), stride))
            if primary:
                np.save(out_dir / f"bar_net_{label}.npy", p_net)
                if label == "adapted":
                    np.save(out_dir / "bar_bnd_adapted.npy", bnd)
            elif tta == "report":
                np.save(out_dir / "bar_net_adapted_tta.npy", p_net)
                np.save(out_dir / "bar_bnd_adapted_tta.npy", bnd)
            for w in (0.0, ac["w_boost"]):
                probs = log_pool(p_boost, p_net, w)
                lab = m0.decode_match(probs, dcfg, bnd, regime=contact.regime_for(ctx["data"], dcfg))
                res = score_blocks(lab, blocks, bootstrap=1000)["ALL"]
                st = sanity.stats(lab)
                ok, _ = _in_bands(st, bands)
                pb = paired_spell_bootstrap(gt, ref_labels, lab)["p_b_better"]
                row = {"net": label, "w_boost": w, "spell_f1": res["spell_f1_mean"], "macro_f1": res["macro_f1"],
                       "boot": [res.get("spell_bootstrap_low"), res.get("spell_bootstrap_high")],
                       "dead_share": st["dead_share"], "n_possessions": st["n_possessions"],
                       "home_share": st["home_share"], "in_ref_bands": ok, "p_better_vs_main": pb}
                (bar_rows if primary else alt_bar).append(row)
                print(f"  BAR {label + ('' if primary else ' (alt)'):<8} w_boost {w:.2f}: public spell {row['spell_f1']:.4f} macro {row['macro_f1']:.4f} "
                      f"D {st['dead_share']:.3f} poss {st['n_possessions']} home {st['home_share']:.3f} "
                      f"bands {ok} P(better) {pb:.2f}", flush=True)
                if primary and label == "adapted" and w == ac["w_boost"]:
                    np.save(out_dir / "probs.npy", probs)
                    sub = submission.write(out_dir / "submission.csv", lab)
                    reason = submission.validate(sub)
                    print(f"  -> {sub} official format check: {'OK' if reason is None else reason}", flush=True)
                    seed_rows = []
                    for sd, mdl in zip(seeds, models):
                        lab_i = m0.decode_match(log_pool(p_boost, net_probs([mdl], inf_steps, len(p_boost), p_boost, stride, inf_fine), w),
                                                dcfg, net_boundary([mdl], inf_steps, len(p_boost), stride),
                                                regime=contact.regime_for(ctx["data"], dcfg))
                        r_i = score_blocks(lab_i, blocks)["ALL"]
                        seed_rows.append({"seed": sd, "spell_f1": r_i["spell_f1_mean"], "macro_f1": r_i["macro_f1"],
                                          **sanity.stats(lab_i)})
                    (out_dir / "seeds.json").write_text(json.dumps(seed_rows, indent=1, default=float), encoding="utf-8")
    met_rows = []
    if ac["metrica_eval"]:
        from . import metrica_src
        assert not set(ac["metrica_eval"]) & set(ac["metrica_games"]), "metrica_eval games must not be trained on"
        ev_blend, ev_bnd, alt_met = {}, {}, []
        for g in ac["metrica_eval"]:
            m = f"metrica_{g}"
            prep = metrica_src.prepare(g, cfg)
            assert list(prep["feats"].columns) == columns
            pb = decoder_tune.unseen_probs(cfg, [m], folder, preps=[prep])[m]
            st = step_view(prep["feats"], columns, pb, None, prep["runs"], stride)
            if use_tok:
                st = with_tokens(st, tokens(prep["data"], prep["runs"]))
            p_net = net_probs(adapted, st, len(pb), pb, stride, fv(prep["feats"], pb, None, st))
            b_net = net_boundary(adapted, st, len(pb), stride)
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            if tta:
                st_sw = swapped_steps(prep, cfg, columns, pb, None, stride)
                p_t = tta_probs(p_net, net_probs(adapted, st_sw, len(pb), pb[:, SWAP], stride, None))
                b_t = tta_boundary(b_net, net_boundary(adapted, st_sw, len(pb), stride))
                if tta == "apply":
                    p_net, b_net, p_t, b_t = p_t, b_t, p_net, b_net
                pred_t = engine.decode_runs(log_pool(pb, p_t, ac["w_boost"]), prep["runs"], dcfg, boundary=b_t,
                                            regime=contact.regime_for(prep.get("corrupted", prep["data"]), dcfg))
                r_t = pm.evaluate(gt, pred_t)
                alt_met.append({"match": m, "spell": r_t["spell_f1_mean"], "macro": r_t["macro_f1"]})
            ev_blend[m] = log_pool(pb, p_net, ac["w_boost"])
            ev_bnd[m] = b_net
            pred = engine.decode_runs(ev_blend[m], prep["runs"], dcfg, boundary=ev_bnd[m],
                                      regime=contact.regime_for(prep.get("corrupted", prep["data"]), dcfg))
            r = pm.evaluate(gt, pred)
            met_rows.append({"match": m, "spell": r["spell_f1_mean"], "macro": r["macro_f1"]})
            if ac["segment_diagnostics"]:
                from ..validation.possession import diagnostics
                met_rows[-1]["segments"] = diagnostics(gt, pred, prep["runs"], prep["data"].fps)
            print(f"  Metrica eval {m} (never trained on): spell {r['spell_f1_mean']:.4f} macro {r['macro_f1']:.4f}", flush=True)
        np.savez_compressed(out_dir / "metrica_eval_blend.npz", **ev_blend)
        np.savez_compressed(out_dir / "metrica_eval_bnd.npz", **ev_bnd)
        print(f"  Metrica eval mean: spell {np.mean([r['spell'] for r in met_rows]):.4f} "
              f"macro {np.mean([r['macro'] for r in met_rows]):.4f}", flush=True)
        if alt_met:
            alt["metrica_eval"] = alt_met
            print(f"  Metrica eval mean (alt view): spell {np.mean([r['spell'] for r in alt_met]):.4f} "
                  f"macro {np.mean([r['macro'] for r in alt_met]):.4f}", flush=True)
    res = {"config": config, "trains_on_public_frames": trains_on_blocks(cfg), "adapt": ac,
           "eval_matches": fold_matches, "training_matches": matches, "net_seeds": seeds,
           "core_lomo_note": "core folds are not nested across booster inputs; use booster-unseen folds for selection",
           "bar_window_weight": w_bar, "bar_anchor_steps": n_known,
           "dfl_lomo": dfl_rows, "dfl_lomo_eval": eval_rows, "dfl_referee_eval": cfg.get("dfl_referee_eval"),
           "bar": bar_rows, "metrica_eval": met_rows, "cloud": cloud_mode,
           **({("no_tta" if tta == "apply" else "tta"): {**alt, "bar": alt_bar}} if tta else {}),
           "time": dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return res
