"""Second referee: score a configuration on an IDSSE/DFL match with REAL labels.

Protocol (per match): the clean DFL track is first corrupted to look like our ATD, then the
pipeline is trained on the first half and scored on the second half with the official metric.
  corruption "state" (default, Adım 3.2): the ball is hidden in blocks drawn from BAR-ATH's own
             gap-length distribution, ~58% of visible dead frames and ~32% of visible live frames
             (the rates seen on BAR-ATH, labels/masking.py). "uniform": 41% everywhere (Adım 2).
  weak : the BAR-ATH pipeline as-is (weak-label anchors from cfg.weak_labels + uniformly masked
         copies; no true label is used for training)
  sup  : the same model trained on the first half's true (aligned) labels, state-aware copies --
         the information content of the features without weak-label noise
Primary GT = DFL labels with sub-12-frame return spells suppressed (labels/align.py); the raw
labels are scored too. Only source-common groups are used (no Smart Tagging on DFL).
D recall is also split by whether the ball is visible in the corrupted test track: a model that
only finds dead balls it can SEE is relying on DFL's frozen dead ball, which ATD does not have.
"""

import numpy as np

from .. import config as config_mod
from .. import constants as C
from ..data import TrackingData
from ..features import build_features, common_groups
from ..labels import weak
from ..labels.align import suppress_short_returns
from ..labels.masking import mask_ball, mask_ball_by_state
from ..validation.official import pm
from . import engine

DEFAULTS = dict(corruption="state", corruption_fraction=0.41, dead_fraction=0.58, live_fraction=0.32,
                corruption_seed=7, align_max_frames=11, train_period=1, test_period=2,
                dead_jitter_m=0.0, ball_smooth_frames=0, ball_offset_m=0.0, ball_offset_frames=25,
                dead_carry_prob=0.0, dead_carry_offset_m=0.3, player_smooth_frames=0,
                player_gap_frac=0.0, player_gap_mean_s=3.8,
                player_gap_live=0.0, player_gap_live_s=1.0, player_gap_dead=0.0, player_gap_dead_s=7.0,
                ghost_frac=0.0, ghost_mean_s=3.0, ghost_offset_m=2.0)
# Corruption v2b (Gün 7, Ek Q.4): BAR tracks ~10.8 players per team in live play and ~9.6 in dead
# play, and shows an extra (ghost) track on ~4-9% of frames per team. player_gap_live / _dead: per-slot
# missing share in live / dead frames (independent renewal processes, mean gap _s seconds); ghost_frac:
# share of frames per team with one extra track copying a team-mate at a slowly varying offset.
# Corruption v2 (Gün 6, Ek P.7; a teammate's ATD-like protocol, re-calibrated on label-free BAR
# statistics; all 0 = off, cache keys unchanged):
#  dead_carry_prob: per dead segment, with this probability the visible dead ball follows the player
#    nearest to it at the segment's first visible frame (+ dead_carry_offset_m): ATD sees the ball being
#    carried to a restart (BAR predicted-dead visible ball median 2.2 m/s; DFL freezes it at 0).
#  player_smooth_frames: centred moving average of every player track (BAR player frame-to-frame
#    |dv| median 0.03 vs DFL 0.25: the ATD player tracks are smooth, not jittery).
#  player_gap_frac / player_gap_mean_s: per-slot dropouts as an alternating renewal process (BAR 20.5
#    of 22 players tracked, tracklet median ~35 s; DFL 22, whole match).
# ball_smooth_frames / ball_offset_m (Gün 4, Ek M.7): the BAR ATD ball is smoother than DFL's
# (9-frame residual p90 9.8 vs 13-26 cm, frame-to-frame |dv| 0.20 vs 0.34-0.48 m/s) and sits farther
# from the controlling player (slow-ball nearest distance median 0.90 vs 0.61-0.66 m), with no lag.
# A centred moving average over each visible run (13 frames) plus a slowly varying offset (Gaussian-
# filtered noise, 25 frames, 0.5 m per axis) matches all three. 0 = off (main unchanged).
# dead_jitter_m (Gün 2, Adım 4d): DFL freezes the dead ball (speed exactly 0), ATD's visible dead
# ball moves (median 2.05 m/s on BAR-ATH predicted-dead frames). i.i.d. Gaussian noise of 0.2 m on
# the visible dead-ball frames gives a median of 2.08 m/s and removes the "static ball = dead" cue.

_PREPARED = {}

# v3_dr (5 Nov 2026, organisers' November rules): TARGET-FREE domain randomisation of the training
# corruption. Every (corruption_seed, match) draws its own parameters from wide a-priori ranges that
# span clean optical tracking to heavily degraded automatic tracking; nothing is set from statistics of
# the provided game (unlike v1 / v2b, whose constants came from its label-free tracking statistics).
# Ball-gap lengths come from a fixed bimodal prior (short occlusions + long tracker losses) instead of the
# provided game's gap list. Ranges are chosen once, wide, and not tuned on anything.
DR_V3 = dict(live_fraction=(0.10, 0.50), dead_fraction=(0.20, 0.70), dead_jitter_m=(0.0, 1.0),
             dead_carry_prob=(0.0, 0.5), ball_smooth_frames=(1, 5, 9, 13, 17, 25), ball_offset_m=(0.05, 0.55),
             player_smooth_frames=(1, 5, 9, 15, 21), player_gap_live=(0.0, 0.08), player_gap_live_s=(0.5, 5.0),
             player_gap_dead=(0.0, 0.25), player_gap_dead_s=(2.0, 15.0), ghost_frac=(0.0, 0.10),
             ghost_mean_s=(1.0, 8.0), ghost_offset_m=(0.5, 3.0),
             gap_short_share=0.15, gap_short_frames=(5, 50), gap_long_frames=(125, 4500))
_LOG = ("player_gap_live_s", "player_gap_dead_s", "ghost_mean_s")
_CHOICE = ("ball_smooth_frames", "player_smooth_frames")


def dr_lengths(rng, n=4000, r=DR_V3):
    """Ball-gap lengths (frames) from the bimodal prior: short occlusions and long tracker losses."""
    short = rng.random(n) < r["gap_short_share"]
    lo = np.where(short, r["gap_short_frames"][0], r["gap_long_frames"][0])
    hi = np.where(short, r["gap_short_frames"][1], r["gap_long_frames"][1])
    return np.exp(rng.uniform(np.log(lo), np.log(hi))).astype(np.int64)


def sample_dr(rc, name):
    """(corruption config with drawn parameters, ball-gap lengths) for one (corruption_seed, match)."""
    import zlib
    if rc["dr"] != "v3":
        raise ValueError(f"unknown domain randomisation {rc['dr']}")
    rng = np.random.default_rng([int(rc["corruption_seed"]), zlib.crc32(name.encode())])
    out = dict(rc)
    for k, v in DR_V3.items():
        if k.startswith("gap_"):
            continue
        if k in _CHOICE:
            out[k] = int(rng.choice(v))
        elif k in _LOG:
            out[k] = round(float(np.exp(rng.uniform(np.log(v[0]), np.log(v[1])))), 3)
        else:
            out[k] = round(float(rng.uniform(*v)), 3)
    out["player_gap_frac"] = 0.0
    return out, dr_lengths(rng)


def train_cfg(cfg, seed=None):
    """Config of the TRAINING copies: dfl_referee + dfl_referee_train (e.g. {dr: v3}), optionally with
    another corruption_seed. Scoring copies keep dfl_referee / dfl_referee_eval. Same object back when
    there is nothing to change (existing configs and caches are untouched)."""
    tr = cfg.get("dfl_referee_train")
    if not tr and seed is None:
        return cfg
    import copy
    out = copy.deepcopy(cfg)
    out["dfl_referee"] = dict(cfg.get("dfl_referee") or {}, **(tr or {}))
    if seed is not None:
        out["dfl_referee"]["corruption_seed"] = int(seed)
    out.pop("dfl_referee_train", None)
    return out


def referee_cfg(cfg):
    return dict(DEFAULTS, **cfg.get("dfl_referee", {}))


def eval_cfg(cfg):
    """Gün 7 (Ek Q.1): a second, EVALUATION-ONLY DFL corruption. `dfl_referee_eval` (dict of
    dfl_referee overrides) leaves training untouched: the LOMO models trained under `dfl_referee`
    additionally predict the held-out match corrupted this way. None when the key is absent."""
    ev = cfg.get("dfl_referee_eval")
    if not ev:
        return None
    import copy
    out = copy.deepcopy(cfg)
    out["dfl_referee"] = dict(cfg.get("dfl_referee") or {}, **ev)
    return out


def eval_tag(cfg):
    """File-name tag of evaluation-corruption caches ('' without dfl_referee_eval)."""
    ev = cfg.get("dfl_referee_eval")
    return f"_ev{config_mod.digest(ev)[:8]}" if ev else ""


def corrupt(data, rc, bar_lengths):
    """(corrupted copy, cache key) -- the ATD-like version of a clean DFL track."""
    dr = rc.get("dr")
    if dr:
        rc, bar_lengths = sample_dr(rc, data.name)
    rng = np.random.default_rng(int(rc["corruption_seed"]))
    tag = engine.lengths_tag(bar_lengths)
    if rc["corruption"] == "state":
        out, _ = mask_ball_by_state(data, rng, data.labels, rc["dead_fraction"], rc["live_fraction"],
                                    bar_lengths)
        key = f"{data.name}__scorr_s{rc['corruption_seed']}_d{rc['dead_fraction']}_l{rc['live_fraction']}_{tag}"
    elif rc["corruption"] == "uniform":
        out, _ = mask_ball(data, rng, rc["corruption_fraction"], bar_lengths)
        key = f"{data.name}__corr_s{rc['corruption_seed']}_f{rc['corruption_fraction']}_{tag}"
    elif rc["corruption"] == "none":            # source already has ATD-like ball gaps (Metrica open data)
        out = data.copy()
        key = f"{data.name}__nomask_s{rc['corruption_seed']}"
    else:
        raise ValueError(rc["corruption"])
    sig = float(rc.get("dead_jitter_m", 0.0))
    if sig > 0:
        jrng = np.random.default_rng(int(rc["corruption_seed"]) + 1)
        sel = (data.labels == C.D) & ~np.isnan(out.ball[:, 0])
        out.ball[sel] += jrng.normal(0.0, sig, (int(sel.sum()), 2)).astype(np.float32)
        key += f"_j{sig}"
    pc = float(rc.get("dead_carry_prob", 0.0))
    if pc > 0:
        out.ball = carry_dead_ball(out.ball, out.players, data.labels, data.period, pc,
                                   float(rc.get("dead_carry_offset_m", 0.3)),
                                   np.random.default_rng(int(rc["corruption_seed"]) + 3))
        key += f"_dc{pc}_{rc.get('dead_carry_offset_m', 0.3)}"
    dd = float(rc.get("dead_drift", 0.0))
    if dd > 0:      # 11 Nov: the visible dead ball wanders at the target's out-of-bounds ball speeds (metrica_src)
        from . import metrica_src
        tmp = out.copy()
        tmp.labels = data.labels
        out.ball = metrica_src.drift_dead_ball(tmp, np.asarray(rc["dead_drift_speeds"], float), dd,
                                               int(rc["corruption_seed"]) + 11).ball
        key += f"_dd{dd}"
    pw = int(rc.get("player_smooth_frames", 0))
    gf = float(rc.get("player_gap_frac", 0.0))
    if pw > 1 or gf > 0:
        out.players = atd_like_players(out.players, pw, gf, float(rc.get("player_gap_mean_s", 3.8)), data.fps,
                                       np.random.default_rng(int(rc["corruption_seed"]) + 4))
        key += f"_ps{pw}_pg{gf}_{rc.get('player_gap_mean_s', 3.8)}"
    gl, gd, gh = (float(rc.get(k, 0.0)) for k in ("player_gap_live", "player_gap_dead", "ghost_frac"))
    if gl > 0 or gd > 0 or gh > 0:
        out.players = state_gaps_and_ghosts(out.players, data.team, data.labels, data.fps, gl,
                                            float(rc.get("player_gap_live_s", 1.0)), gd,
                                            float(rc.get("player_gap_dead_s", 7.0)), gh,
                                            float(rc.get("ghost_mean_s", 3.0)), float(rc.get("ghost_offset_m", 2.0)),
                                            np.random.default_rng(int(rc["corruption_seed"]) + 5))
        key += f"_sg{gl}_{rc.get('player_gap_live_s', 1.0)}_{gd}_{rc.get('player_gap_dead_s', 7.0)}_gh{gh}"
    w, off = int(rc.get("ball_smooth_frames", 0)), float(rc.get("ball_offset_m", 0.0))
    if w > 1 or off > 0:
        out.ball = atd_like_ball(out.ball, w, off, int(rc.get("ball_offset_frames", 25)),
                                 np.random.default_rng(int(rc["corruption_seed"]) + 2))
        key += f"_bs{w}_bo{off}_{int(rc.get('ball_offset_frames', 25))}"
    if dr:
        key += f"_dr{dr}"
    return out, key


def _segments(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def carry_dead_ball(ball, players, labels, period, prob, offset_m, rng):
    """Visible dead-ball frames of a randomly chosen share of dead segments follow the player nearest to
    the ball at the segment's first visible frame (a fixed offset of offset_m in a random direction).
    Frames where that player is untracked keep their ball position. NaN stays NaN."""
    b = np.asarray(ball, np.float32).copy()
    P = np.asarray(players, np.float32)
    dead = (np.asarray(labels) == C.D) & (np.asarray(period) > 0)
    for s, e in _segments(dead):
        if rng.random() >= prob:
            continue
        vis = np.flatnonzero(~np.isnan(b[s:e, 0])) + s
        if len(vis) == 0:
            continue
        d = np.linalg.norm(P[vis[0]] - b[vis[0]], axis=1)
        if np.all(np.isnan(d)):
            continue
        j = int(np.nanargmin(d))
        ang = rng.uniform(0, 2 * np.pi)
        follow = P[vis, j] + offset_m * np.array([np.cos(ang), np.sin(ang)], np.float32)
        ok = ~np.isnan(follow[:, 0])
        b[vis[ok]] = follow[ok]
    return b


def atd_like_players(players, smooth_frames, gap_frac, gap_mean_s, fps, rng):
    """Player tracks degraded like ATD: per-slot dropouts (alternating exponential visible / missing
    spells, missing share gap_frac, mean gap gap_mean_s) and a centred moving average over every
    visible run. NaN stays NaN."""
    from scipy.ndimage import uniform_filter1d
    P = np.asarray(players, np.float64).copy()
    n = P.shape[0]
    if gap_frac > 0:
        g = gap_mean_s * fps
        v = g * (1 - gap_frac) / gap_frac
        for j in range(P.shape[1]):
            t = int(rng.exponential(v))
            while t < n:
                L = max(1, int(rng.exponential(g)))
                P[t:t + L, j] = np.nan
                t += L + max(1, int(rng.exponential(v)))
    if smooth_frames > 1:
        for j in range(P.shape[1]):
            vis = ~np.isnan(P[:, j, 0])
            for s, e in _segments(vis):
                if e - s >= smooth_frames:
                    P[s:e, j] = uniform_filter1d(P[s:e, j], smooth_frames, axis=0, mode="nearest")
    return P.astype(np.float32)


def _renewal_mask(n, frac, mean_frames, rng):
    """True where a slot is missing: alternating exponential visible / missing spells, missing share frac."""
    m = np.zeros(n, bool)
    if frac <= 0:
        return m
    v = mean_frames * (1 - frac) / frac
    t = int(rng.exponential(v))
    while t < n:
        L = max(1, int(rng.exponential(mean_frames)))
        m[t:t + L] = True
        t += L + max(1, int(rng.exponential(v)))
    return m


def state_gaps_and_ghosts(players, team, labels, fps, gap_live, gap_live_s, gap_dead, gap_dead_s,
                          ghost_frac, ghost_mean_s, ghost_offset_m, rng):
    """State-dependent per-slot dropouts (live / dead frames, true labels of the clean track) and
    ghost tracks (an extra track copying a random tracked team-mate at a slowly varying offset, placed in
    a slot of that team that is empty at that time). NaN stays NaN outside ghosts."""
    from scipy.ndimage import gaussian_filter1d
    P = np.asarray(players, np.float64).copy()
    n = P.shape[0]
    dead = np.asarray(labels) == C.D
    for j in range(P.shape[1]):
        if np.isnan(P[:, j, 0]).all():
            continue
        miss = (_renewal_mask(n, gap_live, gap_live_s * fps, rng) & ~dead) | \
               (_renewal_mask(n, gap_dead, gap_dead_s * fps, rng) & dead)
        P[miss, j] = np.nan
    if ghost_frac > 0:
        team = np.asarray(team)
        for t in np.unique(team):
            slots = np.flatnonzero(team == t)
            on = _renewal_mask(n, ghost_frac, ghost_mean_s * fps, rng)
            off = gaussian_filter1d(rng.normal(0, 1, (n, 2)), fps, axis=0)
            off = off / off.std(axis=0) * ghost_offset_m
            for s, e in _segments(on):
                vis = [j for j in slots if not np.isnan(P[s:e, j, 0]).any()]
                empty = [j for j in slots if np.isnan(P[s:e, j, 0]).all()]
                if not vis or not empty:
                    continue
                src, dst = vis[rng.integers(len(vis))], empty[rng.integers(len(empty))]
                P[s:e, dst] = P[s:e, src] + off[s:e]
    return P.astype(np.float32)


def atd_like_ball(ball, smooth_frames, offset_m, offset_frames, rng):
    """Ball track degraded like the BAR ATD ball: moving average over each visible run (runs shorter
    than the window are left as they are) plus a slowly varying 2-D offset. NaN stays NaN."""
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    b = np.asarray(ball, np.float64).copy()
    vis = ~np.isnan(b[:, 0])
    if smooth_frames > 1:
        edges = np.flatnonzero(np.diff(np.r_[0, vis.astype(np.int8), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= smooth_frames:
                b[s:e] = uniform_filter1d(b[s:e], smooth_frames, axis=0, mode="nearest")
    if offset_m > 0:
        z = gaussian_filter1d(rng.normal(0.0, 1.0, b.shape), offset_frames, axis=0)
        b[vis] += (z / z.std(axis=0) * offset_m)[vis]
    return b.astype(np.float32)


def prepare(match_id, cfg, bar_lengths):
    """Clean + corrupted track, features of the corrupted track, aligned labels (memoized)."""
    rc = referee_cfg(cfg)
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    memo_key = (match_id, config_mod.digest({"rc": rc, "f": fcfg["features"], "p": fcfg["preprocess"]}),
                engine.lengths_tag(bar_lengths))
    if memo_key not in _PREPARED:
        from ..io.idsse import cached_idsse
        data = cached_idsse(match_id)          # builds cache/idsse_<id> from the raw files if missing
        corrupted, key = corrupt(data, rc, bar_lengths)
        feats = build_features(corrupted, fcfg, None, cache_key=key)
        aligned = suppress_short_returns(data.labels, data.runs(), int(rc["align_max_frames"]), data,
                                         bool(rc.get("align_check_control", False)))
        lengths = sample_dr(rc, data.name)[1] if rc.get("dr") else bar_lengths
        _PREPARED[memo_key] = dict(data=data, corrupted=corrupted, feats=feats, key=key,
                                   aligned=aligned, runs=data.runs(), cfg=fcfg, lengths=lengths)
    return _PREPARED[memo_key]


def score(pred, prep, period):
    """Official metric on one period (or all played frames if period is None) + D detail."""
    data, aligned = prep["data"], prep["aligned"]
    sel = data.period > 0 if period is None else data.period == period
    gt = np.where(sel, aligned, C.X).astype(np.int8)
    gt_raw = np.where(sel, data.labels, C.X).astype(np.int8)
    r, r_raw = pm.evaluate(gt, pred), pm.evaluate(gt_raw, pred)
    vis = ~np.isnan(prep["corrupted"].ball[:, 0])
    dead = sel & (aligned == C.D)
    rec = lambda m: float(np.mean(pred[m] == C.D)) if m.any() else float("nan")  # noqa: E731
    return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "spell_f1_raw": r_raw["spell_f1_mean"], "macro_f1_raw": r_raw["macro_f1"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
            "d_recall": r["per_class"]["D"]["recall"], "d_precision": r["per_class"]["D"]["precision"],
            "d_recall_ball_visible": rec(dead & vis), "d_recall_ball_hidden": rec(dead & ~vis),
            "team_acc": r["team_acc_both_alive"]}


def evaluate(match_id, groups, cfg, bar_lengths, modes=("weak", "sup")):
    rc = referee_cfg(cfg)
    cfg = config_mod.with_groups(cfg, common_groups(groups))
    prep = prepare(match_id, cfg, bar_lengths)
    data, corrupted, feats = prep["data"], prep["corrupted"], prep["feats"]
    train_ok = data.period == rc["train_period"]
    test_runs = [(a, b) for a, b in prep["runs"] if data.period[a] == rc["test_period"]]
    stride = int(cfg["lgbm"]["row_stride"])

    out = {}
    for mode in modes:
        info = {}
        if mode == "weak":
            labels, team_ok = weak.anchors(feats, data.period, cfg, (), data=corrupted, info=info)
            state = None
            sel = train_ok & (labels == C.D)
            info["d_anchor_precision_h1"] = float(np.mean(data.labels[sel] == C.D)) if sel.any() else float("nan")
            info["d_anchor_cover_h1"] = float(np.mean(labels[train_ok & (data.labels == C.D)] == C.D))
        elif mode == "sup":
            labels, team_ok, state = prep["aligned"], np.ones(data.n_frames, bool), prep["aligned"]
        else:
            raise ValueError(mode)
        src = engine.TrainSource(f"dfl_{match_id}_{mode}", corrupted, feats, labels, team_ok, train_ok,
                                 prep["key"], bar_lengths, None, stride, state_labels=state,
                                 d_share=cfg["weak_labels"].get("d_weight_share") if mode == "weak" else None)
        (per_seed,), _, counts = engine.fit_predict_sources([src], [feats], cfg)
        pred = engine.decode_runs(engine.ensemble(per_seed), test_runs, cfg)
        out[mode] = {**score(pred, prep, rc["test_period"]), "train_rows": counts[src.name], **info}
    return out
