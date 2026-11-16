"""Metrica open games as a source for the temporal net (Gün 2 akşam).

prepare(game, cfg) returns the same dict as dfl_referee.prepare so adapt / evalkit code can treat
a Metrica game like a DFL match:
  data / corrupted  the RAW Metrica track (its ball gaps already look like our ATD: 59-68% of
                    played frames visible) -- no synthetic corruption
  feats             source-common feature groups (no Smart Tagging), cached under metrica_<g>
  aligned           event-derived labels (io/metrica_open.derive_labels; X = loose ball)
  runs, key, cfg    as for DFL
"""

from .. import config as config_mod
from ..features import build_features, common_groups
from ..io.metrica_open import ball_labels, cached_game as _cached_game

_MEMO = {}


def label_variant(cfg):
    """None (event-derived dead ball) or ('ball', min_dead): `metrica_labels: ball` (11 Nov)."""
    v = cfg.get("metrica_labels")
    if not v:
        return None
    if v != "ball":
        raise ValueError(f"unknown metrica_labels {v!r}")
    return ("ball", int(cfg.get("metrica_min_dead", 50)))


def cached_game(game, cfg=None):
    """The cached game; with a label variant a copy carrying that variant's labels (features unaffected)."""
    data = _cached_game(game)
    v = label_variant(cfg or {})
    if v is None:
        return data
    data = data.copy()
    data.labels = ball_labels(game, v[1]).astype(data.labels.dtype)
    return data


def fill_dead_ball(data, ramp=75, reach=250):
    """Copy of `data` whose ball is visible through every dead-ball gap (11 Nov, `metrica_dead_fill`): it
    moves in a straight line from its last position before the gap to the restart position (first position
    after the gap; each searched within `reach` frames) during the first `ramp` frames and rests there. Raw Metrica never shows the ball in dead
    play, the target's tracking shows it in ~42 % of the dead frames; the filled track is then masked like
    a DFL track (dead_fraction / live_fraction of the corruption), so neither "hidden" nor "visible" alone
    tells dead from live."""
    import numpy as np
    from .. import constants as C
    out = data.copy()
    b = out.ball
    src = data.ball
    vis = np.flatnonzero(~np.isnan(src[:, 0]))
    dead = (np.asarray(data.labels) == C.D) & (np.asarray(data.period) > 0)
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        before = vis[(vis < s) & (vis >= s - reach)]
        after = vis[(vis >= e) & (vis < e + reach)]
        if len(before) == 0 and len(after) == 0:
            continue
        p0 = src[before[-1]] if len(before) else src[after[0]]         # last seen position / restart position
        p1 = src[after[0]] if len(after) else p0
        m = min(e - s, int(ramp))
        t = (np.arange(1, m + 1) / m)[:, None]
        b[s:s + m] = p0 + t * (p1 - p0)
        b[s + m:e] = p1
    return out


def bar_dead_speeds(cfg):
    """Label-free: speed quantiles (5 %, 10 %, ... 95 %) of the target's VISIBLE ball while it is physically
    out of bounds (ball_oob_flag). No label of the provided game is read: out of bounds is geometry."""
    import numpy as np
    from .. import cloud
    if cloud.enabled():                          # cloud mode (Tiki-Data, 13 Nov): the 19 quantiles exported locally
        return np.load(cloud.lengths_path().parent / "bar_dead_speeds.npy")
    from . import adapt
    ctx = adapt.bar_context(cfg)
    f, per = ctx["feats"], ctx["data"].period > 0
    m = per & (f["ball_oob_flag"].to_numpy() > 0) & (f["ball_vis"].to_numpy() > 0)
    return np.nanpercentile(f["ball_speed"].to_numpy()[m], np.arange(5, 100, 5))


def drift_dead_ball(data, speeds, scale, seed, block=50):
    """Copy of `data` whose dead ball wanders (11 Nov, `metrica_dead_drift`): inside every dead segment the
    visible ball gets a smooth offset path -- a new velocity every `block` frames, speed drawn from `speeds`
    (bar_dead_speeds) times `scale`, random direction, linear between blocks; the path is detrended so the
    ball is back at its own position at both ends. The simulated dead ball (static + jitter, or carried) moves
    at a median of ~1.3 m/s, the target's out-of-bounds ball at ~2.7 m/s."""
    import numpy as np
    from .. import constants as C
    out = data.copy()
    rng = np.random.default_rng(seed)
    dead = (np.asarray(data.labels) == C.D) & (np.asarray(data.period) > 0)
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        n = e - s
        if n < 2 * block:
            continue
        k = n // block + 2
        ang = rng.uniform(0, 2 * np.pi, k)
        v = (rng.choice(speeds, k) * scale)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)   # m/s at the knots
        vt = np.stack([np.interp(np.arange(n), np.arange(k) * block, v[:, j]) for j in (0, 1)], 1)
        off = np.cumsum(vt, 0) / data.fps
        off -= np.linspace(0, 1, n)[:, None] * off[-1]
        out.ball[s:e] += off.astype(np.float32)
    return out


def dead_drift(cfg):
    """None or the speed scale of `metrica_dead_drift` (true = 1.0)."""
    v = cfg.get("metrica_dead_drift")
    return None if not v else (1.0 if v is True else float(v))


def dead_fill(cfg):
    return bool(cfg.get("metrica_dead_fill"))


def _with_labels(track, labels):
    if track.labels is None:
        return track
    track = track.copy()
    track.labels = labels.copy()
    return track


def prepare_corrupted(game, cfg, rc, ball_mask=0.0, lengths=()):
    """Metrica game degraded like the target's tracking with a DFL corruption config `rc`
    (dfl_referee.referee_cfg of the evaluation corruption) EXCEPT the ball masking: Metrica's own
    ball track already has ATD-like gaps. Same dict as prepare(); labels unchanged.
    `ball_mask` > 0 (11 Nov, adapt.metrica_ball_mask): additionally hide that fraction of the visible LIVE
    ball in blocks drawn from `lengths` (the target's gap lengths). Raw Metrica shows the ball in 97-100 %
    of live play and never in dead play, so "ball hidden" alone marks the dead ball; the target hides it
    in ~32 % of live play."""
    from . import dfl_referee
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    rc = dict(rc, corruption="none")
    ball_mask = float(ball_mask or 0.0)
    key = (game, "corrupted", config_mod.digest({"rc": rc, "f": fcfg["features"], "p": fcfg["preprocess"]}),
           label_variant(cfg), ball_mask, dead_fill(cfg), dead_drift(cfg))
    if key not in _MEMO:
        data = cached_game(game, cfg)
        if dead_fill(cfg):
            assert label_variant(cfg) and not rc.get("dr") and len(lengths), "metrica_dead_fill: ball labels + gap lengths"
            rcm = dict(rc, corruption="state")                              # the corruption's own dead / live fractions
            filled = fill_dead_ball(data)
            dd = dead_drift(cfg)
            if dd:
                filled = drift_dead_ball(filled, bar_dead_speeds(cfg), dd, int(rc["corruption_seed"]) + 11)
            corrupted, ckey = dfl_referee.corrupt(filled, rcm, lengths)
            ckey += f"_df_lv{label_variant(cfg)[1]}" + (f"_dd{dd}" if dd else "")
        elif ball_mask > 0:
            assert not rc.get("dr"), "metrica_ball_mask does not support domain randomisation"
            rcm = dict(rc, corruption="state", dead_fraction=0.0, live_fraction=ball_mask)
            corrupted, ckey = dfl_referee.corrupt(data, rcm, lengths)       # live / dead from the variant's labels
            ckey += f"_lv{label_variant(cfg)[1] if label_variant(cfg) else 0}"
        else:
            corrupted, ckey = dfl_referee.corrupt(_cached_game(game), rc, [])   # track only: cache shared by variants
        corrupted = _with_labels(corrupted, data.labels)
        feats = build_features(corrupted, fcfg, None, cache_key=ckey)
        _MEMO[key] = dict(data=data, corrupted=corrupted, feats=feats, key=ckey, aligned=data.labels,
                          runs=data.runs(), cfg=fcfg)
    return _MEMO[key]


def prepare(game, cfg):
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    key = (game, config_mod.digest({"f": fcfg["features"], "p": fcfg["preprocess"]}), label_variant(cfg))
    if key not in _MEMO:
        data = cached_game(game, cfg)
        feats = build_features(data, fcfg, None, cache_key=data.name)
        _MEMO[key] = dict(data=data, corrupted=data, feats=feats, key=data.name, aligned=data.labels,
                          runs=data.runs(), cfg=fcfg)
    return _MEMO[key]
