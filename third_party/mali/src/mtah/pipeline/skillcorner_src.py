"""SkillCorner open games as a training source for the temporal net (Kasım 2026, external data only).

Same dict as metrica_src.prepare / dfl_referee.prepare:
  data / corrupted  raw broadcast track (players extrapolated off camera, ball only where detected)
  feats             source-common feature groups, cached under skillcorner_<id>
  aligned           phase-of-play labels (io/skillcorner.py; D = no phase)
prepare_corrupted adds the BAR-like player degradation of the evaluation corruption WITHOUT ball
masking (the broadcast ball already has gaps), exactly like metrica_src.prepare_corrupted.
SkillCorner is training data only: never a referee (the 4 gates stay DFL + Metrica).
"""

from .. import config as config_mod
from ..features import build_features, common_groups
from ..io.skillcorner import cached_game

_MEMO = {}


PFF = "pff_"      # 11 Nov: ids "pff_<game>" are PFF FC World Cup 2022 games (io/pff.py) on the same training path


def _pff(game, cfg, rc, tag):
    """PFF game as a DFL-like source. The provider never shows the ball in dead play (the target shows it
    in ~42 % of the dead frames), so the dead ball is filled in (metrica_src.fill_dead_ball) and hidden
    again with the corruption's dead fraction; the live ball keeps its own broadcast gaps (34-47 %).
    Labels: touch-level events with short returns suppressed (35 frames, like the DFL net target)."""
    from .. import cloud
    from ..io import pff
    import numpy as np
    from .. import constants as C
    from ..labels.align import suppress_short_returns
    from ..labels.masking import mask_ball_by_state
    from . import dfl_referee, metrica_src
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    assert not rc.get("dr"), "PFF sources do not support domain randomisation"
    dead_fraction, seed = float(rc["dead_fraction"]), int(rc["corruption_seed"])
    rc = dict(rc, corruption="none")
    key = (str(game), tag, config_mod.digest({"rc": rc, "f": fcfg["features"], "p": fcfg["preprocess"]}),
           metrica_src.dead_drift(cfg))
    if key not in _MEMO:
        data = pff.cached_game(str(game)[len(PFF):])
        aligned = suppress_short_returns(data.labels, data.runs(), 35, data, True)
        filled = metrica_src.fill_dead_ball(data)
        dd = metrica_src.dead_drift(cfg)
        if dd:                                                          # wandering dead ball (metrica_dead_drift)
            filled = metrica_src.drift_dead_ball(filled, metrica_src.bar_dead_speeds(cfg), dd, seed + 11)
        _, hide = mask_ball_by_state(filled, np.random.default_rng(seed), data.labels, dead_fraction, 0.0,
                                     cloud.bar_lengths(cfg))
        filled.ball[hide & (data.labels == C.D)] = np.nan            # dead frames only: no spill into live play
        corrupted, ckey = dfl_referee.corrupt(filled, rc, [])
        ckey += f"_dfd{dead_fraction}" + (f"_dd{dd}" if dd else "")
        feats = build_features(corrupted, fcfg, None, cache_key=ckey)
        _MEMO[key] = dict(data=data, corrupted=corrupted, feats=feats, key=ckey, aligned=aligned,
                          runs=data.runs(), cfg=fcfg)
    return _MEMO[key]


def name(game, cfg=None):
    """Match name; PFF games built with the wandering dead ball get their own name (booster-probability caches)."""
    from . import metrica_src
    dd = metrica_src.dead_drift(cfg or {}) if str(game).startswith(PFF) else None
    return f"skillcorner_{game}" + (f"_dd{dd}" if dd else "")


def prepare(game, cfg):
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    if str(game).startswith(PFF):
        from . import dfl_referee
        return _pff(game, cfg, dfl_referee.referee_cfg(cfg), "v1")
    key = (str(game), config_mod.digest({"f": fcfg["features"], "p": fcfg["preprocess"]}))
    if key not in _MEMO:
        data = cached_game(game)
        feats = build_features(data, fcfg, None, cache_key=data.name)
        _MEMO[key] = dict(data=data, corrupted=data, feats=feats, key=data.name, aligned=data.labels,
                          runs=data.runs(), cfg=fcfg)
    return _MEMO[key]


def prepare_corrupted(game, cfg, rc):
    from . import dfl_referee
    groups = common_groups(cfg["features"]["groups"])
    fcfg = config_mod.with_groups(cfg, groups)
    if str(game).startswith(PFF):
        return _pff(game, cfg, rc, "corrupted")
    rc = dict(rc, corruption="none")
    key = (str(game), "corrupted", config_mod.digest({"rc": rc, "f": fcfg["features"], "p": fcfg["preprocess"]}))
    if key not in _MEMO:
        data = cached_game(game)
        corrupted, ckey = dfl_referee.corrupt(data, rc, [])
        feats = build_features(corrupted, fcfg, None, cache_key=ckey)
        _MEMO[key] = dict(data=data, corrupted=corrupted, feats=feats, key=ckey, aligned=data.labels,
                          runs=data.runs(), cfg=fcfg)
    return _MEMO[key]
