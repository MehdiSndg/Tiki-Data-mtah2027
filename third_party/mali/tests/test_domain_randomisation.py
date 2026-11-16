"""v3_dr: target-free domain randomisation of the training corruption."""

import numpy as np

from mtah import config
from mtah.pipeline import dfl_referee


def test_draws_are_deterministic_per_seed_and_match_and_inside_the_ranges():
    rc = dict(dfl_referee.referee_cfg(config.load("ev_ext_met")), dr="v3")
    a, la = dfl_referee.sample_dr(rc, "idsse_X")
    b, lb = dfl_referee.sample_dr(rc, "idsse_X")
    c, _ = dfl_referee.sample_dr(rc, "idsse_Y")
    d, _ = dfl_referee.sample_dr(dict(rc, corruption_seed=11), "idsse_X")
    assert a == b and np.array_equal(la, lb) and a != c and a != d
    r = dfl_referee.DR_V3
    for k, v in r.items():
        if k.startswith("gap_"):
            continue
        assert (a[k] in v) if k in ("ball_smooth_frames", "player_smooth_frames") else v[0] <= a[k] <= v[1]
    assert la.min() >= r["gap_short_frames"][0] and la.max() <= r["gap_long_frames"][1]
    assert 0.08 < np.mean(la < r["gap_long_frames"][0]) < 0.25       # ~15 % short occlusions


def test_train_cfg_leaves_existing_configs_untouched_and_scoring_copies_fixed():
    ref = config.load("ev_ext_met")
    assert dfl_referee.train_cfg(ref) is ref
    dr = config.load("ev_ext_dr3")
    t = dfl_referee.train_cfg(dr, seed=13)
    assert dfl_referee.referee_cfg(t)["dr"] == "v3" and dfl_referee.referee_cfg(t)["corruption_seed"] == 13
    assert "dr" not in dfl_referee.referee_cfg(dr)                     # v1 referee copy stays fixed
    assert "dr" not in dfl_referee.referee_cfg(dfl_referee.eval_cfg(dr))  # v2b referee copy stays fixed
    assert not dr["dfl_mix_eval"] and dr["adapt"]["extra_draws"] == [11, 13]
