"""Decoder behaviour on synthetic emissions."""

import numpy as np

from mtah import config
from mtah.constants import A, D, H
from mtah.decoding.hsmm import decode_run, emission_logprob


def _probs(labels, conf=0.9):
    p = np.full((len(labels), 3), (1 - conf) / 2)
    p[np.arange(len(labels)), labels] = conf
    return p


def _cfg():
    return config.load("m0")


def test_flicker_is_removed():
    lab = np.full(2000, H)
    lab[1000:1004] = A            # 4-frame flicker: below the possession minimum
    out = decode_run(emission_logprob(_probs(lab), _cfg()), _cfg())
    assert np.all(out == H)


def test_long_segments_are_kept_with_exact_boundaries():
    lab = np.concatenate([np.full(800, H), np.full(400, D), np.full(900, A)])
    out = decode_run(emission_logprob(_probs(lab), _cfg()), _cfg())
    assert np.array_equal(out, lab)


def test_h_d_h_stays_two_possessions():
    lab = np.concatenate([np.full(600, H), np.full(300, D), np.full(600, H)])
    out = decode_run(emission_logprob(_probs(lab), _cfg()), _cfg())
    assert np.array_equal(out, lab)


def test_numba_and_numpy_paths_agree():
    import pytest
    from mtah.decoding import hsmm
    if not hsmm.HAVE_NUMBA:
        pytest.skip("numba not installed")
    stride, dur, edge, dmin, trans = hsmm._tables(_cfg())
    rng = np.random.default_rng(0)
    for _ in range(3):
        lab = rng.choice([H, A, D], size=60).repeat(rng.integers(5, 400, size=60))[:6000]
        p = _probs(lab, conf=0.7) + rng.uniform(0, 0.3, size=(len(lab), 3))
        lp = emission_logprob(p / p.sum(axis=1, keepdims=True), _cfg())
        S = len(lp) // stride
        step = lp[:S * stride].reshape(S, stride, 3).sum(axis=1)
        bonus = np.zeros(S + 1)
        tadj = np.where(rng.uniform(size=(S + 1, 3)) < 0.1, rng.uniform(-8, 8, size=(S + 1, 3)), 0.0)
        cstep = (rng.uniform(size=S + 1) < 0.1).astype(np.int64)
        dur_c = hsmm.duration_logprior(dict(_cfg()['decoder']['duration']['possession'], median=40, sigma=0.6),
                                       stride, 4.0, dur.shape[1] - 1)
        fast = hsmm._viterbi(step, dur, edge, dmin, trans, bonus, tadj, cstep, dur_c)
        slow = hsmm._viterbi_numpy(step, dur, edge, dmin, trans, bonus, tadj, cstep, dur_c)
        assert np.array_equal(fast, slow)


def test_forced_dead_overrides_emissions():
    lab = np.full(1500, H)
    forced = np.zeros(1500, bool)
    forced[:400] = True
    out = decode_run(emission_logprob(_probs(lab), _cfg()), _cfg(), forced)
    assert np.all(out[:400] == D) and np.all(out[400:] == H)


def test_emission_scale_and_boundary_defaults_are_no_ops():
    """Gün 3 decoder knobs: beta = 1 and gamma = 0 (or no boundary) reproduce the old decoder."""
    rng = np.random.default_rng(1)
    lab = rng.choice([H, A, D], size=40).repeat(rng.integers(20, 300, size=40))[:5000]
    p = _probs(lab, conf=0.6) + rng.uniform(0, 0.4, size=(len(lab), 3))
    p /= p.sum(axis=1, keepdims=True)
    base = decode_run(emission_logprob(p, _cfg()), _cfg())
    c = _cfg()
    c["decoder"]["emission_scale"] = 1.0
    c["decoder"]["boundary_gamma"] = 0.0
    bnd = rng.uniform(0, 1, len(lab))
    assert np.array_equal(base, decode_run(emission_logprob(p, c), c, None, bnd))
    c["decoder"]["boundary_gamma"] = 1.0
    assert np.array_equal(base, decode_run(emission_logprob(p, c), c, None, None))


def test_boundary_bonus_recovers_a_short_possession():
    """An ambiguous short A spell inside H is kept only when the boundary head fires at its edges."""
    lab = np.full(3000, H)
    lab[1500:1550] = A                          # 2 s opponent possession
    p = _probs(lab, conf=0.9)
    p[1500:1550] = [0.19, 0.76, 0.05]           # A evidence just short of 2 turnovers + duration cost
    c = _cfg()
    assert np.all(decode_run(emission_logprob(p, c), c) == H)
    bnd = np.full(3000, 0.02)
    bnd[1500:1505] = 0.95                       # step grid: frame 1500 = step 300
    bnd[1550:1555] = 0.95
    c["decoder"]["boundary_gamma"] = 3.0
    out = decode_run(emission_logprob(p, c), c, None, bnd)
    assert (out[1505:1545] == A).all() and (out[:1400] == H).all() and (out[1650:] == H).all()


def test_emission_scale_sharpens():
    lab = np.full(3000, H)
    lab[1500:1560] = A
    p = _probs(lab, conf=0.9)
    p[1500:1560] = [0.40, 0.55, 0.05]
    c = _cfg()
    c["decoder"]["emission_scale"] = 8.0
    out = decode_run(emission_logprob(p, c), c)
    assert (out[1510:1550] == A).all()


def test_contested_turnover_discount_is_local_and_off_by_default():
    """Gün 3 (Ek L.10): a short opponent spell with moderate evidence is kept only where the
    contested mask lowers the turnover penalty; without the key the decoder is unchanged."""
    lab = np.full(3000, H)
    lab[1500:1650] = A                          # 6 s opponent spell, evidence just short of 2 x 14
    p = _probs(lab, conf=0.9)
    p[1500:1650] = [0.37, 0.56, 0.07]
    c = _cfg()
    base = decode_run(emission_logprob(p, c), c)
    assert np.all(base == H)
    mask = np.zeros(3000, bool)
    mask[1495:1655] = True
    assert np.array_equal(base, decode_run(emission_logprob(p, c), c, None, None, mask))   # key unset
    c["decoder"]["contested_turnover"] = 2.0
    out = decode_run(emission_logprob(p, c), c, None, None, mask)
    assert (out[1510:1640] == A).all() and (out[:1400] == H).all()
    far = np.zeros(3000, bool)
    far[100:140] = True                         # discount somewhere else: no effect here
    assert np.all(decode_run(emission_logprob(p, c), c, None, None, far)[1400:1700] == H)


def test_contested_duration_mixture_opens_a_short_spell():
    """A 1 s opponent spell is blocked by the controlled-possession duration prior (median 250)
    even without any turnover penalty; the short contested prior at contested steps opens it."""
    lab = np.full(3000, H)
    lab[1500:1525] = A
    p = _probs(lab, conf=0.9)
    p[1500:1525] = [0.20, 0.75, 0.05]
    c = _cfg()
    c["decoder"]["emission_scale"] = 1.75       # main's blend decoder
    mask = np.zeros(3000, bool)
    mask[1495:1530] = True
    c["decoder"]["contested_turnover"] = 6.0
    assert np.all(decode_run(emission_logprob(p, c), c, None, None, mask) == H)
    c["decoder"]["contested_duration"] = {"median": 40, "sigma": 0.6}
    out = decode_run(emission_logprob(p, c), c, None, None, mask)
    assert (out[1505:1520] == A).all() and (out[:1400] == H).all() and (out[1600:] == H).all()


def test_input_driven_stoppage_penalty_keeps_slow_live_play_alive():
    """Gün 4 (Ek M.5): a higher local stoppage penalty keeps a weakly-dead dip inside a
    possession alive; regime=None / zero arrays reproduce the static decoder."""
    lab = np.full(3000, H)
    p = _probs(lab, conf=0.9)
    p[1400:1650] = [0.38, 0.05, 0.57]          # 10 s of mild dead evidence (just above the static cost)
    c = _cfg()
    base = decode_run(emission_logprob(p, c), c)
    assert (base[1450:1600] == D).all()
    zero = {"stoppage_adj": np.zeros(3000), "restart_adj": np.zeros(3000)}
    assert np.array_equal(base, decode_run(emission_logprob(p, c), c, regime=zero))
    adj = np.zeros(3000)
    adj[1300:1750] = -25.0                     # stoppage penalty +25 here
    out = decode_run(emission_logprob(p, c), c, regime={"stoppage_adj": adj, "restart_adj": np.zeros(3000)})
    assert (out == H).all()
