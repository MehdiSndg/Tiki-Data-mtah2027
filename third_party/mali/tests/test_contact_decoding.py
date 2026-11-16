"""Gün 5 (Ek M.12): contact-gated decoding helpers (decoding.contact)."""
from types import SimpleNamespace

import numpy as np

from mtah.constants import A, D, H
from mtah.decoding import contact
from mtah.decoding.hsmm import decode_run


def _toy(n=600):
    ball = np.c_[np.linspace(10, 90, n), np.full(n, 34.0)].astype(np.float32)
    players = np.full((n, 4, 2), 50.0, np.float32)
    players[:, 0] = ball + [0.5, 0.0]                   # a home player next to the ball ...
    players[200:300, 0] = ball[200:300] + [10.0, 0.0]   # ... except during a free flight
    return SimpleNamespace(ball=ball, players=players, team=np.array([0, 0, 1, 1]), pitch=(105.0, 68.0), name="toy",
                           n_frames=n)


def test_turnover_adj_free_flight_dearer_and_contact_cheaper():
    d = _toy()
    ev = dict(frame=np.array([100, 400, 450]), team=np.array([H, A, H]), deflect_rad=np.array([0.1, 1.2, 0.2]),
              collinearity=np.array([np.nan, 0.1, np.nan]), n=600)
    cp = contact.DEFAULTS
    adj = contact.turnover_adj(ev, d, 600, cp)
    assert adj[250] == -(cp["free_penalty"] - 14.0)      # free flight: penalty 14 -> 24
    assert adj[400] == 14.0 - cp["contact_penalty"]      # switch with deflection: 14 -> 8.5
    assert adj[450] == 0.0                               # switch without deflection: unchanged
    assert adj[50] == 0.0


def test_regime_for_is_off_by_default():
    assert contact.regime_for(_toy(), {"decoder": {}}) is None


def test_negative_turnover_adj_blocks_a_weak_switch():
    from mtah import config
    cfg = config.load("main")
    n = 600
    lp = np.log(np.tile([0.7, 0.29, 0.01], (n, 1)))
    lp[300:] = np.log([0.35, 0.64, 0.01])
    free = decode_run(lp, cfg)
    assert (free[:290] == H).all() and (free[310:] == A).all()
    locked = decode_run(lp, cfg, regime={"turnover_adj": np.full(n, -200.0)})
    assert (locked == H).all()


def test_recover_inserts_supported_short_spell_only():
    d = _toy(800)
    d.ball[:, 0] = np.linspace(10, 30, 800)
    lab = np.full(800, H, np.int8)
    pb = np.tile([0.8, 0.19, 0.01], (800, 1))
    pb[300:330] = [0.2, 0.79, 0.01]                       # booster sees A for 30 frames
    ev = dict(frame=np.array([300, 330, 600, 620]), team=np.array([A, H, A, H]),
              deflect_rad=np.array([1.0, 0.2, 1.0, 0.2]), collinearity=np.array([np.nan, 0.1, 0.1, np.nan]), n=800)
    out = contact.recover(lab, [(0, 800)], pb, pb, ev, d, np.zeros(800), dict(contact.DEFAULTS), info := {})
    assert (out[300:330] == A).all() and (out[:300] == H).all() and (out[330:] == H).all()
    assert info["n_short_inserted"] == 1                 # 600-620: no booster support -> kept H


def test_snap_moves_boundary_to_new_team_touch_without_changing_segments():
    d = _toy(400)
    lab = np.full(400, H, np.int8)
    lab[200:] = A
    ev = dict(frame=np.array([188, 260]), team=np.array([A, H]), deflect_rad=np.zeros(2), collinearity=np.full(2, np.nan), n=400)
    out = contact.snap_to_touch(lab, [(0, 400)], ev, d, window=25)
    assert (out[:188] == H).all() and (out[188:] == A).all()
    assert np.count_nonzero(np.diff(out)) == np.count_nonzero(np.diff(lab))
    assert D not in out
