"""Gün 4 (rapor 5 §3.2-3.3, Sıra 2; Ek M.5): physical regime indicators -> input-driven HSMM.

R_slow_open: ball hidden, 1.1 <= p80 < 2.6 m/s, coupled block shift and direction coherence high,
             no set-piece evidence -> slow open circulation (alive).
S_setpiece:  ball-exit memory, a slow player on a pitch line, Smart Tagging set piece (BAR only)
             with low coupled shift -> stoppage / restart preparation.
Per-frame log-score adjustments for decoding.hsmm.decode_run(regime=...):
  stoppage_adj = k_exit * E_out - k_slow * R_slow_open     (penalty 10 -> 10 - 6 E_out + 8 R_slow)
  restart_adj  = -k_set * S_setpiece * [ball speed < 3 m/s] (penalty 6 -> 6 + 8 S_setpiece)
Smoothed 3 s windows (_m75) of ck / tempo are used instead of instantaneous values.
"""

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _col(f, name, default=0.0):
    return np.nan_to_num(f[name].to_numpy(np.float64), nan=default) if name in f else np.full(len(f), default)


def indicators(f):
    """(R_slow_open, S_setpiece) per frame from a feature frame with core + ck (+ tags)."""
    vis = _col(f, "ball_vis") > 0.5
    p80 = _col(f, "pl_p80_speed_m75")
    a = _col(f, "ck_adv_mag_m75")
    phi = _col(f, "ck_phi_max_m75")
    e_out = _col(f, "ck_ball_exit")
    spot = _col(f, "ck_restart_spot")
    tag = _col(f, "tag_sp_in")
    s_set = np.clip(0.6 * e_out + 0.5 * spot + 0.5 * tag * _sig((0.9 - a) / 0.25), 0.0, 1.0)
    r_slow = (~vis) * ((p80 >= 1.1) & (p80 < 2.6)) * _sig((a - 1.1) / 0.3) * _sig((phi - 0.55) / 0.1) * (1 - s_set)
    return r_slow, s_set


def adjustments(f, k_exit=6.0, k_slow=8.0, k_set=8.0):
    """{'stoppage_adj', 'restart_adj'} per frame (log-score added to those transitions)."""
    r_slow, s_set = indicators(f)
    e_out = _col(f, "ck_ball_exit")
    slow_ball = _col(f, "ball_speed", default=0.0) < 3.0           # hidden ball counts as slow
    return {"stoppage_adj": np.round(k_exit * e_out - k_slow * r_slow, 6),
            "restart_adj": np.round(-k_set * s_set * slow_ball, 6)}
