"""Hand-written emission model: a transparent reference for the learned one.

p_alive falls when players slow down and when the ball is outside the lines; p_home follows the
smoothed nearest-player advantage while the ball is visible, and the gap-edge evidence while not.
"""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def rule_probs(feats):
    p80 = feats["pl_p80_speed_m25"].to_numpy(np.float64)
    oob = np.nan_to_num(feats["ball_oob_flag_m25"].to_numpy(np.float64), nan=0.0)
    p_alive = _sigmoid(2.5 * (np.nan_to_num(p80, nan=2.0) - 1.9) - 6.0 * oob)

    vis = feats["ball_vis"].to_numpy() > 0
    ddiff = np.nan_to_num(feats["prox_ddiff_m25"].to_numpy(np.float64), nan=0.0)
    prev = np.nan_to_num(feats["gap_team_prev"].to_numpy(np.float64), nan=0.0)
    nxt = np.nan_to_num(feats["gap_team_next"].to_numpy(np.float64), nan=0.0)
    pos = np.nan_to_num(feats["gap_pos"].to_numpy(np.float64), nan=0.5)
    edge = (1 - pos) * prev + pos * nxt
    cen = np.nan_to_num(feats["team_cen_att_diff_m25"].to_numpy(np.float64), nan=0.0)
    z_home = np.where(vis, ddiff / 1.5, 2.0 * edge + cen / 10.0)
    return p_alive, _sigmoid(z_home)
