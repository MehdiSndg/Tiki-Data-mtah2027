"""High-precision weak labels (labelling functions) computed from tracking only.

M0 anchors:
  D  : ball visible and clearly outside the pitch (public window: >99% of such frames are D),
       plus any forced-dead span (BAR-ATH: second half before kick-off).
  H/A: ball visible, inside the pitch, and one team's player in close control while no opponent
       is near, sustained over most of a short window -- and the game must look live: ball
       moving and players not in the slow collective tempo of a stoppage. Without that last
       condition a player placing the ball for a goal kick or carrying it to a throw-in (ball
       slow, opponents far) looks exactly like "close control"; on the public window those
       frames were ~26% of the home anchors.

Adım 3 dead-ball anchors (config switches, off in m0.yaml so M0 stays reproducible):
  d_tempo       : the median speed of all players (1 s rolling median) stays below
                  d_tempo_max_ms for at least d_tempo_min_s; only the core of such a run is
                  anchored (d_tempo_erode_s cut at both ends, because tempo lags the restart and
                  the whistle). Ball-free, so it works in ATD gaps. Calibrated on DFL J03WPY real
                  labels (1.2 m/s, 10 s, 3 s: precision 0.990, dead coverage 0.50); the
                  OOB-only rule covered 8% of visible dead frames there (DFL places the ball ON
                  the line: OOB median 0.28 m).
  d_static_ball : ball visible and nearly static for K frames. Near-perfect on DFL (precision
                  0.99) because DFL freezes the dead ball, but it NEVER fires on BAR-ATH (0
                  frames on the public window: the ATD ball jitters/is carried in dead play).
                  A DFL artefact -- off by default; kept to measure that artefact.
  d_tags_tempo  : Smart Tagging set-piece window AND a shorter slow-tempo core (BAR-ATH only).
                  Off by default: no external labels to calibrate it (public-window precision
                  0.73-0.84 is a diagnostic, not a tuning target).
Frames where a new dead anchor meets a team anchor become X (ambiguous), never silently D.
Everything else is X (no anchor). Anchor quality is reported on labelled blocks by the pipeline.
"""

import warnings

import numpy as np
import pandas as pd

from ..constants import A, D, H, X
from ..preprocess import velocity

DEFAULTS = dict(d_tempo=False, d_tempo_max_ms=1.2, d_tempo_min_s=10.0, d_tempo_erode_s=3.0,
                d_tempo_window=25,
                d_static_ball=False, d_static_max_ms=0.5, d_static_min_frames=25,
                d_tags_tempo=False, d_tags_max_ms=1.2, d_tags_min_s=3.0, d_tags_erode_s=1.0)


def player_median_speed(data, window=25):
    """Median speed of all present players per frame, smoothed by a centred rolling median."""
    v = velocity(data.players, data.fps, 2)
    speed = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speed, axis=1)
    return pd.Series(med).rolling(int(window), center=True, min_periods=5).median().to_numpy()


def run_cores(mask, min_len, erode):
    """Frames of True-runs of length >= min_len, with `erode` frames cut at both ends."""
    out = np.zeros(len(mask), bool)
    edges = np.diff(np.concatenate(([0], np.asarray(mask, np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len and b - a > 2 * erode:
            out[a + erode:b - erode] = True
    return out


def _run_lengths(mask):
    out = np.zeros(len(mask), np.int32)
    edges = np.diff(np.concatenate(([0], np.asarray(mask, np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def dead_anchor_masks(feats, period, wc, data=None, fps=25):
    """{name: bool mask} of the Adım 3 dead-ball labelling functions that are switched on."""
    wc = dict(DEFAULTS, **wc)
    playing = np.asarray(period) > 0
    out = {}
    med = None
    if wc["d_tempo"] or wc["d_tags_tempo"]:
        if data is None:
            raise ValueError("d_tempo / d_tags_tempo need the tracking data (players)")
        med = player_median_speed(data, wc["d_tempo_window"])
    if wc["d_tempo"]:
        slow = playing & (np.nan_to_num(med, nan=99.0) < wc["d_tempo_max_ms"])
        out["d_tempo"] = run_cores(slow, int(wc["d_tempo_min_s"] * fps), int(wc["d_tempo_erode_s"] * fps))
    if wc["d_static_ball"]:
        vis = feats["ball_vis"].to_numpy() > 0
        speed = np.nan_to_num(feats["ball_speed"].to_numpy(), nan=99.0)
        static = playing & vis & (speed < wc["d_static_max_ms"])
        out["d_static_ball"] = _run_lengths(static) >= int(wc["d_static_min_frames"])
    if wc["d_tags_tempo"] and "tag_sp_in" in feats:
        slow = playing & (np.nan_to_num(med, nan=99.0) < wc["d_tags_max_ms"])
        core = run_cores(slow, int(wc["d_tags_min_s"] * fps), int(wc["d_tags_erode_s"] * fps))
        out["d_tags_tempo"] = core & (feats["tag_sp_in"].to_numpy() > 0)
    return out


CONSENSUS_COLUMNS = ("geo_compact_diff_m25", "team_adv_diff_m25", "tc_last_team", "prox_ddiff_m25")


def team_consensus(feats, codes, allowed, tcc):
    """Gün 5 (Ek N.10): BAR team anchors from the agreement of continuous team signals.

    On the blind pool the proximity anchors cover 11% of alive frames at 88% precision (9% fall on
    dead frames); unanimity of team compactness, team advance, last-touch team and proximity covers
    21% at 96% (2.5% dead), DFL alike (25%, 96%). Adds H/A where codes is still X, `allowed`
    (playing, live tempo) and at least `need` of the signals agree in sign. Then every team anchor
    within `mask_switch_frames` of a touch-sequence team switch (tc_last_team changes) becomes X:
    the slow collective signals turn seconds after a change, so anchors there teach the lag.
    Returns (codes, n_added, n_masked)."""
    cols = tuple(tcc.get("columns") or CONSENSUS_COLUMNS)
    missing = [c for c in cols if c not in feats]
    if missing:
        raise KeyError(f"team_consensus needs feature columns {missing} (groups core, geo, tc)")
    V = np.stack([np.sign(np.nan_to_num(feats[c].to_numpy(np.float64))) for c in cols])
    need = int(tcc.get("need", len(cols)))
    lab = np.where((V > 0).sum(0) >= need, H, np.where((V < 0).sum(0) >= need, A, X)).astype(np.int8)
    add = (codes == X) & allowed & (lab != X)
    codes = codes.copy()
    codes[add] = lab[add]
    n_masked = 0
    w = int(tcc.get("mask_switch_frames", 0))
    if w > 0 and "tc_last_team" in feats:
        lt = feats["tc_last_team"].to_numpy(np.float64)
        sw = np.flatnonzero((lt[1:] != lt[:-1]) & (lt[1:] != 0) & (lt[:-1] != 0)) + 1
        near = np.zeros(len(codes), bool)
        for t in sw:
            near[max(0, t - w):t + w + 1] = True
        drop = near & np.isin(codes, (H, A))
        codes[drop] = X
        n_masked = int(drop.sum())
    return codes, int(add.sum()), n_masked


def anchors(feats, period, cfg, forced_dead=(), data=None, info=None):
    """(codes, team_ok). `data` (TrackingData) is needed only by the player-tempo dead rules.
    If `info` is a dict, the frame count of every labelling function is written into it."""
    wc = cfg["weak_labels"]
    n = len(feats)
    vis = feats["ball_vis"].to_numpy() > 0
    oob = feats["ball_oob_dist"].to_numpy()
    d_h = feats["prox_dH"].to_numpy()
    d_a = feats["prox_dA"].to_numpy()
    playing = period > 0

    codes = np.full(n, X, np.int8)

    speed = np.nan_to_num(feats["ball_speed"].to_numpy(), nan=0.0)
    tempo = np.nan_to_num(feats["pl_p80_speed_m25"].to_numpy(), nan=0.0)
    moving = speed >= wc["live_min_ball_speed_ms"]
    # The tempo condition only gates which anchors train the TEAM model. Applying it to the
    # alive/dead model as well makes every alive example fast and every dead one slow, and the
    # model learns "slow tempo = dead" (full-match dead share jumped to 55%).
    team_ok = tempo >= wc["live_min_p80_speed_ms"]
    inside = vis & (oob < -wc["inside_margin_m"]) & moving
    rc, rf = wc["contact_radius_m"], wc["free_radius_m"]
    c_h = (inside & (d_h < rc) & (d_a > rf)).astype(np.float32)
    c_a = (inside & (d_a < rc) & (d_h > rf)).astype(np.float32)
    w = 2 * wc["contact_half_window"] + 1
    frac_h = pd.Series(c_h).rolling(w, center=True, min_periods=1).mean().to_numpy()
    frac_a = pd.Series(c_a).rolling(w, center=True, min_periods=1).mean().to_numpy()
    codes[playing & (c_h > 0) & (frac_h >= wc["contact_min_frac"])] = H
    codes[playing & (c_a > 0) & (frac_a >= wc["contact_min_frac"])] = A
    n_consensus = n_masked = 0
    tcc = wc.get("team_consensus") or {}
    if tcc.get("enabled"):
        codes, n_consensus, n_masked = team_consensus(feats, codes, playing & team_ok, tcc)
    team = np.isin(codes, (H, A))

    extra = dead_anchor_masks(feats, period, wc, data, int(getattr(data, "fps", 25) or 25))
    for m in extra.values():
        codes[m & ~team] = D
        codes[m & team] = X

    oob_dead = playing & vis & (oob > wc["dead_oob_min_m"])
    codes[oob_dead] = D
    for a, b in forced_dead:
        codes[a:b] = D
    if info is not None:
        info.update({"team_H": int((codes == H).sum()), "team_A": int((codes == A).sum()),
                     "team_consensus_added": int(n_consensus), "team_masked_near_switch": int(n_masked),
                     "d_oob": int(oob_dead.sum()), **{k: int(m.sum()) for k, m in extra.items()},
                     "D_total": int((codes == D).sum())})
    return codes, team_ok


def precision_report(codes, gt):
    """Per-class precision/coverage of anchors where gt is known (diagnostic only)."""
    lines = []
    known = gt != X
    for k, name in ((H, "H"), (A, "A"), (D, "D")):
        sel = known & (codes == k)
        prec = float(np.mean(gt[sel] == k)) if sel.any() else float("nan")
        cover = float(np.mean(codes[known & (gt == k)] == k)) if np.any(known & (gt == k)) else 0.0
        lines.append(f"    {name}: {int(sel.sum()):6d} anchor frames  precision {prec:.3f}  coverage {cover:.3f}")
    return "\n".join(lines)
