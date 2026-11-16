"""Contact-gated decoding (Gün 5, rapor 6 'Sıra 1'; MASTER_PLAN Ek M.12). Off unless configured.

Three pieces, all on top of the semi-Markov decoder and all driven by the touch events of the
'tc' feature group (features/groups/touch.py: rate-calibrated kicks + receptions):
  turnover_adj     per-frame log-score added to H <-> A transitions: free flight (ball visible, no
                   player near, no touch nearby) makes a change dearer (14 -> 24), a verified
                   contact (team switch + deflection) cheaper (14 -> 8.5)
  dynamic_weight   per-frame booster weight of the log-pool blend: inside 8-55-frame windows that
                   start at a touch of the other team, where the booster sees the new team and the
                   net does not, the booster weight rises (0.25 -> 0.65)
  recover          after Viterbi: (a) 8-50-frame exchanges bounded by an opponent touch and a touch
                   of the original team, with booster support and not a collinear deflection, are
                   inserted as short possessions; (b) short dead segments inside a ball gap with
                   moderate P(dead) and live tempo are turned back into play.
Thresholds follow rapor 6 §4 and are config keys (decoder.contact). The decoder never reads GT.
"""

import numpy as np

from ..constants import A, D, H

DEFAULTS = dict(
    touch_m=2.0, free_flight_m=2.2, free_penalty=24.0, contact_penalty=8.5, gate_steps=1,
    deflect_rad=0.785, collinear_max=0.82, w_boost_dyn=0.65, dyn_min=8, dyn_max=55,
    dyn_boost_min=0.62, dyn_net_max=0.45, dyn_gap=0.25,
    min_short=8, max_short=50, min_flank=20, boost_max=0.68, boost_mean=0.55, vis_min=0.70,
    occl_max_len=450, occl_vis_max=0.15, occl_pdead_max=0.80, occl_tempo_z=-0.35,
)


def params(cfg):
    return dict(DEFAULTS, **((cfg.get("decoder") or {}).get("contact") or {}))


_EVENTS = {}


def regime_for(data, cfg):
    """Decoder regime dict with the contact-gated turnover adjustment when decoder.contact.gate is
    set (Ek M.12: free flight 14 -> 24, verified contact 14 -> 8.5), else None. Events are memoised
    per track name."""
    cp = params(cfg)
    if not cp.get("gate"):
        return None
    key = (data.name, data.n_frames)
    if key not in _EVENTS:
        _EVENTS[key] = touch_events(data, cfg)
    return {"turnover_adj": turnover_adj(_EVENTS[key], data, data.n_frames, cp)}


def touch_events(data, cfg):
    """Touch events of a TrackingData: dict of arrays (frame, team H/A code, deflect_rad,
    collinearity with the previous and next event, n_frames)."""
    from ..features.context import FeatureContext
    from ..features.groups.touch import DEFAULTS as TC_DEFAULTS
    from ..features.groups.touch import _events
    ctx = FeatureContext(data, cfg)
    kc = dict(TC_DEFAULTS, **cfg["features"].get("touch", {}))
    ev, _, _ = _events(ctx, kc)
    idx = np.flatnonzero(ev)
    dh, da = np.nan_to_num(ctx.d_home, nan=99.0), np.nan_to_num(ctx.d_away, nan=99.0)
    team = np.where(dh[idx] <= da[idx], H, A).astype(np.int8)
    B = data.ball.astype(np.float64)
    n = len(B)

    def vec(a, b):
        a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
        return B[b] - B[a]

    v_in, v_out = vec(idx - 6, idx - 1), vec(idx + 1, idx + 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v_in * v_out, 1) / (np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1) + 1e-5)
    deflect = np.arccos(np.clip(np.nan_to_num(cos, nan=1.0), -1, 1))
    coll = np.full(len(idx), np.nan)
    if len(idx) >= 3:
        disp = B[idx[2:]] - B[idx[:-2]]
        vo = v_out[:-2]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.sum(vo * disp, 1) / (np.linalg.norm(vo, axis=1) * np.linalg.norm(disp, axis=1) + 1e-5)
        coll[1:-1] = c
    return dict(frame=idx, team=team, deflect_rad=deflect, collinearity=coll, n=n)


def turnover_adj(events, data, n, cp):
    """Per-frame log-score added to H <-> A transitions (positive = cheaper), base penalty from
    decoder.penalty_turnover is applied by the decoder; this returns base - target."""
    out = np.zeros(n)
    vis = ~np.isnan(data.ball[:, 0])
    dist = np.linalg.norm(data.players - data.ball[:, None, :], axis=-1)
    with np.errstate(all="ignore"):
        dmin = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    near_touch = np.zeros(n, bool)
    for f in events["frame"]:
        near_touch[max(0, f - 10):f + 11] = True
    free = vis & (dmin > cp["free_flight_m"]) & ~near_touch
    out[free] = -(cp["free_penalty"] - 14.0)
    fr, tm = events["frame"], events["team"]
    if len(fr) >= 2:
        switch = np.r_[False, tm[1:] != tm[:-1]]
        ok = switch & (events["deflect_rad"] >= cp["deflect_rad"])
        for f in fr[ok]:
            out[max(0, f - 5):f + 6] = 14.0 - cp["contact_penalty"]
    return out


def dynamic_weight(events, p_boost, p_net, w_base, n, cp):
    """(T, 1) booster weight: w_base, raised to w_boost_dyn inside short opponent windows where
    the booster sees the new team and the net does not."""
    w = np.full((n, 1), float(w_base))
    fr, tm = events["frame"], events["team"]
    for k in range(len(fr) - 1):
        a, b = fr[k], fr[k + 1]
        if tm[k] == tm[k + 1] or not (cp["dyn_min"] <= b - a <= cp["dyn_max"]):
            continue
        new = int(tm[k])
        pb = p_boost[a:b, new] / np.maximum(p_boost[a:b, 0] + p_boost[a:b, 1], 1e-9)
        pn = p_net[a:b, new] / np.maximum(p_net[a:b, 0] + p_net[a:b, 1], 1e-9)
        if pb.mean() >= cp["dyn_boost_min"] and pn.mean() < cp["dyn_net_max"] and pb.mean() - pn.mean() > cp["dyn_gap"]:
            w[a:b] = cp["w_boost_dyn"]
    return w


def snap_to_touch(labels, runs, events, data, window=25, before=None, min_seg=8, info=None):
    """Move every H <-> A boundary (ball visible) to the nearest touch event of the NEW team within
    [t - before, t + window] (before defaults to window). Never creates or deletes a segment."""
    lab = labels.copy()
    before = window if before is None else before
    vis = ~np.isnan(data.ball[:, 0])
    fr, tm = events["frame"], events["team"]
    moved = 0
    for a, b in runs:
        t = a + 1
        while t < b:
            if lab[t] != lab[t - 1] and {int(lab[t]), int(lab[t - 1])} == {H, A} and vis[max(a, t - 5):t + 6].mean() > 0.5:
                new, old = int(lab[t]), int(lab[t - 1])
                s_old = t - 1
                while s_old > a and lab[s_old - 1] == old:
                    s_old -= 1
                e_new = t
                while e_new < b and lab[e_new] == new:
                    e_new += 1
                lo, hi = max(t - before, s_old + min_seg), min(t + window, e_new - min_seg)
                i = np.flatnonzero((fr >= lo) & (fr <= hi) & (tm == new))
                if len(i):
                    c = int(fr[i[np.argmin(np.abs(fr[i] - t))]])
                    if c < t:
                        lab[c:t] = new
                    elif c > t:
                        lab[t:c] = old
                    moved += c != t
                    t = max(c, t) + 1
                    continue
            t += 1
    if info is not None:
        info["n_snapped"] = moved
    return lab


def _segments(lab, a, b):
    out = []
    s = a
    for t in range(a + 1, b + 1):
        if t == b or lab[t] != lab[s]:
            out.append([int(lab[s]), s, t])
            s = t
    return out


def recover(labels, runs, p_boost, p_blend, events, data, tempo_z, cp, info=None):
    """Post-Viterbi short-spell insertion + occlusion alive-hold (rapor 6 §4), per graded run."""
    lab = labels.copy()
    vis = (~np.isnan(data.ball[:, 0])).astype(float)
    L, W = data.pitch
    bx, by = data.ball[:, 0], data.ball[:, 1]
    with np.errstate(invalid="ignore"):
        oob = np.nan_to_num(np.maximum.reduce([-bx, bx - L, -by, by - W]), nan=-1.0)
    fr, tm = events["frame"], events["team"]
    n_occl = n_short = 0
    for a, b in runs:
        segs = _segments(lab, a, b)
        # (b) occlusion alive-hold: false stoppages inside a ball gap
        for i, (st, s, e) in enumerate(segs):
            if st != D or e - s > cp["occl_max_len"] or i == 0 or i + 1 >= len(segs):
                continue
            pt, nt = segs[i - 1][0], segs[i + 1][0]
            if pt not in (H, A) or nt not in (H, A):
                continue
            if (vis[s:e].mean() < cp["occl_vis_max"] and oob[max(a, s - 10):min(b, e + 10)].max() <= 0.0
                    and p_blend[s:e, D].mean() < cp["occl_pdead_max"] and tempo_z[s:e].mean() > cp["occl_tempo_z"]):
                if pt == nt:
                    lab[s:e] = pt
                else:
                    diff = p_blend[s:e, H] - p_blend[s:e, A]
                    want = diff >= 0 if nt == H else diff < 0
                    cross = int(np.argmax(want)) if want.any() else (e - s) // 2
                    split = int(np.clip(s + cross, s + 5, e - 5))
                    lab[s:split], lab[split:e] = pt, nt
                n_occl += 1
        # (a) short-spell insertion inside long possessions
        for st, s, e in _segments(lab, a, b):
            if st not in (H, A) or e - s < cp["min_short"] + 2 * cp["min_flank"]:
                continue
            opp = A if st == H else H
            k = int(np.searchsorted(fr, s + cp["min_flank"]))
            cur = s
            while k < len(fr) and fr[k] <= e - cp["min_flank"]:
                if tm[k] != opp or fr[k] < cur + cp["min_flank"]:
                    k += 1
                    continue
                t_in, t_out, n_opp = int(fr[k]), None, 1
                m = k + 1
                while m < len(fr) and fr[m] - t_in <= cp["max_short"] and fr[m] <= e - cp["min_flank"]:
                    if tm[m] == opp:
                        n_opp += 1
                    elif fr[m] - t_in >= cp["min_short"]:
                        t_out = int(fr[m])
                        break
                    m += 1
                if t_out is not None:
                    pb = p_boost[t_in:t_out, opp] / np.maximum(p_boost[t_in:t_out, 0] + p_boost[t_in:t_out, 1], 1e-9)
                    coll = events["collinearity"][k]
                    ok = (vis[t_in:t_out].mean() >= cp["vis_min"] and pb.max() >= cp["boost_max"]
                          and pb.mean() >= cp["boost_mean"]
                          and not (np.isfinite(coll) and coll >= cp["collinear_max"])
                          and (events["deflect_rad"][k] >= cp["deflect_rad"] or n_opp >= 2))
                    if ok:
                        lab[t_in:t_out] = opp
                        n_short += 1
                        cur = t_out
                        k = int(np.searchsorted(fr, t_out))
                        continue
                k += 1
    if info is not None:
        info.update(n_occlusion_fixed=n_occl, n_short_inserted=n_short)
    return lab
