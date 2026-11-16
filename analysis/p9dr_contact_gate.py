"""P9dr: Mali's contact gate (his final decoder: decoding/contact.py turnover_adj with touch events from
features/groups/touch.py, ported here unchanged in logic) on v26's fused out-of-fold probabilities, DFL 7 + Metrica 3.
Free flight (ball seen, no player within 2.2 m, no touch within 10 frames) makes an H <-> A change dearer by
(24 - 14); a touch of the other team with a deflection >= 45 degrees makes it cheaper by (14 - 8.5) within +-5 frames.
Touch events are detected on the same degraded test copy our probabilities were computed on (p9a_oof: rates_for +
smooth 25 + dead_carry 0.25, seed 0).

    /home/mehdi/venvs/mali/bin/python analysis/p9dr_contact_gate.py

Rerun 13 Nov evening with EV_SMOOTH=13: touch events detected on a copy whose ball is smoothed over 13 frames (our ATD:
5.5 events / min, ball acceleration p90 26.6; smooth 25 copies: 3.4 and 17.4; smooth 13: 4.1-5.0 and 27-29), same rule.

Registered before the run (13 Nov): arms gate x1 (his values) and x0.5; adopt the better only if DFL is ahead at
1.6 s and neither DFL 3 s nor Metrica at either referee is behind.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_dfl, from_metrica_sample  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

KC = dict(sg_window=7, sg_order=2, jump_speed_ms=40.0, lag=3, min_dtheta_deg=20.0, min_dspeed_ms=3.0,
          kick_rate_per_min=12.0, reception_rate_per_min=8.0, touch_m=2.0)
CP = dict(free_flight_m=2.2, free_penalty=24.0, contact_penalty=8.5, deflect_rad=0.785)
FPS = 25


def rate_threshold(values, candidates, minutes, rate):
    vals = np.sort(values[candidates & np.isfinite(values)])[::-1]
    if len(vals) == 0:
        return np.inf
    k = int(round(rate * minutes))
    return float(vals[min(max(k, 1), len(vals)) - 1])


def _segments(vis, jump):
    starts = vis & (np.concatenate(([True], ~vis[:-1])) | jump)
    seg = np.cumsum(starts) * vis
    out = []
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(seg) != 0) + 1, [len(seg)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if seg[a] > 0:
            out.append((int(a), int(b)))
    return seg, out


def geometry(m):
    L, W = m["pitch"]
    B = np.stack([m["bx"] * L, m["by"] * W], 1).astype(np.float64)
    P = np.stack([m["px"] * L, m["py"] * W], -1).astype(np.float64)
    dist = np.hypot(P[:, :, 0] - B[:, 0:1], P[:, :, 1] - B[:, 1:2])
    big = np.where(np.isnan(dist), np.inf, dist)
    dh = big[:, m["slot_team"] == 0].min(1)
    da = big[:, m["slot_team"] == 1].min(1)
    return B, dh, da, np.minimum(dh, da)


def events(m, B, dh, da):
    n, vis = len(B), ~np.isnan(B[:, 0])
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(B[:, 0]), np.diff(B[:, 1])) * FPS
    seg, segments = _segments(vis, np.nan_to_num(step, nan=0.0) > KC["jump_speed_ms"])
    vel, acc = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    for a, b in segments:
        if b - a >= KC["sg_window"]:
            vel[a:b] = savgol_filter(B[a:b], KC["sg_window"], KC["sg_order"], deriv=1, delta=1 / FPS, axis=0)
            acc[a:b] = savgol_filter(B[a:b], KC["sg_window"], KC["sg_order"], deriv=2, delta=1 / FPS, axis=0)
    speed, amag = np.hypot(*vel.T), np.hypot(*acc.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (vel[:, 0] * acc[:, 0] + vel[:, 1] * acc[:, 1]) / speed
    lag = KC["lag"]
    same = np.zeros(n, bool)
    same[lag:-lag] = (seg[2 * lag:] == seg[:-2 * lag]) & (seg[lag:-lag] > 0)
    v_next, v_prev = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    v_next[:-lag], v_prev[lag:] = vel[lag:], vel[:-lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.sum(v_next * v_prev, axis=1) / (np.hypot(*v_next.T) * np.hypot(*v_prev.T))
    dtheta = np.where(same, np.degrees(np.arccos(np.clip(cosang, -1, 1))), np.nan)
    dspeed = np.where(same, np.hypot(*v_next.T) - np.hypot(*v_prev.T), np.nan)
    playing = m["half"] > 0
    minutes = np.sum(vis & playing) / FPS / 60.0
    lmax = np.zeros(n, bool)
    lmax[1:-1] = (amag[1:-1] >= amag[:-2]) & (amag[1:-1] > amag[2:])
    turn = (np.nan_to_num(dtheta) >= KC["min_dtheta_deg"]) | (np.abs(np.nan_to_num(dspeed)) >= KC["min_dspeed_ms"])
    cand_k = lmax & turn & playing
    kick = cand_k & (np.nan_to_num(amag) >= rate_threshold(amag, cand_k, minutes, KC["kick_rate_per_min"]))
    dec = -np.nan_to_num(a_tan, nan=0.0)
    lmin = np.zeros(n, bool)
    lmin[1:-1] = (dec[1:-1] >= dec[:-2]) & (dec[1:-1] > dec[2:])
    cand_r = lmin & (np.nan_to_num(dspeed) <= -KC["min_dspeed_ms"]) & playing & ~kick
    recv = cand_r & (dec >= rate_threshold(dec, cand_r, minutes, KC["reception_rate_per_min"]))
    near = np.where(np.isfinite(np.fmin(dh, da)), np.fmin(dh, da), 99.0) < KC["touch_m"]
    idx = np.flatnonzero((kick | recv) & near)
    team = np.where(dh[idx] <= da[idx], 0, 1)
    vec = lambda a, b: B[np.clip(b, 0, n - 1)] - B[np.clip(a, 0, n - 1)]
    v_in, v_out = vec(idx - 6, idx - 1), vec(idx + 1, idx + 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = np.sum(v_in * v_out, 1) / (np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1) + 1e-5)
    return idx, team, np.arccos(np.clip(np.nan_to_num(cs, nan=1.0), -1, 1))


def turnover_adj(m, scale=1.0):
    B, dh, da, dmin = geometry(m)
    fr, tm, defl = events(m, B, dh, da)
    n = len(B)
    out = np.zeros(n)
    near_touch = np.zeros(n, bool)
    for f in fr:
        near_touch[max(0, f - 10):f + 11] = True
    free = ~np.isnan(B[:, 0]) & (dmin > CP["free_flight_m"]) & ~near_touch
    out[free] = -(CP["free_penalty"] - 14.0)
    if len(fr) >= 2:
        ok = np.r_[False, tm[1:] != tm[:-1]] & (defl >= CP["deflect_rad"])
        for f in fr[ok]:
            out[max(0, f - 5):f + 6] = 14.0 - CP["contact_penalty"]
    return out * scale, dict(events_per_min=len(fr) / (np.sum(m["half"] > 0) / FPS / 60), free_share=free[m["half"] > 0].mean())


def decode(p_home, p_dead, g, b, adj):
    P = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], 1)
    out = np.full(len(g), 3, np.int8)
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, c in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        reg = None if adj is None else {"turnover_adj": adj[a:c]}
        out[a:c] = sm.decode_run(sm.emission_logprob(P[a:c], sm.CONFIG), sm.CONFIG, boundary=b[a:c], regime=reg)
    return out


def one(f):
    n = os.path.basename(f)
    z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
    g, lab, name = z["g"], z["labels"], n[4:-4]
    raw, key = (from_dfl(name[-6:]), name[-6:]) if "DFL" in name else (from_metrica_sample(int(name[-1])), "metrica")
    m = degrade(raw, dict(rates_for(key), smooth_frames=int(os.environ.get("EV_SMOOTH", 25)), dead_carry_p=0.25), seed=0)
    assert len(m["bx"]) == len(g) and np.array_equal(m["labels"], lab), name
    l3 = merged(lab, 3.0)
    b = np.nan_to_num(z["b"])
    p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
    M = theirs("mehmet_ali_topkara", name, g)
    team, dead = sg(0.75 * lg(p) + 0.25 * lg(M["team"])), (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
    sc = lambda c: [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
    r = {"base": sc(decode(team, dead, g, b, None))}
    info = None
    for s in (1.0, 0.5):
        adj, info = turnover_adj(m, s)
        r[s] = sc(decode(team, dead, g, b, adj))
    print(name, info, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    return name, r


def main():
    files = sorted(glob.glob(C("oof_tcn128d3", "*.npz")))
    with Pool(int(os.environ.get("NP", 4))) as pool:
        res = dict(pool.map(one, files))
    names = sorted(res)
    base = np.array([res[m]["base"] for m in names])
    for k in (1.0, 0.5):
        a = np.array([res[m][k] for m in names])
        out = [str(k)]
        for src in ("DFL", "metrica"):
            ix = [i for i, m in enumerate(names) if src in m]
            d = a[ix] - base[ix]
            out.append("{} {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *a[ix].mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), len(ix)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
