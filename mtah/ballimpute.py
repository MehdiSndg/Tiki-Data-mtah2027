"""Ball position from the players in tracking gaps (P9aa). P9y: the true ball put back only on the in-play frames of
gaps >= 10 s (6% of the lost frames) gives the team GRU +0.015 on DFL (5 / 7), and still about +0.012 with 2-4 m of
correlated noise; so a filler good to a few metres is worth trying. A player-set GRU (mtah.setgru.set_net, two
outputs) reads the team features, the players near the last seen ball and the gap (time since / until the ball was
seen, where it was seen next) and regresses the true ball's offset from its last sighting, smooth L1 on lost frames,
over 90 s windows (a long gap needs both its ends in view).

    from mtah.ballimpute import inputs, offset_target, train_imputer, predict_offset, fill
    X = inputs(F_team, m); y, w = offset_target(m, raw["bx"], raw["by"])
    net, mu, sd = train_imputer([(X, y, w, m["half"] > 0), ...], seed, "cuda")
    m_filled = fill(m, predict_offset(net, mu, sd, X, m["half"] > 0, "cuda"), min_gap_s=10)
"""

import numpy as np
import pandas as pd

FPS = 25
WIN, STRIDE, CHUNK, CTX = 90 * FPS, 30 * FPS, 90 * FPS, 45 * FPS
EPOCHS, BATCH, LR = 8, 16, 1e-3
SCALE = 10.0                 # m: the offset target is in tens of metres
CAP_S = 60.0


def _last_next(m):
    L, W = m["pitch"]
    x, y = np.asarray(m["bx"], np.float64) * L, np.asarray(m["by"], np.float64) * W
    seen = np.isfinite(x)
    idx = np.arange(len(x))
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().values
    nxt = pd.Series(np.where(seen, idx, np.nan)).bfill().values
    lx, ly = pd.Series(x).ffill().bfill().values, pd.Series(y).ffill().bfill().values
    nx, ny = pd.Series(x).bfill().ffill().values, pd.Series(y).bfill().ffill().values
    return seen, idx, last, nxt, lx, ly, nx, ny


def gap_features(m):
    """(n, 4): seconds since the last and until the next ball sighting (capped), and the next sighting minus the last
    one in m (x along the pitch, y across); all zero where the ball is seen."""
    seen, idx, last, nxt, lx, ly, nx, ny = _last_next(m)
    since = np.where(np.isfinite(last), idx - last, CAP_S * FPS) / FPS
    until = np.where(np.isfinite(nxt), nxt - idx, CAP_S * FPS) / FPS
    out = np.c_[np.minimum(since, CAP_S), np.minimum(until, CAP_S), nx - lx, ny - ly].astype(np.float32)
    out[seen] = 0
    return out


def inputs(F_team, m):
    from mtah.setgru import player_set
    return np.concatenate([F_team.astype(np.float32), gap_features(m), player_set(m)], axis=1)


def offset_target(m, true_bx, true_by):
    """(n, 2) true ball minus the last sighting, in SCALE m; weight 1 on lost frames with a true ball."""
    L, W = m["pitch"]
    seen, _, _, _, lx, ly, _, _ = _last_next(m)
    tx, ty = np.asarray(true_bx, np.float64) * L, np.asarray(true_by, np.float64) * W
    y = np.nan_to_num(np.c_[tx - lx, ty - ly] / SCALE).astype(np.float32)
    w = (~seen & np.isfinite(tx) & (np.asarray(m["half"]) > 0)).astype(np.float32)
    return y, w


def _net(nf):
    from mtah.grumodel import widen
    from mtah.setgru import K, NFP, set_net
    return widen(lambda n: set_net(n - 2 * K * NFP), nf, heads=2)


def _windows(X, y, w, g):
    out = []
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        for s in range(a, max(a + 1, b - WIN + 1), STRIDE):
            z = min(b, s + WIN)
            if w[s:z].any():
                out.append((X[s:z], y[s:z], w[s:z]))
    return out


def norm(X, mu, sd):
    return np.nan_to_num((X - mu) / sd).astype(np.float32).clip(-10, 10)


def train_imputer(data, seed, dev, epochs=EPOCHS):
    """data: list of (inputs, offset target, weight, graded). Returns (net, mu, sd)."""
    import torch
    allX = np.concatenate([d[0] for d in data])
    mu, sd = np.nanmean(allX, axis=0), np.nanstd(allX, axis=0) + 1e-6
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    wins = [w_ for X, y, w, g in data for w_ in _windows(norm(X, mu, sd), y, w, g)]
    net = _net(allX.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(epochs):
        order = rng.permutation(len(wins))
        for a in range(0, len(order), BATCH):
            batch = [wins[i] for i in order[a:a + BATCH]]
            T = max(len(b[0]) for b in batch)
            Xb = np.zeros((len(batch), T, allX.shape[1]), np.float32)
            Yb = np.zeros((len(batch), T, 2), np.float32)
            Wb = np.zeros((len(batch), T), np.float32)
            for i, (x, y, w) in enumerate(batch):
                Xb[i, :len(x)], Yb[i, :len(x)], Wb[i, :len(x)] = x, y, w
            Xb, Yb, Wb = (torch.from_numpy(v).to(dev) for v in (Xb, Yb, Wb))
            loss = (torch.nn.functional.smooth_l1_loss(net(Xb), Yb, reduction="none").sum(-1) * Wb).sum() / Wb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval(), mu, sd


def predict_offset(net, mu, sd, X, g, dev):
    """(n, 2) predicted offset (SCALE m) on graded frames, 0 elsewhere; CHUNK frames at a time with CTX of context."""
    import torch
    X = norm(X, mu, sd)
    out = np.zeros((len(X), 2), np.float32)
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    with torch.no_grad():
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            for s in range(a, b, CHUNK):
                lo, hi = max(a, s - CTX), min(b, s + CHUNK + CTX)
                q = net(torch.from_numpy(np.ascontiguousarray(X[lo:hi][None])).to(dev)).cpu().numpy()[0]
                z = min(b, s + CHUNK)
                out[s:z] = q[s - lo:z - lo]
    return out


def fill(m, offset, min_gap_s=10.0):
    """A copy of m with the lost ball of every gap of >= min_gap_s inside a half set to its last sighting + offset,
    clipped to the pitch."""
    L, W = m["pitch"]
    seen, _, _, _, lx, ly, _, _ = _last_next(m)
    lost = ~seen & (np.asarray(m["half"]) > 0)
    use = np.zeros(len(lost), bool)
    e = np.diff(np.r_[0, lost.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if (b - a) / FPS >= min_gap_s:
            use[a:b] = True
    bx, by = np.asarray(m["bx"], np.float32).copy(), np.asarray(m["by"], np.float32).copy()
    bx[use] = np.clip((lx[use] + offset[use, 0] * SCALE) / L, 0, 1)
    by[use] = np.clip((ly[use] + offset[use, 1] * SCALE) / W, 0, 1)
    out = dict(m)
    out["bx"], out["by"] = bx, by
    return out
