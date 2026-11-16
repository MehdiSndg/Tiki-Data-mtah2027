"""Player-level input for the team GRU: each team's K players nearest the ball, as a set. (P7d)

    from mtah.setgru import player_set, SetNet
    P = player_set(m)                  # (n, 2 * K * NFP): home block then away block, nearest first
    F = np.concatenate([features(m), P], axis=1)

mtah.grumodel's GRU sees team summaries (distances, counts, chase and touch features). Here every
player near the ball is given to it as well: position relative to the ball (x along the home attack),
distance, speed, the speed at which it closes on the ball, and a presence flag. The ball is the last
seen one (the first seen one before it); slots are sorted by distance, so the set carries no identity.
SetNet encodes each player with a shared MLP, pools each team by mean and max, and feeds the pooled
vectors with the team features into mtah.grumodel's GRU shape.
"""

import numpy as np
import pandas as pd

K = 11
NFP = 6                      # rel_x, rel_y, dist, speed, closing, present
VEL_FRAMES = 5               # velocity over 0.2 s


def player_set(m, k=K):
    L, W = m["pitch"]
    n = len(m["bx"])
    sign = np.zeros(n)
    for h, s in m["sign"].items():
        sign[m["half"] == h] = s
    bx = np.asarray(pd.Series(np.asarray(m["bx"], np.float64) * L).ffill().bfill())
    by = np.asarray(pd.Series(np.asarray(m["by"], np.float64) * W).ffill().bfill())
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    vx = np.full_like(X, np.nan)
    vy = vx.copy()
    vx[VEL_FRAMES:] = (X[VEL_FRAMES:] - X[:-VEL_FRAMES]) * 25 / VEL_FRAMES
    vy[VEL_FRAMES:] = (Y[VEL_FRAMES:] - Y[:-VEL_FRAMES]) * 25 / VEL_FRAMES
    rx, ry = X - bx[:, None], Y - by[:, None]
    dist = np.hypot(rx, ry)
    closing = -(rx * vx + ry * vy) / np.maximum(dist, 0.1)
    per = np.stack([rx * sign[:, None], ry * np.where(by[:, None] >= W / 2, 1.0, -1.0), dist, np.hypot(vx, vy), closing, np.ones_like(dist)], axis=-1)
    out = np.full((n, 2, k, NFP), np.nan, np.float32)
    out[..., -1] = 0
    for t in (0, 1):
        cols = np.flatnonzero(m["slot_team"] == t)
        d = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols])
        order = np.argsort(d, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        sel = per[:, cols][rows, order]
        ok = np.isfinite(d[rows, order])
        sel[~ok] = np.nan
        sel[..., -1] = ok
        out[:, t, :order.shape[1]] = sel
    return out.reshape(n, -1)


def set_net(nft, k=K, nfp=NFP, hidden=32, gru=128, layers=2):
    """gru, layers: the recurrent part's size (P9av tries 256 x 3, as the big team GRU of v17)."""
    import torch

    class SetNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = torch.nn.Sequential(torch.nn.Linear(nfp, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
            self.inp = torch.nn.Linear(nft + 4 * hidden, gru)
            self.gru = torch.nn.GRU(gru, gru, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(2 * gru, 1)

        def forward(self, x):
            B, T, _ = x.shape
            s = self.phi(x[..., nft:].reshape(B, T, 2, k, nfp))
            pooled = torch.cat([s.mean(3), s.amax(3)], dim=-1).reshape(B, T, -1)
            h, _ = self.gru(torch.relu(self.inp(torch.cat([x[..., :nft], pooled], dim=-1))))
            return self.out(h).squeeze(-1)

    return SetNet()
