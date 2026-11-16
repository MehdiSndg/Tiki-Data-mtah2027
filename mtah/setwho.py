"""Possessor auxiliary head for the player-set member (P9x): besides the team logit, one logit per player slot of
mtah.setgru.player_set, trained to pick the labelled team's player nearest the true ball. Kept apart from
mtah.setgru so the production models' cache keys do not change.

    from mtah.setwho import possessor_target, set_net_who
    Y = possessor_target(m, true_bx, true_by)          # (n, 2 * K) one-hot, all zero in flight
    net = set_net_who(nft)                             # (B, T, 1 + 2 * K): team logit, then slot logits
"""

import numpy as np
import pandas as pd

from mtah.setgru import K, NFP

POSS_M = 1.5                 # m: the labelled team's player this close to the true ball holds it


def slot_columns(m, k=K):
    """The px column in each slot of mtah.setgru.player_set (same ball and ordering), (n, 2, k), -1 where empty."""
    L, W = m["pitch"]
    n = len(m["bx"])
    bx = np.asarray(pd.Series(np.asarray(m["bx"], np.float64) * L).ffill().bfill())
    by = np.asarray(pd.Series(np.asarray(m["by"], np.float64) * W).ffill().bfill())
    dist = np.hypot(np.asarray(m["px"], np.float64) * L - bx[:, None], np.asarray(m["py"], np.float64) * W - by[:, None])
    slots = np.full((n, 2, k), -1, np.int64)
    for t in (0, 1):
        cols = np.flatnonzero(m["slot_team"] == t)
        d = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols])
        order = np.argsort(d, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        slots[:, t, :order.shape[1]] = np.where(np.isfinite(d[rows, order]), cols[order], -1)
    return slots


def possessor_target(m, true_bx, true_by, k=K, radius=POSS_M):
    """(n, 2 * k) one-hot per frame: the slot of the labelled team's player nearest the true ball, when within `radius`
    m; all zero in flight, dead time or when that player is not tracked."""
    from mtah.degrade import pm
    L, W = m["pitch"]
    slots = slot_columns(m, k)
    n = len(slots)
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    d = np.hypot(X - np.asarray(true_bx, np.float64)[:, None] * L, Y - np.asarray(true_by, np.float64)[:, None] * W)
    y = np.zeros((n, 2, k), np.float32)
    lab = np.asarray(m["labels"])
    for t, code in ((0, pm.H), (1, pm.A)):
        rows = np.flatnonzero(lab == code)
        s = slots[rows, t]
        dt = np.where(s >= 0, np.take_along_axis(d[rows], np.maximum(s, 0), axis=1), np.inf)
        dt = np.nan_to_num(dt, nan=np.inf)
        best = np.argmin(dt, axis=1)
        hit = dt[np.arange(len(rows)), best] <= radius
        y[rows[hit], t, best[hit]] = 1
    return y.reshape(n, -1)


def set_net_who(nft, k=K, nfp=NFP, hidden=32):
    """mtah.setgru.set_net's shape plus one logit per slot from the slot's embedding and the GRU state."""
    import torch

    class SetWhoNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = torch.nn.Sequential(torch.nn.Linear(nfp, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
            self.inp = torch.nn.Linear(nft + 4 * hidden, 128)
            self.gru = torch.nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(256, 1)
            self.who = torch.nn.Sequential(torch.nn.Linear(hidden + 256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

        def forward(self, x):
            B, T, _ = x.shape
            s = self.phi(x[..., nft:].reshape(B, T, 2, k, nfp))
            pooled = torch.cat([s.mean(3), s.amax(3)], dim=-1).reshape(B, T, -1)
            h, _ = self.gru(torch.relu(self.inp(torch.cat([x[..., :nft], pooled], dim=-1))))
            s = s.reshape(B, T, 2 * k, -1)
            who = self.who(torch.cat([s, h[:, :, None].expand(B, T, 2 * k, h.shape[-1])], dim=-1)).squeeze(-1)
            return torch.cat([self.out(h), who], dim=-1)

    return SetWhoNet()
