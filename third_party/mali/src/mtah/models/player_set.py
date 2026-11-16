"""Small team-wise DeepSets pool and compact anonymous player inputs; no attention."""

import numpy as np
import torch
from torch import nn

SET_SHAPE = (24, 9)  # up to 12 visible tracks per team, including the goalkeeper


class TeamSetPool(nn.Module):
    def __init__(self, n_feat=9, dim=16, capacity=12):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_feat - 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out_dim = 4 * dim + 2
        self.capacity = capacity

    def forward(self, tok):
        present = tok[..., -1] > .5
        safe = torch.where(present.unsqueeze(-1), tok, torch.zeros_like(tok))
        embedded = self.phi(safe[..., :-1])
        pooled = []
        for sign in (1, -1):
            mask = present & (safe[..., 4] * sign > 0)
            count = mask.sum(-1, keepdim=True).to(embedded.dtype)
            mean = (embedded * mask.unsqueeze(-1)).sum(-2) / count.clamp(min=1)
            maximum = embedded.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(-2)
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            pooled.extend((mean, maximum, count / self.capacity))
        return torch.cat(pooled, -1)


def compact(tokens, per_team=12):
    """Prefer present keepers, then nearby players. Lexical tie keys avoid slot-order bias.

    Tokens are already normalized/finite. Missing ball offsets are zero; in that case
    coordinate/velocity keys give a deterministic subset. Counts are of retained tracks;
    full visible counts remain available in the existing scalar feature branch.
    """
    out = np.zeros((len(tokens), 2 * per_team, tokens.shape[-1]), np.float32)
    distance = np.sum(tokens[..., 6:8] ** 2, -1)
    for team, sign in enumerate((1, -1)):
        valid = (tokens[..., -1] > .5) & (tokens[..., 4] * sign > 0)
        order = np.lexsort((tokens[..., 3], tokens[..., 2], tokens[..., 1], tokens[..., 0],
                            distance, -tokens[..., 5], (~valid).astype(np.int8)), axis=1)
        take = order[:, :min(per_team, tokens.shape[1])]
        chosen = np.take_along_axis(tokens, take[..., None], axis=1)
        good = np.take_along_axis(valid, take, axis=1)
        out[:, team * per_team:team * per_team + take.shape[1]] = np.where(good[..., None], chosen, 0.)
    return out


def token_view(data, cfg, runs, stride, p):
    from ..pipeline.hybrid import player_tokens
    if p.get("token_pool") != "deepsets_team":
        return player_tokens(data, cfg, runs, stride)
    raw = player_tokens(data, cfg, runs, stride, n_max=data.players.shape[1])
    return [compact(t.reshape(len(t), data.players.shape[1], 9)).reshape(len(t), -1) for t in raw]
