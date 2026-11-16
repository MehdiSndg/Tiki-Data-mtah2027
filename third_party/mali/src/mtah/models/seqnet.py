"""Temporal model family (Gün 2): dilated TCN + BiGRU, dual head (state + boundary).

Input per 5-frame step (5 Hz): the source-common frame features (per-match robust z-score,
block mean, NaN -> 0) + log of the gradient-boosting probabilities [H, A, D]. The net learns the
long-range hysteresis the frame model cannot see (who had the ball 30 s ago, how long the ball
has been dead) and refines the frame probabilities.

Heads: state logits (H, A, D) and a boundary logit (a label change at this step).
Loss  = CE(state)                                   on known steps
      + lambda_tmse * truncated MSE of log-prob changes between neighbouring steps (MS-TCN)
      + lambda_iou  * (1 - soft IoU) per class      (spell-level overlap, the metric's spirit)
      + lambda_bnd  * BCE(boundary, Gaussian-smoothed change targets)

Determinism: fixed seeds, torch.use_deterministic_algorithms(True), cuDNN deterministic, no
benchmark, CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA initialises). The same GPU + driver
+ torch version reproduces bit-for-bit; across GPU models it may not (documented risk).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULTS = dict(stride=5, chunk=512, chunk_step=256, hidden=128, gru_hidden=96, gru_layers=2,
                dilations=(1, 2, 4, 8, 16, 32), dropout=0.1, epochs=25, batch=16, lr=1e-3,
                weight_decay=1e-4, lambda_tmse=0.15, tmse_tau=4.0, lambda_iou=0.5,
                tmse_team_boundary_radius=-1,
                tmse_weak_support="all",
                lambda_bnd=0.5, bnd_sigma=1.0, bnd_pos_weight=8.0, n_seeds=3,
                heads="joint", lambda_touch=0.5,
                fine_dim=0, fine_hidden=48, lambda_fine=1.0, lambda_fine_tmse=0.3, fine_gate=2, fine_mix=0.0)
# fine_dim (Gün 4 / Sıra 3, rapor 5 §1.2-1.3): > 0 adds a 25 fps residual branch. Each 5 Hz step
# carries its 5 frames x fine_dim frame-level inputs (train(..., fine=...)); a 3-layer dilated
# Conv1d runs at 25 fps, is fused with the 5 Hz trunk (expand, not repeat_interleave: the
# backward is a plain sum, deterministic on CUDA) and gives 25 fps state logits; the output is the
# fine head's log-probs, optionally mixed with the upsampled coarse log-probs (fine_mix; rapor 5
# proposes 0.5, but a confident coarse head then caps a short opposite spell near 0.5 -- the F2
# erasure again -- so the default is 0: coarse context enters through the fusion input and the
# coarse loss stays as an auxiliary term). Fine loss:
# CE on frame labels + boundary-gated truncated MSE (no smoothing within +-fine_gate frames of a
# true label change). fine_dim 0 = the 5 Hz net of all models up to Gün 4.
# heads (Gün 3 / Adım 2): 'joint' = one 3-class state head (all models up to 23 Oct);
# 'split' = alive + team_ctrl (inference) + team_touch (DFL touch-level labels, training only,
# weighted by lambda_touch; train(..., touch=[per-step targets or None per sequence])).


def set_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # x (B, T, C)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + self.drop(F.gelu(y)))


class TokenPool(nn.Module):
    """Permutation-invariant attention pooling over player tokens (report 2: factorised
    space -> time). Each step's tokens (N, F; the last feature is the 'present' flag) are embedded
    by a small MLP; K learned queries attend over the present tokens; the K pooled vectors are
    concatenated. Slot order never matters (slots are tracklets, not people)."""

    def __init__(self, n_feat, dim, n_queries):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_feat, dim), nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Parameter(torch.randn(n_queries, dim) / dim ** 0.5)
        self.out_dim = n_queries * dim

    def forward(self, tok):                                 # tok (B, T, N, F)
        mask = tok[..., -1] > 0.5                           # (B, T, N)
        h = self.phi(tok)                                   # (B, T, N, d)
        score = torch.einsum("btnd,kd->btkn", h, self.q) / h.shape[-1] ** 0.5
        score = score.masked_fill(~mask[:, :, None, :], -1e4)
        att = torch.softmax(score, dim=-1)
        pooled = torch.einsum("btkn,btnd->btkd", att, h)
        any_present = mask.any(-1, keepdim=True)[..., None].float()   # all-missing steps -> 0
        return (pooled * any_present).flatten(2)


class SeqNet(nn.Module):
    def __init__(self, n_in, p):
        super().__init__()
        h = p["hidden"]
        self.tok_shape = tuple(p["token_shape"]) if p.get("use_tokens") else None
        if self.tok_shape:
            n_tok = self.tok_shape[0] * self.tok_shape[1]
            self.n_step = n_in - n_tok
            if p.get("token_pool") == "deepsets_team":
                from .player_set import TeamSetPool
                self.pool = TeamSetPool(self.tok_shape[1], int(p.get("token_dim", 16)), self.tok_shape[0] // 2)
            else:
                self.pool = TokenPool(self.tok_shape[1], int(p.get("token_dim", 32)), int(p.get("token_queries", 4)))
            n_in = self.n_step + self.pool.out_dim
        self.inp = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Dropout(p["dropout"]))
        self.tcn = nn.Sequential(*[TCNBlock(h, d, p["dropout"]) for d in p["dilations"]])
        self.gru = nn.GRU(h, p["gru_hidden"], num_layers=p["gru_layers"], batch_first=True,
                          bidirectional=True, dropout=p["dropout"])
        self.split = p.get("heads", "joint") == "split"
        if self.split:
            # Gün 3 / Adım 2 (source-specific heads): alive (dead vs alive, every source),
            # team_ctrl (H vs A given alive, CONTROLLED possession: DFL gt35, Metrica events, BAR
            # anchors) and team_touch (H / A / None=dead, DFL touch-level labels only, training
            # only). Inference composes [H, A, D] from alive + team_ctrl; team_touch never enters it.
            self.alive = nn.Linear(2 * p["gru_hidden"], 1)
            self.ctrl = nn.Linear(2 * p["gru_hidden"], 1)
            self.touch = nn.Linear(2 * p["gru_hidden"], 3)
        else:
            self.state = nn.Linear(2 * p["gru_hidden"], 3)
        self.bnd = nn.Linear(2 * p["gru_hidden"], 1)
        self.fine_dim = int(p.get("fine_dim", 0) or 0)
        self.fine_mix = float(p.get("fine_mix", 0.0))
        if self.fine_dim:
            fh = int(p.get("fine_hidden", 48))
            self.fine_conv = nn.Sequential(
                nn.Conv1d(self.fine_dim, fh, 3, padding=1, dilation=1), nn.GELU(),
                nn.Conv1d(fh, fh, 3, padding=2, dilation=2), nn.GELU(),
                nn.Conv1d(fh, fh, 3, padding=4, dilation=4), nn.GroupNorm(8, fh), nn.GELU())
            self.fuse = nn.Sequential(nn.Conv1d(2 * p["gru_hidden"] + fh, h, 3, padding=1), nn.GELU())
            self.fine_state = nn.Conv1d(h, 3, 1)

    def trunk(self, x):
        if self.tok_shape:
            B, T, _ = x.shape
            tok = x[..., self.n_step:].reshape(B, T, *self.tok_shape)
            x = torch.cat([x[..., :self.n_step], self.pool(tok)], dim=-1)
        z = self.tcn(self.inp(x))
        g, _ = self.gru(z)
        return g

    def state_logits(self, g):
        if not self.split:
            return self.state(g)
        a, c = self.alive(g).squeeze(-1), self.ctrl(g).squeeze(-1)
        # normalised log-probabilities are valid logits: log_softmax leaves them unchanged
        return torch.stack([F.logsigmoid(a) + F.logsigmoid(c), F.logsigmoid(a) + F.logsigmoid(-c),
                            F.logsigmoid(-a)], dim=-1)

    def forward(self, x):
        g = self.trunk(x)
        return self.state_logits(g), self.bnd(g).squeeze(-1)

    def forward_all(self, x):
        """(state logits, boundary logit, touch logits or None) with one trunk pass."""
        g = self.trunk(x)
        return self.state_logits(g), self.bnd(g).squeeze(-1), (self.touch(g) if self.split else None)

    def forward_fine(self, x, xf):
        """x (B, T, F) 5 Hz, xf (B, T, 5, Ff) frames -> (coarse logits (B, T, 3), boundary (B, T),
        25 fps log-probs (B, 5T, 3): C2F average of coarse (upsampled) and fine)."""
        g = self.trunk(x)
        coarse = self.state_logits(g)
        B, T, C = g.shape
        up = g.unsqueeze(2).expand(B, T, 5, C).reshape(B, 5 * T, C)
        f = self.fine_conv(xf.reshape(B, 5 * T, -1).transpose(1, 2))
        u = self.fuse(torch.cat([up.transpose(1, 2), f], dim=1))
        zf = self.fine_state(u).transpose(1, 2)
        lc = F.log_softmax(coarse, dim=-1)
        lc_up = lc.unsqueeze(2).expand(B, T, 5, 3).reshape(B, 5 * T, 3)
        mix = float(self.fine_mix)
        fine = F.log_softmax(zf, dim=-1) if mix == 0 else F.log_softmax(mix * lc_up + (1 - mix) * F.log_softmax(zf, dim=-1), dim=-1)
        return coarse, self.bnd(g).squeeze(-1), fine

    def touch_logits(self, x):
        """(B, T, 3) touch-level logits [H, A, None]; training-time auxiliary head (split mode)."""
        assert self.split, "touch head exists only with heads='split'"
        return self.touch(self.trunk(x))


def boundary_targets(y, known, sigma):
    """Gaussian-smoothed 'label changes here' targets on the step grid."""
    ch = np.zeros(len(y), np.float32)
    ok = known[1:] & known[:-1]
    ch[1:] = ((y[1:] != y[:-1]) & ok).astype(np.float32)
    if sigma > 0 and ch.any():
        r = int(3 * sigma)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        ch = np.minimum(1.0, np.convolve(ch, k, mode="same")).astype(np.float32)
    return ch


def _short_team_extra(logp, y, team_weight, team_on=None):
    """Extra conditional H/A CE; the dead logit cancels from this term.

    This multiplier never changes source weights, alive/dead CE, smoothing, IoU or boundary BCE.
    Masks and source weights are applied by the caller, with the original denominator.
    """
    conditional = torch.logsumexp(logp[..., :2], dim=-1) - logp.gather(-1, y.clamp(0, 1)[..., None]).squeeze(-1)
    extra = (team_weight - 1.0) * (y < 2).float() * conditional
    return extra if team_on is None else extra * team_on.float()[:, None]


def mask_team_boundary_tmse(d, y, known, p, bnd_on=None):
    """Optional dense-external H/A boundary mask; preserve the original denominator.

    -1 disables the feature without adding tensor operations. 0 exempts exactly the edge
    between two known H/A labels of different teams. D/live and unknown/padding edges
    are unchanged. bnd_on excludes every BAR weak-anchor window, even if two anchors
    happen to be adjacent. This is a training mask, never an inference-time GT input.
    """
    radius = p.get("tmse_team_boundary_radius", -1)
    if not isinstance(radius, (int, np.integer)) or radius < -1:
        raise ValueError("tmse_team_boundary_radius must be an integer >= -1")
    if radius < 0 or d.shape[1] == 0:
        return d
    live = (y == 0) | (y == 1)
    eligible = known[:, 1:] & known[:, :-1] & live[:, 1:] & live[:, :-1]
    changes = eligible & (y[:, 1:] != y[:, :-1])
    if radius:
        changes = F.max_pool1d(changes.float().unsqueeze(1), 2 * radius + 1,
                              stride=1, padding=radius).squeeze(1).bool() & eligible
    if bnd_on is not None:
        changes = changes & bnd_on.bool()[:, None]
    return d * (~changes).to(d.dtype).unsqueeze(-1)


def weak_support_tmse(d, known, p, bnd_on=None):
    """Change only weak-anchor windows, keeping the historical all-edge denominator.

    all: old behavior, no tensor operations. scale: uniformly reduce the weak-window
    penalty by its known-pair fraction. known: keep just those known pairs. The last two
    have equal total gate weight per window, separating support location from strength.
    Dense external pretraining (bnd_on absent) and dense external windows are unchanged.
    """
    mode = p.get("tmse_weak_support", "all")
    if mode not in ("all", "scale", "known"):
        raise ValueError("tmse_weak_support must be all, scale, or known")
    if mode == "all" or bnd_on is None or d.shape[1] == 0:
        return d
    support = (known[:, 1:] & known[:, :-1]).to(d.dtype)
    gate = support if mode == "known" else support.mean(1, keepdim=True).expand_as(support)
    gate = torch.where(bnd_on.bool()[:, None], torch.ones_like(gate), gate)
    return d * gate.unsqueeze(-1)


def loss_fn(logits, bnd_logit, y, known, bnd_t, p, w=None, bnd_on=None, team_on=None, team_weight=None):
    """y (B,T) int64 in {0,1,2} (anything where known=False), known (B,T) bool.
    w (B,): per-window weight (source balancing, Gün 2 BAR adaptation); bnd_on (B,): windows
    whose boundary targets are meaningful (dense true labels; not sparse weak anchors)."""
    if w is not None:
        return _weighted_loss(logits, bnd_logit, y, known, bnd_t, p, w, bnd_on, team_on, team_weight)
    logp = F.log_softmax(logits, dim=-1)
    m = known.float()
    nll = F.nll_loss(logp.transpose(1, 2), y.clamp(0, 2), reduction="none")
    if team_weight is not None:
        nll = nll + _short_team_extra(logp, y, team_weight)
    ce = (nll * m).sum() / m.sum().clamp(min=1)
    d = (logp[:, 1:] - logp[:, :-1].detach()) ** 2
    d = torch.clamp(d, max=p["tmse_tau"] ** 2)
    tmse = mask_team_boundary_tmse(d, y, known, p, bnd_on).mean()
    prob = logp.exp() * m.unsqueeze(-1)
    onehot = F.one_hot(y.clamp(0, 2), 3).float() * m.unsqueeze(-1)
    inter = (prob * onehot).sum(dim=(0, 1))
    union = (prob + onehot - prob * onehot).sum(dim=(0, 1))
    present = onehot.sum(dim=(0, 1)) > 0
    iou = (inter / union.clamp(min=1e-6))[present].mean() if present.any() else torch.zeros((), device=logits.device)
    bce = F.binary_cross_entropy_with_logits(bnd_logit, bnd_t, reduction="none",
                                             pos_weight=torch.tensor(p["bnd_pos_weight"], device=logits.device))
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    total = ce + p["lambda_tmse"] * tmse + p["lambda_iou"] * (1 - iou) + p["lambda_bnd"] * bce
    return total, {"ce": float(ce), "tmse": float(tmse), "iou": float(iou), "bnd": float(bce)}


def _weighted_loss(logits, bnd_logit, y, known, bnd_t, p, w, bnd_on, team_on=None, team_weight=None):
    """w: (B,) per-window or (B, T) per-step weights (class balancing of weak anchors).
    team_on (B,) float lambda in [0, 1]: the window's loss is (1 - lambda) * marginal 2-class
    dead-vs-alive loss (-log P(D) or -log(P(H)+P(A))) + lambda * full 3-class loss; lambda also
    weights the window in the soft-IoU term (lambda = 1: ordinary window)."""
    logp = F.log_softmax(logits, dim=-1)
    if w.dim() == 2:
        w_step = w
        w = (w * known.float()).sum(1) / known.float().sum(1).clamp(min=1)    # window weight
    else:
        w_step = w[:, None]
    m = known.float() * w_step
    nll = F.nll_loss(logp.transpose(1, 2), y.clamp(0, 2), reduction="none")
    if team_on is not None:
        lp_alive = torch.logsumexp(logp[..., :2], dim=-1)
        marg = torch.where(y == 2, -logp[..., 2], -lp_alive)
        lam = team_on.float()[:, None]
        nll = lam * nll + (1 - lam) * marg
    if team_weight is not None:
        nll = nll + _short_team_extra(logp, y, team_weight, team_on)
    ce = (nll * m).sum() / m.sum().clamp(min=1e-6)
    d = torch.clamp((logp[:, 1:] - logp[:, :-1].detach()) ** 2, max=p["tmse_tau"] ** 2)
    d = mask_team_boundary_tmse(d, y, known, p, bnd_on)
    d = weak_support_tmse(d, known, p, bnd_on)
    tmse = (d.mean(dim=(1, 2)) * w).sum() / w.sum().clamp(min=1e-6)
    mi = m if team_on is None else m * team_on.float()[:, None]
    prob = logp.exp() * mi.unsqueeze(-1)
    onehot = F.one_hot(y.clamp(0, 2), 3).float() * mi.unsqueeze(-1)
    inter = (prob * onehot).sum(dim=(0, 1))
    union = (prob + onehot - prob * onehot).sum(dim=(0, 1))
    present = onehot.sum(dim=(0, 1)) > 0
    iou = (inter / union.clamp(min=1e-6))[present].mean() if present.any() else torch.zeros((), device=logits.device)
    mb = known.float() * w_step * bnd_on.float()[:, None]
    bce = F.binary_cross_entropy_with_logits(bnd_logit, bnd_t, reduction="none",
                                             pos_weight=torch.tensor(p["bnd_pos_weight"], device=logits.device))
    bce = (bce * mb).sum() / mb.sum().clamp(min=1e-6)
    total = ce + p["lambda_tmse"] * tmse + p["lambda_iou"] * (1 - iou) + p["lambda_bnd"] * bce
    return total, {"ce": float(ce), "tmse": float(tmse), "iou": float(iou), "bnd": float(bce)}


def fine_gate(yf, radius):
    """(n,) 1 where a known frame label changes within +-radius frames, else 0."""
    y = np.asarray(yf).reshape(-1)
    ch = np.zeros(len(y), bool)
    ok = (y[1:] >= 0) & (y[:-1] >= 0)
    ch[1:] = ok & (y[1:] != y[:-1])
    if radius > 0 and ch.any():
        ch = np.convolve(ch.astype(np.float64), np.ones(2 * radius + 1), mode="same") > 0
    return ch.astype(np.float32)


def _fine_windows(seqs, p, fine):
    """Fine-channel windows with exactly _chunks' window selection: [(xf (L,5,Ff), yf (L*5,),
    gate (L*5,))]. fine[i] = (xf (T,5,Ff), yf (T,5) int, -1 unknown) or None."""
    out = []
    L, step = p["chunk"], p["chunk_step"]
    ff = int(p["fine_dim"])
    for i, item in enumerate(seqs):
        x, y, known = item[:3]
        n = len(x)
        if fine[i] is None:
            xf, yf = np.zeros((n, 5, ff), np.float32), np.full((n, 5), -1, np.int64)
        else:
            xf, yf = fine[i]
        starts = list(range(0, max(1, n - L + 1), step))
        if n > L and starts[-1] != n - L:
            starts.append(n - L)
        for s in starts:
            e = min(n, s + L)
            if known[s:e].mean() < 0.05:
                continue
            pad = L - (e - s)
            xw = np.pad(np.asarray(xf, np.float32)[s:e], [(0, pad), (0, 0), (0, 0)])
            yw = np.pad(np.asarray(yf, np.int64)[s:e], [(0, pad), (0, 0)], constant_values=-1).reshape(-1)
            out.append((xw, yw, fine_gate(yw, int(p.get("fine_gate", 2)))))
    return out


def _chunks(seqs, p, touch=None, team_weights=None):
    """[(x (L,F), y (L,), known (L,), bnd (L,)[, w, bnd_on])] training windows.
    A sequence may carry (x, y, known) or (x, y, known, weight, boundary_on).
    touch: optional per-sequence touch targets (int (T,), -1 unknown) or None; when given the
    function returns (windows, touch_windows) with the same window selection."""
    out, tout = [], []
    if team_weights is not None and len(team_weights) != len(seqs):
        raise ValueError("one team-weight array is required per sequence")
    L, step = p["chunk"], p["chunk_step"]
    for i, item in enumerate(seqs):
        x, y, known = item[:3]
        extra = tuple(item[3:])
        step_w = len(extra) > 0 and np.ndim(extra[0]) == 1
        bnd = boundary_targets(y, known, p["bnd_sigma"])
        n = len(x)
        tw = None if team_weights is None else np.asarray(team_weights[i], np.float32)
        if tw is not None and (tw.shape != (n,) or not np.isfinite(tw).all() or (tw < 1).any()):
            raise ValueError("team weights must be finite multipliers >= 1, aligned with the sequence")
        starts = list(range(0, max(1, n - L + 1), step))
        if n > L and starts[-1] != n - L:
            starts.append(n - L)
        for s in starts:
            e = min(n, s + L)
            if known[s:e].mean() < 0.05:
                continue
            pad = L - (e - s)
            ex = ((np.pad(np.asarray(extra[0], np.float32)[s:e], (0, pad)),) + extra[1:]) if step_w else extra
            out.append(tuple(np.pad(a[s:e], [(0, pad)] + [(0, 0)] * (a.ndim - 1)) for a in
                             (x, y, known.astype(np.float32), bnd)) + ex)
            if tw is not None:
                out[-1] += (np.pad(tw[s:e], (0, pad), constant_values=1.0),)
            if touch is not None:
                t = touch[i] if touch[i] is not None else np.full(n, -1, np.int64)
                tout.append(np.pad(np.asarray(t, np.int64)[s:e], (0, pad), constant_values=-1))
    return (out, tout) if touch is not None else out


def train(seqs, p, seed, log=None, init=None, epochs=None, lr=None, touch=None, fine=None, team_weights=None):
    """seqs: [(x (T,F) float32, y (T,) int, known (T,) bool[, weight, boundary_on])] -> trained
    SeqNet (eval mode). init: a trained SeqNet to continue from (fine-tuning, a copy is made);
    epochs / lr override p for that run."""
    import copy
    set_determinism(seed)
    dev = device()
    use_touch = touch is not None and p.get("heads", "joint") == "split"
    if use_touch:
        chunks, tchunks = _chunks(seqs, p, touch, team_weights)
        TT = torch.tensor(np.stack(tchunks), dtype=torch.int64)
    else:
        chunks = _chunks(seqs, p, team_weights=team_weights)
    Tw = None
    if team_weights is not None:
        Tw = torch.tensor(np.stack([c[-1] for c in chunks]), dtype=torch.float32)
        chunks = [c[:-1] for c in chunks]
    use_fine = fine is not None and int(p.get("fine_dim", 0) or 0) > 0
    if use_fine:
        fw = _fine_windows(seqs, p, fine)
        assert len(fw) == len(chunks)
        XF = torch.tensor(np.stack([w[0] for w in fw]), dtype=torch.float32)
        YF = torch.tensor(np.stack([w[1] for w in fw]), dtype=torch.int64)
        GF = torch.tensor(np.stack([w[2] for w in fw]), dtype=torch.float32)
    X = torch.tensor(np.stack([c[0] for c in chunks]), dtype=torch.float32)
    Y = torch.tensor(np.stack([c[1] for c in chunks]), dtype=torch.int64)
    K = torch.tensor(np.stack([c[2] for c in chunks]) > 0.5)
    Bt = torch.tensor(np.stack([c[3] for c in chunks]), dtype=torch.float32)
    weighted = len(chunks[0]) > 4
    if weighted:
        if any(np.ndim(c[4]) == 1 for c in chunks):
            L = len(chunks[0][0])
            Wt = torch.tensor(np.stack([np.full(L, c[4], np.float32) if np.ndim(c[4]) == 0 else c[4]
                                        for c in chunks]), dtype=torch.float32)
        else:
            Wt = torch.tensor([float(c[4]) for c in chunks], dtype=torch.float32)
        Bon = torch.tensor([bool(c[5]) for c in chunks])
        Ton = torch.tensor([float(c[6]) if len(c) > 6 else 1.0 for c in chunks]) if len(chunks[0]) > 6 else None
    model = copy.deepcopy(init).to(dev) if init is not None else SeqNet(X.shape[-1], p).to(dev)
    n_ep = int(epochs or p["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr or p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(n_ep):
        model.train()
        order = torch.randperm(len(X), generator=gen)
        tot = 0.0
        for i in range(0, len(order), p["batch"]):
            idx = order[i:i + p["batch"]]
            if use_touch:
                logits, bl, tl = model.forward_all(X[idx].to(dev))
            elif use_fine:
                logits, bl, lf = model.forward_fine(X[idx].to(dev), XF[idx].to(dev))
            else:
                logits, bl = model(X[idx].to(dev))
            if weighted:
                loss, parts = loss_fn(logits, bl, Y[idx].to(dev), K[idx].to(dev), Bt[idx].to(dev), p,
                                      Wt[idx].to(dev), Bon[idx].to(dev),
                                      None if Ton is None else Ton[idx].to(dev),
                                      None if Tw is None else Tw[idx].to(dev))
            else:
                loss, parts = loss_fn(logits, bl, Y[idx].to(dev), K[idx].to(dev), Bt[idx].to(dev), p,
                                      team_weight=None if Tw is None else Tw[idx].to(dev))
            if use_touch:
                tt = TT[idx].to(dev)
                tm = (tt >= 0).float()
                tce = F.cross_entropy(tl.transpose(1, 2), tt.clamp(min=0), reduction="none")
                tce = (tce * tm).sum() / tm.sum().clamp(min=1)
                loss = loss + float(p.get("lambda_touch", 0.5)) * tce
                parts = dict(parts, touch=float(tce))
            if use_fine:
                yf = YF[idx].to(dev)
                fm = (yf >= 0).float()
                if weighted:
                    wt = Wt[idx].to(dev)
                    wf = (wt[:, :, None].expand(-1, -1, 5).reshape(wt.shape[0], -1) if wt.dim() == 2
                          else wt[:, None].expand(-1, yf.shape[1]))
                    fm = fm * wf
                fce = (F.nll_loss(lf.transpose(1, 2), yf.clamp(min=0), reduction="none") * fm).sum() / fm.sum().clamp(min=1e-6)
                d = torch.clamp((lf[:, 1:] - lf[:, :-1].detach()) ** 2, max=p["tmse_tau"] ** 2).mean(-1)
                gate = 1.0 - GF[idx].to(dev)[:, 1:]
                ftmse = (d * gate).sum() / gate.sum().clamp(min=1)
                loss = loss + float(p.get("lambda_fine", 1.0)) * (fce + float(p.get("lambda_fine_tmse", 0.3)) * ftmse)
                parts = dict(parts, fine_ce=float(fce), fine_tmse=float(ftmse))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if log is not None:
            log.append({"epoch": ep, "loss": tot / len(X), **parts})
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    """(T,3) probabilities for one full sequence x (T,F)."""
    dev = next(model.parameters()).device
    logits, _ = model(torch.tensor(x[None], dtype=torch.float32, device=dev))
    return torch.softmax(logits[0], dim=-1).double().cpu().numpy()


@torch.no_grad()
def predict_fine(model, x, xf):
    """(5T, 3) 25 fps probabilities for one sequence (fine branch)."""
    dev = next(model.parameters()).device
    _, _, lf = model.forward_fine(torch.tensor(x[None], dtype=torch.float32, device=dev),
                                  torch.tensor(xf[None], dtype=torch.float32, device=dev))
    return lf[0].exp().double().cpu().numpy()


@torch.no_grad()
def predict_boundary(model, x):
    """(T,) boundary-head probability per step: P(a new segment starts at this step). Trained
    with pos_weight on Gaussian-smoothed targets, so it is a score, not a calibrated probability."""
    dev = next(model.parameters()).device
    _, bl = model(torch.tensor(x[None], dtype=torch.float32, device=dev))
    return torch.sigmoid(bl[0]).double().cpu().numpy()
