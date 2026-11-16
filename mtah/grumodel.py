"""The team model as a bidirectional GRU over mtah.teammodel's production features. (P6r, P6s)

    from mtah.grumodel import external_gru, predict
    net = external_gru()                       # trained on degraded DFL, or loaded from models/team_gru_<hash>.pt
    p_home = predict(net, m)                   # (n,) P(home) on frames with half > 0, 0.5 elsewhere

Training data as validated in P6s: the 7 DFL matches degraded with the ball smoothed over SMOOTH frames
(seed 0), no extra copies; with SWAP each also with home and away swapped (mtah.match.swap_teams; P7e:
team call 0.710 -> 0.730 on DFL LOMO, 7 of 7). Features standardised on that data (mean and std kept with
the weights), NaN -> 0, clipped to +-10. Net: linear 128 -> 2-layer BiGRU 128 -> logit; BCE on alive frames;
30 s windows (stride 15 s), EPOCHS epochs, Adam 1e-3, batch 32; SEEDS nets averaged. Inference in 60 s
chunks with 15 s of context each side. The weights live in models/ (gitignored; published with the
submission), keyed by a hash of the code they depend on.
"""

import hashlib
from functools import partial
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
FPS = 25
WIN, STRIDE, CHUNK = 30 * FPS, 15 * FPS, 60 * FPS
EPOCHS, BATCH, LR = 8, 32, 1e-3
SEEDS = (0, 1, 2)
SWITCH = 8.0                 # P6s: the mean of the 3 seeds, best fixed switch cost on DFL LOMO (team call and full chain)
# P6x: a boundary GRU (target: a change of team within +-BOUNDARY_R frames) sets the switch cost per frame,
# SWITCH_HI * (1 - b) + SWITCH_LO * b: 0.704 -> 0.711 on DFL LOMO, 7 of 7 matches; (12, 0.5) chosen on 6 of 7
BOUNDARY_R, SWITCH_HI, SWITCH_LO = 10, 12.0, 0.5
SWAP = True                  # P7e


def _net(nf, hidden=128, layers=2):
    """The production team GRU (128 hidden, 2 layers); "big" (v17, P9am / P9at) is 256 hidden, 3 layers."""
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, hidden)
            self.gru = torch.nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(2 * hidden, 1)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h).squeeze(-1)

    return Net()


def _tcn(nf, hidden=128, dilations=(1, 2, 4, 8, 16, 32), k=5):
    """A non-causal dilated temporal CNN with the GRU net's input and output (P7u; ~10 s of context a side)."""
    import torch

    class Block(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = torch.nn.Conv1d(hidden, hidden, k, padding=d * (k - 1) // 2, dilation=d)
            self.c2 = torch.nn.Conv1d(hidden, hidden, 1)
            self.drop = torch.nn.Dropout(0.1)

        def forward(self, x):
            return x + self.c2(self.drop(torch.relu(self.c1(x))))

    class TCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, hidden)
            self.blocks = torch.nn.Sequential(*[Block(d) for d in dilations])
            self.out = torch.nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.blocks(torch.relu(self.inp(x)).transpose(1, 2)).transpose(1, 2)
            return self.out(torch.relu(h)).squeeze(-1)

    return TCN()


def _set(nf):
    """The player-set GRU (mtah.setgru, P8b): the team features then each team's players nearest the ball."""
    from mtah.setgru import K, NFP, set_net
    return set_net(nf - 2 * K * NFP)


NETS = {"gru": _net, "tcn": _tcn, "set": _set,
        "big": partial(_net, hidden=256, layers=3),                              # v17: P9am / P9at
        "tcn128": partial(_tcn, dilations=(1, 2, 4, 8, 16, 32, 64, 128))}       # P9ao: about +-20 s of context


def model_features(arch, m):
    """The input of a member: the production team features, plus mtah.setgru.player_set for "set"."""
    from mtah.teammodel import features
    F = features(m)
    if arch != "set":
        return F
    from mtah.setgru import player_set
    return np.concatenate([F.astype(np.float32), player_set(m)], axis=1)


def short_mask(labels, short_s=5.0, pad=12):
    """Frames inside a true possession shorter than short_s seconds, and within pad frames of it (P7s)."""
    import possession_metrics as pm
    out = np.zeros(len(labels), bool)
    for a, b, _ in pm.possession_spells(labels, pm.valid_runs(labels)):
        if (b - a) / FPS < short_s:
            out[max(0, a - pad):b + pad] = True
    return out


def _windows(F, y, w, g, offset=0, tail=False):
    """WIN-frame training windows every STRIDE frames over each graded run, the first one `offset` frames in; tail: plus
    one window ending at the run's end, which the fixed grid otherwise never trains on (research report, team audit)."""
    out = []
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        a0 = min(a + offset, max(a, b - WIN))
        last = None
        for s in range(a0, max(a0 + 1, b - WIN + 1), STRIDE):
            z = min(b, s + WIN)
            out.append((F[s:z], y[s:z], w[s:z]))
            last = z
        if tail and last is not None and last < b and b - a >= WIN:
            out.append((F[b - WIN:b], y[b - WIN:b], w[b - WIN:b]))
    return out


def train(data, seed, dev, make_net=None, jitter=False, tail=False):
    """data: list of (F standardised, y float32, weight float32, graded bool). Returns a trained net;
    make_net(nf) builds another net of the same input / output shape (mtah.setgru.set_net); jitter: every epoch the
    window grid starts at a random offset and each run's tail gets a window (P9o); tail: only the tail windows, no
    offset (P9aq)."""
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    wins = [w for d in data for w in _windows(*d, tail=tail)]
    nf = data[0][0].shape[1]
    net = (make_net or _net)(nf).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        if jitter:
            wins = [w for d in data for w in _windows(*d, offset=int(rng.integers(STRIDE)), tail=True)]
        order = rng.permutation(len(wins))
        for a in range(0, len(order), BATCH):
            batch = [wins[i] for i in order[a:a + BATCH]]
            L = max(len(b[0]) for b in batch)
            X = np.zeros((len(batch), L, nf), np.float32)
            Y = np.zeros((len(batch), L) + batch[0][1].shape[1:], np.float32)     # (L,) or (L, heads)
            Wt = np.zeros((len(batch), L) + batch[0][1].shape[1:], np.float32)
            for i, (f, y, w) in enumerate(batch):
                X[i, :len(f)], Y[i, :len(f)], Wt[i, :len(f)] = f, y, w
            X, Y, Wt = (torch.from_numpy(v).to(dev) for v in (X, Y, Wt))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(X), Y, reduction="none") * Wt).sum() / Wt.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def widen(make, nf, heads=2):
    """make(nf)'s net with `heads` outputs per frame (P8m/P8n: team and boundary from one net)."""
    import torch
    net = make(nf)
    net.out = torch.nn.Linear(net.out.in_features, heads)
    return net


def _predict_one(net, F, g, dev):
    """(n,) probabilities, or (n, heads) for a widened net; 0.5 off the graded frames."""
    import torch
    p = None
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    with torch.no_grad():
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            for s in range(a, b, CHUNK):
                lo, hi = max(a, s - STRIDE), min(b, s + CHUNK + STRIDE)
                q = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(F[lo:hi][None])).to(dev))).cpu().numpy()[0]
                z = min(b, s + CHUNK)
                if p is None:
                    p = np.full((len(F),) + q.shape[1:], 0.5)
                p[s:z] = q[s - lo:z - lo]
    return np.full(len(F), 0.5) if p is None else p


class GRUEnsemble:
    def __init__(self, nets, mu, sd, dev, arch="gru", heads=1):
        self.nets, self.mu, self.sd, self.dev, self.arch, self.heads = nets, mu, sd, dev, arch, heads

    def norm(self, F):
        return np.nan_to_num((F - self.mu) / self.sd).astype(np.float32).clip(-10, 10)


def predict(model, m):
    g = m["half"] > 0
    F = model.norm(model_features(getattr(model, "arch", "gru"), m))
    p = np.mean([_predict_one(n, F, g, model.dev) for n in model.nets], axis=0)
    return p[:, 0] if getattr(model, "heads", 1) > 1 else p      # a two-head net: its team head


HASH_FILES = ("mtah/grumodel.py", "mtah/teammodel.py", "mtah/degrade.py", "mtah/external.py", "mtah/timing.py", "mtah/match.py",
              "mtah/setgru.py", "mtah/deadball.py", "analysis/m1_dead_classifier.py", "evaluation/possession_metrics.py")


def _hash_files():
    return [os.path.join(ROOT, f) for f in HASH_FILES]


def _hash(extra=""):
    """Model file key: every source the cached nets' features, labels and training depend on, plus extra (e.g. the
    degradation rates a caller trains with), so a code change never loads stale weights."""
    h = hashlib.sha256()
    for f in _hash_files():
        h.update(open(f, "rb").read())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def boundary_target(labels, r=None):
    """1 within r frames (BOUNDARY_R) of a direct change of team."""
    import possession_metrics as pm
    r = BOUNDARY_R if r is None else r
    t = np.zeros(len(labels), np.float32)
    ch = np.flatnonzero(np.isin(labels[1:], (pm.H, pm.A)) & np.isin(labels[:-1], (pm.H, pm.A)) & (labels[1:] != labels[:-1])) + 1
    for c in ch:
        t[max(0, c - r):c + r + 1] = 1
    return t


def switch_costs(model_boundary, m):
    """Per-frame switch cost from the boundary GRU's probability."""
    b = predict(model_boundary, m)
    return SWITCH_HI * (1 - b) + SWITCH_LO * b


def model_tag(swap=SWAP, metrica=False, short_k=1.0, arch="gru", pretrain=None, multitask=False, mixed=False, seeds=SEEDS, tail=False, draws=1):
    """The part of a model file name that tells its training recipe apart."""
    return ("_swap" if swap else "") + ("_met" if metrica else "") + ("_k{:g}".format(short_k) if short_k != 1.0 else "") + ("" if arch == "gru" else "_" + arch) \
        + ("_" + pretrain if pretrain else "") + ("_mt" if multitask else "") + ("_mix" if mixed else "") + ("" if tuple(seeds) == tuple(SEEDS) else "_s" + "".join(map(str, seeds))) + ("_tail" if tail else "") + ("" if draws == 1 else "_d{}".format(draws))


def feature_stats(Fs):
    """Per-column mean and std (+1e-6) over the rows of every array in Fs, ignoring NaN, from running float64 sums: no
    concatenated copy of the training features (WSL ran out of memory on 5 Nov). np.nanmean / np.nanstd up to rounding."""
    nf = Fs[0].shape[1]
    n, s1, s2 = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    for F in Fs:
        ok = ~np.isnan(F)
        n += ok.sum(0)
        s1 += np.where(ok, F, 0).sum(0, dtype=np.float64)
        s2 += np.where(ok, F.astype(np.float64) ** 2, 0).sum(0)
    mu = s1 / n
    return mu, np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6


def metrica_training_matches(draws=1):
    """Metrica games 1-3, degraded with seeds 0 .. draws-1 (P7i; draws: P9aw)."""
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_metrica_sample
    from mtah.teammodel import SMOOTH
    return [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=SMOOTH), seed=d) for g in (1, 2, 3) for d in range(draws)]


def external_gru(dev=None, target="team", swap=SWAP, metrica=False, short_k=1.0, arch="gru", pretrain=None, multitask=False, mixed=False, seeds=SEEDS,
                 tail=False, draws=1):
    """metrica: Metrica games 1-3 added to the training data (P7i); short_k: the loss weight of frames in short
    true possessions (short_mask, P7s); arch: "gru", "tcn" (P7u) or "set" (P8b); pretrain="st": each net first
    trained on the usable SoccerTrack v2 matches (same recipe, same standardisation), then fine-tuned from those
    weights (P8d, P8j); multitask: the net also predicts the boundary target from the same representation
    (a second head on alive frames, unweighted; P8m, P8n), predict() gives the team head; mixed: every training match
    also once with the harsh degradation (mtah.degrade.harsh_rates, seed 1; P9e); seeds: the nets averaged (P9ad: 6
    instead of 3, team call DFL +0.006 6 / 7, Metrica +0.002); draws: every training match degraded with seeds 0 .. draws-1
    (P9aw: 3 draws for the BigGRU, team call DFL +0.016 6 / 7, Metrica +0.016). The defaults are v7-v9's model."""
    import torch
    from mtah.teammodel import SMOOTH, external_training_matches, features
    import possession_metrics as pm
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = model_tag(swap, metrica, short_k, arch, pretrain, multitask, mixed, seeds, tail, draws)
    make = (lambda nf: widen(NETS[arch], nf)) if multitask else NETS[arch]
    heads = 2 if multitask else 1
    path = os.path.join(ROOT, "models", "{}_gru{}_{}.pt".format(target, tag, _hash()))
    if os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        nets = []
        for sd_ in st["nets"]:
            n = make(len(st["mu"])).to(dev)
            n.load_state_dict(sd_)
            nets.append(n.eval())
        return GRUEnsemble(nets, st["mu"], st["sd"], dev, arch, heads)
    ms = external_training_matches(SMOOTH, tuple((SMOOTH, d) for d in range(1, draws)))
    if metrica:
        ms = ms + metrica_training_matches(draws)
    if mixed:
        import glob
        from mtah.degrade import degrade, harsh_rates
        from mtah.external import DFL, from_dfl, from_metrica_sample
        ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
        ms = ms + [degrade(from_dfl(i), harsh_rates(i), seed=1) for i in ids]
        if metrica:
            ms = ms + [degrade(from_metrica_sample(g), harsh_rates("metrica"), seed=1) for g in (1, 2, 3)]
    if swap:
        from mtah.match import swap_teams
        ms = ms + [swap_teams(m) for m in ms]
    Fs = [model_features(arch, m) for m in ms]
    mu, sd = feature_stats(Fs)
    model = GRUEnsemble([], mu, sd, dev, arch, heads)
    y1 = (lambda m: (m["labels"] == pm.H).astype(np.float32)) if target == "team" else (lambda m: boundary_target(m["labels"]))
    alive = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    w1 = lambda m: alive(m) * (np.where(short_mask(m["labels"]), short_k, 1.0).astype(np.float32) if short_k != 1.0 else 1.0)
    if multitask:
        y = lambda m: np.stack([y1(m), boundary_target(m["labels"])], 1)
        w = lambda m: np.stack([w1(m), alive(m)], 1).astype(np.float32)
    else:
        y, w = y1, w1
    data = [(model.norm(F), y(m), w(m), m["half"] > 0) for F, m in zip(Fs, ms)]
    del Fs
    if pretrain == "st":
        import copy
        from mtah.degrade import degrade, rates_for
        from mtah.external import from_soccertrack, soccertrack_ids, soccertrack_usable
        from mtah.match import swap_teams
        st = [degrade(m, dict(rates_for("st"), smooth_frames=SMOOTH), seed=0) for m in (from_soccertrack(i) for i in soccertrack_ids()) if soccertrack_usable(m)]
        st = st + [swap_teams(m) for m in st] if swap else st
        pre = [(model.norm(model_features(arch, m)), y(m), w(m), m["half"] > 0) for m in st]
        model.nets = []
        for s in seeds:
            warm = train(pre, s, dev, make_net=make).train()
            model.nets.append(train(data, s, dev, make_net=lambda nf, wn=warm: copy.deepcopy(wn), tail=tail))
    else:
        model.nets = [train(data, s, dev, make_net=make, tail=tail) for s in seeds]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"nets": [n.state_dict() for n in model.nets], "mu": mu, "sd": sd}, path)
    return model
