"""P9bz: T-DEED team ball action spotting (SoccerNet 2025 baseline, github.com/SoccerNet/sn-teamspotting, GPL-3.0)
fine-tuned on the SoccerTrack v2 virtual-camera frames of P9bx (external data only). Colab GPU.

Same model and losses as the baseline (RegNetY-200MF + GSF, SGP-Mixer encoder-decoder, 100-frame clips at 12.5 fps,
13-way frame classification with event weight 5 and labels dilated +-4 frames, displacement regression, team BCE x2),
started from the published SoccerNet checkpoint (loaded with weights_only=True). Frames: P9bx's 960 x 540 windows, the
even (12.5 fps) frames, resized to 796 x 448 as the checkpoint. Augmentation: the baseline's colour jitter and blur,
plus a horizontal mirror that swaps the team label (2025 winner). Team: 0 left, 1 right (bas and SoccerNet agree).

    python p9bz_tdeed_st_colab.py train <p9bx dir> <out dir> <train matches,...> <test matches,...> [epochs]
        # -> <out>/model.pt and, for each test half, <out>/pred_<match>_<half>.npz (per even frame: 13-way probabilities,
        #    team probability, displacement)
"""

import glob
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("SNTS", "/content/sn-teamspotting")
CKPT = os.environ.get("TDEED_CKPT", "/content/tdeed_ckpt/checkpoint_best.pt")
sys.path.insert(0, REPO)
from model.model import TDEEDModel  # noqa: E402

CLASSES = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
           "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]
L, RADI, H, W = 100, 4, 448, 796
FG, LW = 5.0, (1.0, 1.0, 2.0)


class Half:
    """The even frames of one P9bx half: JPEG bytes by index, labels per index."""

    def __init__(self, d):
        self.name = os.path.basename(d)
        meta = np.load(os.path.join(d, "meta.npz"))
        self.n = len(meta["status"])
        self.shards = []
        loc = {}
        for p in sorted(glob.glob(os.path.join(d, "shard_*.npz")), key=lambda q: int(q.rsplit("_", 1)[1][:-4])):
            z = np.load(p)
            s = len(self.shards)
            self.shards.append((p[:-4] + ".npy", z["offs"]))
            for j, f in enumerate(z["frame"]):
                if f % 2 == 0:
                    loc[int(f) // 2] = (s, j)
        self.m = (self.n + 1) // 2
        self.loc = loc
        self.cls = np.zeros(self.m, np.int64)              # 0 background, 1..12 class
        self.disp = np.zeros(self.m, np.float32)
        self.team = np.full(self.m, -1, np.int64)
        for f, lab, side in zip(meta["a_frame"], meta["a_label"], meta["a_side"]):
            if str(lab) not in CLASSES:
                continue
            i = int(f) // 2
            for j in range(max(0, i - RADI), min(self.m, i + RADI + 1)):
                self.cls[j] = CLASSES.index(str(lab)) + 1
                self.disp[j] = i - j
                self.team[j] = int(side)
        self._mm = {}

    def frame(self, i):
        if i not in self.loc:
            return np.zeros((H, W, 3), np.uint8)
        s, j = self.loc[i]
        path, offs = self.shards[s]
        if path not in self._mm:
            self._mm[path] = np.load(path, mmap_mode="r")
        img = cv2.imdecode(np.asarray(self._mm[path][offs[j]:offs[j + 1]]), cv2.IMREAD_COLOR)
        return cv2.resize(img[:, :, ::-1], (W, H), interpolation=cv2.INTER_AREA)

    def clip(self, start):
        return np.stack([self.frame(i) for i in range(start, start + L)])


class Clips(torch.utils.data.Dataset):
    def __init__(self, halves, n, mirror=True):
        self.halves, self.n, self.mirror = halves, n, mirror

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        h = random.choice(self.halves)
        # half of the clips start near an event so the 12 classes are seen often enough
        ev = np.flatnonzero(h.cls > 0)
        if len(ev) and random.random() < 0.5:
            s = int(np.clip(random.choice(ev) - random.randrange(L), 0, h.m - L))
        else:
            s = random.randrange(0, h.m - L)
        x = h.clip(s)
        cls, disp, team = h.cls[s:s + L].copy(), h.disp[s:s + L].copy(), h.team[s:s + L].copy()
        if self.mirror and random.random() < 0.5:
            x = x[:, :, ::-1]
            team = np.where(team >= 0, 1 - team, -1)
        return {"frame": torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), "cls": torch.from_numpy(cls),
                "disp": torch.from_numpy(disp), "team": torch.from_numpy(team)}


def build(dev):
    args = SimpleNamespace(modality="rgb", temporal_arch="ed_sgp_mixer", radi_displacement=RADI, feature_arch="rny002_gsf",
                           event_team=True, clip_len=L, n_layers=2, sgp_ks=9, sgp_r=4, num_classes=12, crop_dim=None)
    impl = TDEEDModel.Impl(args=args).to(dev)
    impl.update_pred_head([13, 18])
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = impl.load_state_dict(sd, strict=False)
    print("checkpoint loaded; missing", len(missing), "unexpected", len(unexpected), flush=True)
    return impl.to(dev)


def losses(out, b, dev):
    logits = out["im_feat"][..., :13]
    cls, disp, team = b["cls"].to(dev), b["disp"].to(dev), b["team"].to(dev)
    w = torch.tensor([1.0] + [FG] * 12, device=dev)
    lc = F.cross_entropy(logits.reshape(-1, 13).float(), cls.reshape(-1), weight=w)
    ld = F.mse_loss(out["displ_feat"].float(), disp)
    m = team >= 0
    lt = F.binary_cross_entropy_with_logits(out["team_feat"][m].float(), team[m].float()) if m.any() else logits.sum() * 0
    return LW[0] * lc + LW[1] * ld + LW[2] * lt, (lc.item(), ld.item(), lt.item())


def predict(impl, h, dev, step=50):
    """Per even frame: 13-way probabilities, team probability, displacement; clips every `step`, averaged."""
    P, T, D, c = np.zeros((h.m, 13)), np.zeros(h.m), np.zeros(h.m), np.zeros(h.m)
    impl.eval()
    starts = list(range(0, max(h.m - L, 0) + 1, step))
    if starts[-1] != h.m - L:
        starts.append(h.m - L)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for s in starts:
            x = torch.from_numpy(h.clip(s)).permute(0, 3, 1, 2)[None].to(dev).float()
            out, _ = impl(x, inference=True)
            P[s:s + L] += torch.softmax(out["im_feat"][0, :, :13].float(), -1).cpu().numpy()
            T[s:s + L] += torch.sigmoid(out["team_feat"][0].float()).cpu().numpy()
            D[s:s + L] += out["displ_feat"][0].float().cpu().numpy()
            c[s:s + L] += 1
    c = np.maximum(c, 1)
    return P / c[:, None], T / c, D / c


def train(src, out, train_ids, test_ids, epochs=12, per_epoch=5000, bs=4, lr=8e-4, workers=8):
    dev = "cuda"
    os.makedirs(out, exist_ok=True)
    halves = lambda ids: [Half(d) for d in sorted(glob.glob(os.path.join(src, "*_*"))) if os.path.isdir(d)
                          and os.path.basename(d).split("_")[0] in ids and os.path.exists(os.path.join(d, "meta.npz"))]
    tr, te = halves(train_ids), halves(test_ids)
    print("train halves", [h.name for h in tr], "test halves", [h.name for h in te], flush=True)
    impl = build(dev)
    if tr and epochs > 0:
        dl = torch.utils.data.DataLoader(Clips(tr, per_epoch), batch_size=bs, num_workers=workers, shuffle=False, drop_last=True,
                                         persistent_workers=True)
        opt = torch.optim.AdamW(impl.parameters(), lr=lr)
        steps = epochs * len(dl)
        warm = len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
        scaler = torch.amp.GradScaler("cuda")
        for ep in range(epochs):
            impl.train()
            t0, acc = time.time(), np.zeros(3)
            for b in dl:
                with torch.amp.autocast("cuda"):
                    o, _ = impl(b["frame"].to(dev).float(), inference=False)
                    loss, parts = losses(o, b, dev)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                acc += parts
            print("epoch", ep, "loss C/D/T", np.round(acc / len(dl), 4), "s", round(time.time() - t0), flush=True)
            torch.save(impl.state_dict(), os.path.join(out, "model.pt"))
    for h in te:
        P, T, D = predict(impl, h, dev)
        np.savez_compressed(os.path.join(out, "pred_{}.npz".format(h.name)), P=P.astype(np.float32), T=T.astype(np.float32),
                            D=D.astype(np.float32), cls=h.cls, team=h.team)
        print("predicted", h.name, flush=True)


if __name__ == "__main__":
    cmd, src, out, a, b = sys.argv[1:6]
    train(src, out, a.split(","), b.split(",") if b != "-" else [], *(int(x) for x in sys.argv[6:7]))
