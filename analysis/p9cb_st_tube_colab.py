"""P9cb: a temporal player-touch model on SoccerTrack v2 (external data only). Colab GPU.

Peral et al. (2025) show player crops lift possessor accuracy on a tactical camera (82.1 -> 86.4 -> 89.6 %); the touch
literature is temporal (T-DEED, Sanford's clip models). Here: for each ball action, the K players nearest the ball at the
action frame (pitch distance, GSR), each as a tube of T frames at 25 fps centred on a frame t; crops as for BAR-ATH
(square 1.4 x the YOLO box's larger side, 128 x 128; P9ca boxes). Target "this player touches the ball at t": 1 for the
actor at the action frame (t = f, f +- 1), 0 for the actor at t = f +- 5..12 (timing) and for the other players at f.
Model: ResNet-18 (ImageNet) per crop -> two temporal convolutions -> the centre frame's logit; missing crops are zeros
with a mask channel in the temporal part. Augmentation: mirror, colour jitter, a random down-and-up scale (0.6-1.0, the
BAR-ATH crops' sharpness differs), small shifts.

Gate A (notes/BULGULAR.md 9 Nov), on the test fold's action frames: the team of the highest-scoring candidate against the
action's side; also the actor's top-1. Writes <out>/gateA_<test fold>.txt and per-action scores.

    python p9cb_st_tube_colab.py <p9bx dir> <p9ca dir> <out dir> <train matches,...> <test matches,...> [epochs]
"""

import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

K, T, SIZE, MARGIN = 6, 9, 128, 1.4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Half:
    def __init__(self, vdir, ddir):
        self.name = os.path.basename(vdir)
        meta = np.load(os.path.join(vdir, "meta.npz"))
        self.bx, self.by = meta["bx"], meta["by"]
        self.acts = [(int(f), int(p), int(s)) for f, p, s in zip(meta["a_frame"], meta["a_pid"], meta["a_side"]) if p >= 0]
        R = meta["rows"]
        self.rows_f, self.rows_pid, self.rows_xy = R[:, 0].astype(int), meta["rows_pid"], R[:, 7:9]
        self.shards, self.loc, self._mm = [], {}, {}
        for p in sorted(glob.glob(os.path.join(vdir, "shard_*.npz")), key=lambda q: int(q.rsplit("_", 1)[1][:-4])):
            z = np.load(p)
            s = len(self.shards)
            self.shards.append((p[:-4] + ".npy", z["offs"]))
            for j, f in enumerate(z["frame"]):
                self.loc[int(f)] = (s, j)
        d = np.load(os.path.join(ddir, self.name + ".npz"))
        self.det = {}
        for f, pid, side, b in zip(d["frame"], d["pid"], d["side"], d["box"]):
            self.det[(int(f), int(pid))] = (b, int(side))
        self.side = {int(p): int(s) for p, s in zip(d["pid"], d["side"])}

    def image(self, f):
        s, j = self.loc[f]
        path, offs = self.shards[s]
        if path not in self._mm:
            self._mm[path] = np.load(path, mmap_mode="r")
        return cv2.imdecode(np.asarray(self._mm[path][offs[j]:offs[j + 1]]), cv2.IMREAD_COLOR)

    def nearest(self, f, k=K):
        a, b = np.searchsorted(self.rows_f, f), np.searchsorted(self.rows_f, f, side="right")
        if b <= a or np.isnan(self.bx[f]):
            return []
        dd = np.hypot(self.rows_xy[a:b, 0] - self.bx[f], self.rows_xy[a:b, 1] - self.by[f])
        return [int(self.rows_pid[a + i]) for i in np.argsort(dd)[:k]]

    def tube(self, f, pid, aug=None):
        x = np.zeros((T, SIZE, SIZE, 3), np.float32)
        m = np.zeros(T, np.float32)
        for i, g in enumerate(range(f - T // 2, f + T // 2 + 1)):
            if g not in self.loc or (g, pid) not in self.det:
                continue
            (cx, cy, w, h), _ = self.det[(g, pid)]
            if aug:
                cx, cy = cx + aug["dx"] * w, cy + aug["dy"] * h
            c = crop(self.image(g), cx, cy, w, h)
            if aug:
                c = aug_img(c, aug)
            x[i], m[i] = (c[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD, 1
        return x, m


def crop(img, x, y, w, h):
    s = MARGIN * max(w, h)
    x0, y0 = int(round(x - s / 2)), int(round(y - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    H, W = img.shape[:2]
    pad = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1), 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def aug_img(c, a):
    if a["flip"]:
        c = c[:, ::-1]
    if a["scale"] < 1:
        s = max(int(SIZE * a["scale"]), 16)
        c = cv2.resize(cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    c = np.clip(c.astype(np.float32) * a["gain"] + a["bias"], 0, 255).astype(np.uint8)
    return c


def new_aug():
    return {"flip": random.random() < 0.5, "scale": random.uniform(0.6, 1.0), "gain": random.uniform(0.8, 1.2),
            "bias": random.uniform(-15, 15), "dx": random.uniform(-0.1, 0.1), "dy": random.uniform(-0.1, 0.1)}


class Samples(torch.utils.data.Dataset):
    """(half, frame, player, target) tuples."""

    def __init__(self, halves, train):
        self.halves, self.train, self.items = halves, train, []
        for hi, h in enumerate(halves):
            for f, actor, _ in h.acts:
                cands = h.nearest(f)
                if actor not in cands:
                    continue
                for p in cands:
                    self.items.append((hi, f, p, 1.0 if p == actor else 0.0))
                if train:
                    for o in (-1, 1):
                        self.items.append((hi, f + o, actor, 1.0))
                    for o in random.sample([-12, -10, -8, -6, -5, 5, 6, 8, 10, 12], 3):
                        self.items.append((hi, f + o, actor, 0.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        hi, f, p, y = self.items[i]
        x, m = self.halves[hi].tube(f, p, new_aug() if self.train else None)
        return torch.from_numpy(x).permute(0, 3, 1, 2), torch.from_numpy(m), torch.tensor(y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        r.fc = nn.Identity()
        self.cnn = r
        self.t1 = nn.Conv1d(513, 256, 3, padding=1)
        self.t2 = nn.Conv1d(256, 256, 3, padding=1)
        self.out = nn.Linear(256, 1)

    def forward(self, x, m):
        b, t = x.shape[:2]
        f = self.cnn(x.flatten(0, 1)).view(b, t, -1) * m[..., None]
        h = torch.cat([f, m[..., None]], -1).transpose(1, 2)
        h = F.relu(self.t2(F.relu(self.t1(h))))
        return self.out(h[:, :, t // 2]).squeeze(-1)


def run(vsrc, dsrc, out, train_ids, test_ids, epochs=6, bs=32, lr=3e-4, workers=8):
    dev = "cuda"
    os.makedirs(out, exist_ok=True)
    pick = lambda ids: [Half(d, dsrc) for d in sorted(glob.glob(os.path.join(vsrc, "*_*"))) if os.path.isdir(d)
                        and os.path.basename(d).split("_")[0] in ids and os.path.exists(os.path.join(dsrc, os.path.basename(d) + ".npz"))]
    tr, te = pick(train_ids), pick(test_ids)
    net = Net().to(dev)
    if tr and epochs > 0:
        ds = Samples(tr, True)
        pos = np.array([it[3] for it in ds.items])
        w = np.where(pos > 0, 0.5 / max(pos.sum(), 1), 0.5 / max((pos == 0).sum(), 1))
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=min(len(ds), 60000), replacement=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl))
        scaler = torch.amp.GradScaler("cuda")
        print("train samples", len(ds), "positives", int(pos.sum()), flush=True)
        for ep in range(epochs):
            net.train()
            t0, tot = time.time(), 0.0
            for x, m, y in dl:
                with torch.amp.autocast("cuda"):
                    loss = F.binary_cross_entropy_with_logits(net(x.to(dev), m.to(dev)).float(), y.to(dev).float())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            print("epoch", ep, "loss", round(tot / len(dl), 4), "s", round(time.time() - t0), flush=True)
            torch.save(net.state_dict(), os.path.join(out, "tube.pt"))
    # gate A on the test fold
    net.eval()
    res = []
    for h in te:
        for f, actor, side in h.acts:
            cands = h.nearest(f)
            if not cands:
                continue
            xs, ms = zip(*[h.tube(f, p) for p in cands])
            with torch.no_grad(), torch.amp.autocast("cuda"):
                s = net(torch.from_numpy(np.stack(xs)).permute(0, 1, 4, 2, 3).to(dev), torch.from_numpy(np.stack(ms)).to(dev)).float().cpu().numpy()
            s = np.where(np.stack(ms)[:, T // 2] > 0, s, -np.inf)
            if not np.isfinite(s).any():
                res.append((h.name, f, side, -1, 0, 0))
                continue
            b = int(np.argmax(s))
            res.append((h.name, f, side, h.side.get(cands[b], -1), int(cands[b] == actor), float(s[b])))
    R = np.array([r[2:] for r in res], np.float64)
    acc = float(np.mean(R[:, 1] == R[:, 0])) if len(R) else float("nan")
    cov = float(np.mean(R[:, 1] >= 0)) if len(R) else float("nan")
    top1 = float(np.mean(R[:, 2])) if len(R) else float("nan")
    msg = "test {}: actions {} team accuracy {:.3f} (crop available {:.3f}) actor top-1 {:.3f}".format(",".join(test_ids), len(R), acc, cov, top1)
    print(msg, flush=True)
    open(os.path.join(out, "gateA_{}.txt".format(test_ids[0])), "w").write(msg + "\n")
    np.savez(os.path.join(out, "gateA_{}.npz".format(test_ids[0])), res=np.array([r[1:] for r in res], np.float64), halves=np.array([r[0] for r in res]))


if __name__ == "__main__":
    v, d, o, a, b = sys.argv[1:6]
    run(v, d, o, a.split(","), b.split(","), *(int(x) for x in sys.argv[6:7]))
