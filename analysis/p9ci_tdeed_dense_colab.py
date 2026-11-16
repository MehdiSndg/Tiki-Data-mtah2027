"""P9ci: T-DEED with a dense possession head on the SoccerTrack v2 virtual-camera frames of P9bx (external data only).
Colab GPU.

P9ch (simulated, fold A): frame-wise team evidence is worth far more than action-frame evidence (+0.03-0.07 at 85-90 %
block accuracy against +0.02 for actions at recall 0.7 / team accuracy 0.9). So the same network as P9bz (SoccerNet
checkpoint, same clips, same action losses as auxiliary tasks) gets a fourth head on the temporal features: per frame
left team / right team / dead ball, trained with cross-entropy on the ball file's status (loose ball and unknown frames
are ignored). Left / right from the status codes by the majority status at each side's actions (as
mtah.external.soccertrack_side_codes); the mirror augmentation swaps them.

    python p9ci_tdeed_dense_colab.py train <p9bx dir> <out dir> <train matches,...> <test matches,...> [epochs]
        # -> <out>/model.pt and <out>/pred_<match>_<half>.npz: P, T, D as P9bz, Q (per even frame: left, right, dead
        #    probabilities), poss (the labels: 0 left, 1 right, 2 dead, -1 ignored)
"""

import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p9bz_tdeed_st_colab as Z  # noqa: E402

L, POSS_W = Z.L, 2.0


def dense_labels(d, m):
    meta = np.load(os.path.join(d, "meta.npz"))
    st = meta["status"]
    code = {}
    for side in (0, 1):
        c = [int(st[min(int(f), len(st) - 1)]) for f, s in zip(meta["a_frame"], meta["a_side"]) if int(s) == side]
        c = [x for x in c if x in (1, 2)]
        code[side] = max((1, 2), key=c.count) if c else None
    out = np.full(m, -1, np.int64)
    s2 = st[::2][:m]
    out[:len(s2)][s2 == 0] = 2
    if code[0] is not None and code[1] is not None and code[0] != code[1]:
        out[:len(s2)][s2 == code[0]] = 0
        out[:len(s2)][s2 == code[1]] = 1
    return out


class Half(Z.Half):
    def __init__(self, d):
        super().__init__(d)
        self.poss = dense_labels(d, self.m)


class Clips(torch.utils.data.Dataset):
    def __init__(self, halves, n):
        self.halves, self.n = halves, n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        h = random.choice(self.halves)
        ev = np.flatnonzero(h.cls > 0)
        if len(ev) and random.random() < 0.5:
            s = int(np.clip(random.choice(ev) - random.randrange(L), 0, h.m - L))
        else:
            s = random.randrange(0, h.m - L)
        x = h.clip(s)
        cls, disp, team, poss = h.cls[s:s + L].copy(), h.disp[s:s + L].copy(), h.team[s:s + L].copy(), h.poss[s:s + L].copy()
        if random.random() < 0.5:
            x = x[:, :, ::-1]
            team = np.where(team >= 0, 1 - team, -1)
            poss = np.where(poss == 0, 1, np.where(poss == 1, 0, poss))
        return {"frame": torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), "cls": torch.from_numpy(cls),
                "disp": torch.from_numpy(disp), "team": torch.from_numpy(team), "poss": torch.from_numpy(poss)}


class Dense(nn.Module):
    """The T-DEED network with a possession head on the output of its temporal module."""

    def __init__(self, impl):
        super().__init__()
        self.impl = impl
        self.head = nn.Sequential(nn.Linear(impl._feat_dim, impl._feat_dim), nn.ReLU(), nn.Linear(impl._feat_dim, 3))
        self._f = None
        impl._temp_fine.register_forward_hook(lambda mod, i, o: setattr(self, "_f", o))

    def forward(self, x, inference):
        out, _ = self.impl(x, inference=inference)
        out["poss"] = self.head(self._f)
        return out


def predict(net, h, dev, step=50):
    P, T, D, Q, c = np.zeros((h.m, 13)), np.zeros(h.m), np.zeros(h.m), np.zeros((h.m, 3)), np.zeros(h.m)
    net.eval()
    starts = list(range(0, max(h.m - L, 0) + 1, step))
    if starts[-1] != h.m - L:
        starts.append(h.m - L)
    from concurrent.futures import ThreadPoolExecutor
    with torch.no_grad(), torch.amp.autocast("cuda"), ThreadPoolExecutor(10) as pool:
        for s in starts:
            x = np.stack(list(pool.map(h.frame, range(s, s + L))))            # JPEG decoding is the bottleneck
            out = net(torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(dev).float(), True)
            P[s:s + L] += torch.softmax(out["im_feat"][0, :, :13].float(), -1).cpu().numpy()
            T[s:s + L] += torch.sigmoid(out["team_feat"][0].float()).cpu().numpy()
            D[s:s + L] += out["displ_feat"][0].float().cpu().numpy()
            Q[s:s + L] += torch.softmax(out["poss"][0].float(), -1).cpu().numpy()
            c[s:s + L] += 1
    c = np.maximum(c, 1)
    return P / c[:, None], T / c, D / c, Q / c[:, None]


def train(src, out, train_ids, test_ids, epochs=6, per_epoch=5000, bs=4, lr=8e-4, workers=10):
    dev = "cuda"
    os.makedirs(out, exist_ok=True)
    halves = lambda ids: [Half(d) for d in sorted(glob.glob(os.path.join(src, "*_*"))) if os.path.isdir(d)
                          and os.path.basename(d).split("_")[0] in ids and os.path.exists(os.path.join(d, "meta.npz"))]
    tr, te = halves(train_ids), halves(test_ids)
    print("train halves", [h.name for h in tr], "test halves", [h.name for h in te], flush=True)
    for h in tr + te:
        print(h.name, "dense labels left / right / dead / ignored", [int((h.poss == k).sum()) for k in (0, 1, 2, -1)], flush=True)
    net = Dense(Z.build(dev)).to(dev)
    if tr and epochs > 0:
        dl = torch.utils.data.DataLoader(Clips(tr, per_epoch), batch_size=bs, num_workers=workers, drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        steps, warm = epochs * len(dl), len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
        scaler = torch.amp.GradScaler("cuda")
        start = int(os.environ.get("START_EPOCH", 0))          # resume after a lost runtime: RESUME=<model.pt> START_EPOCH=<done>
        if os.environ.get("RESUME"):
            net.load_state_dict(torch.load(os.environ["RESUME"], map_location=dev, weights_only=True))
            sched.last_epoch = start * len(dl) - 1
            sched.step()
            print("resumed from", os.environ["RESUME"], "at epoch", start, flush=True)
        for ep in range(start, epochs):
            net.train()
            t0, acc = time.time(), np.zeros(4)
            for i, b in enumerate(dl):
                with torch.amp.autocast("cuda"):
                    o = net(b["frame"].to(dev).float(), False)
                    loss, parts = Z.losses(o, b, dev)
                    tp = b["poss"].to(dev).reshape(-1)                    # a batch from a match without a possession status has no target
                    lp = F.cross_entropy(o["poss"].reshape(-1, 3).float(), tp, ignore_index=-1) if (tp >= 0).any() else o["poss"].sum() * 0
                    loss = loss + POSS_W * lp
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                acc += (*parts, lp.item())
                if i % 100 == 0:
                    print("  step", i, "/", len(dl), "loss C/D/T/P", np.round(acc / (i + 1), 4), "s", round(time.time() - t0), flush=True)
            print("epoch", ep, "loss C/D/T/P", np.round(acc / len(dl), 4), "s", round(time.time() - t0), flush=True)
            torch.save(net.state_dict(), os.path.join(out, "model.pt"))
    for h in te:
        P, T, D, Q = predict(net, h, dev)
        ok = h.poss >= 0
        live = h.poss[ok] < 2
        print("predicted", h.name, "dense accuracy {:.3f}".format(float(np.mean(Q[ok].argmax(1) == h.poss[ok]))),
              "team accuracy on live frames {:.3f}".format(float(np.mean(Q[ok][live][:, :2].argmax(1) == h.poss[ok][live]))), flush=True)
        np.savez_compressed(os.path.join(out, "pred_{}.npz".format(h.name)), P=P.astype(np.float32), T=T.astype(np.float32),
                            D=D.astype(np.float32), Q=Q.astype(np.float32), cls=h.cls, team=h.team, poss=h.poss)


if __name__ == "__main__":
    cmd, src, out, a, b = sys.argv[1:6]
    train(src, out, a.split(","), b.split(",") if b != "-" else [], *(int(x) for x in sys.argv[6:7]))
