"""Step 2b: "this person has the ball" from the person's crop (Peral et al. 2025), trained on the
self-labels of p6c. Colab (GPU). GT-free.

    python analysis/p6d_possessor_crops.py train   # cache/possessor_labels.npz -> models/possessor_r18.pt
    python analysis/p6d_possessor_crops.py infer   # every person box (cache/yolo) -> cache/possessor_prob.npz

Crop: square around the person box, side = 1.4 x its larger side (Peral: 20% margin each way), 128 x 128,
from the tactical video. ResNet-18 (ImageNet weights) fine-tuned as a binary classifier, horizontal
flips and mild colour jitter. Reported on the validation blocks only: AUROC on the self-labels, and
on validation frames that have a positive, how often the most likely box of ALL the frame's person
boxes is the positive one (top-1).
"""

import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

LABELS = os.path.join(ROOT, "cache", "possessor_labels.npz")
YOLO = os.path.join(ROOT, "cache", "yolo")
WEIGHTS = os.path.join(ROOT, "models", "possessor_r18.pt")
PROB = os.path.join(ROOT, "cache", "possessor_prob.npz")
SIZE, MARGIN = 128, 1.4
EPOCHS, BATCH, LR, SEED = 6, 256, 3e-4, 0
MIN_CONF = 0.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(img, x, y, w, h):
    s = max(w, h) * MARGIN
    x0, y0 = int(round(x - s / 2)), int(round(y - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(img[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def crops_for(frames, x, y, w, h):
    """uint8 (n, SIZE, SIZE, 3) RGB crops, one per row, reading the video once in order."""
    order = np.argsort(frames, kind="stable")
    out = np.zeros((len(frames), SIZE, SIZE, 3), np.uint8)
    cap = cv2.VideoCapture(asset_paths()["video"])
    cur, img, i = -1, None, 0
    first = int(frames[order[0]])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    cur = first - 1
    while i < len(order):
        f = int(frames[order[i]])
        while cur < f:
            ok, img = cap.read()
            cur += 1
            if not ok:
                raise RuntimeError("video ended at {}".format(cur))
        while i < len(order) and frames[order[i]] == f:
            j = order[i]
            out[j] = crop(img, x[j], y[j], w[j], h[j])
            i += 1
    return out


def model():
    import torch
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 1)
    return net


def to_tensor(batch, dev):
    import torch
    t = torch.from_numpy(batch).to(dev).float().div_(255)
    t = (t - torch.tensor(MEAN, device=dev)) / torch.tensor(STD, device=dev)
    return t.permute(0, 3, 1, 2).contiguous()


def predict(net, imgs, dev):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(imgs), 1024):
            out.append(torch.sigmoid(net(to_tensor(imgs[a:a + 1024], dev))).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train():
    import torch
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    d = np.load(LABELS)
    t0 = time.time()
    imgs = crops_for(d["frame"], d["x"], d["y"], d["w"], d["h"])
    print("crops", imgs.shape, "{:.0f} s".format(time.time() - t0), flush=True)
    y, val = d["label"].astype(np.float32), d["val"]
    tr = np.flatnonzero(~val)
    dev = "cuda"
    net = model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(tr) // BATCH + 1))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        net.train()
        perm = rng.permutation(tr)
        for a in range(0, len(perm), BATCH):
            j = perm[a:a + BATCH]
            b = imgs[j].copy()
            flip = rng.random(len(j)) < 0.5
            b[flip] = b[flip, :, ::-1]
            x = to_tensor(b, dev)
            x = x * (1 + 0.1 * (torch.rand(len(j), 3, 1, 1, device=dev) - 0.5))
            loss = lossf(net(x).squeeze(1), torch.from_numpy(y[j]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        p = predict(net, imgs[val], dev)
        print("epoch", ep, "loss {:.3f}".format(loss.item()), "val AUROC {:.4f}".format(roc_auc_score(y[val], p)), flush=True)
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS)


def infer():
    import pandas as pd
    import torch
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(YOLO, "chunk_*.npz")))]
    b = {c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "w", "h")}
    k = (b["cls"] == 0) & (b["conf"] >= MIN_CONF)
    b = {c: v[k] for c, v in b.items()}
    dev = "cuda"
    net = model().to(dev)
    net.load_state_dict(torch.load(WEIGHTS, map_location=dev))
    prob = np.zeros(len(b["frame"]), np.float32)
    step = 20000                     # crops held in memory at once (~1 GB)
    t0 = time.time()
    order = np.argsort(b["frame"], kind="stable")
    for a in range(0, len(order), step):
        j = order[a:a + step]
        imgs = crops_for(b["frame"][j], b["x"][j], b["y"][j], b["w"][j], b["h"][j])
        prob[j] = predict(net, imgs, dev)
        print(a + len(j), "/", len(order), "{:.0f} s".format(time.time() - t0), flush=True)
    np.savez_compressed(PROB, frame=b["frame"], x=b["x"], y=b["y"], w=b["w"], h=b["h"], prob=prob)
    # top-1 on validation frames that have a self-labelled positive
    d = np.load(LABELS)
    pos = pd.DataFrame({c: d[c] for c in ("frame", "x", "y")})[(d["label"] == 1) & d["val"]]
    allb = pd.DataFrame({"frame": b["frame"], "x": b["x"], "y": b["y"], "prob": prob})
    top = allb[allb.frame.isin(pos.frame)].sort_values("prob").groupby("frame").tail(1).set_index("frame")
    hit = np.hypot(top.loc[pos.frame, "x"].values - pos.x.values, top.loc[pos.frame, "y"].values - pos.y.values) < 1
    print("val top-1 (most likely box of the frame is the self-labelled possessor): {:.1%} of {} frames".format(hit.mean(), len(hit)))


if __name__ == "__main__":
    {"train": train, "infer": infer}[sys.argv[1]]()
