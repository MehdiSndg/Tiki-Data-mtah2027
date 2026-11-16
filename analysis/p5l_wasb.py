"""Open item 11: WASB and the other heatmap models of its soccer zoo against ball_v1, GT-free.

    uv run --with omegaconf python analysis/p5l_wasb.py quick WASB_SRC WEIGHTS_DIR          # scales A/B/C, WASB, 300 frames
    uv run --with omegaconf python analysis/p5l_wasb.py val   WASB_SRC WEIGHTS_DIR SCALE [model ...]
    uv run --with omegaconf python analysis/p5l_wasb.py kinks WASB_SRC WEIGHTS_DIR SCALE [model ...]
    uv run python analysis/p5l_wasb.py score - - SCALE model ...     # score the candidates `val` saved

WASB_SRC is the src/ folder of github.com/nttcom/WASB-SBDT (MIT; commit 923462c), WEIGHTS_DIR holds
its soccer (ISSIA-CNR) checkpoints <model>_soccer_best.pth.tar (MODEL_ZOO.md). Models: wasb (HRNet),
tracknetv2, restracknetv2, monotrack (3 frames in, 3 out, 512 x 288) and ballseg (2 in, 1 out,
1024 x 576); DeepBall is left out (1 frame, another post-processor; F1 44.5 in the paper).

Our ball is ~8 px in the 1920 px frame, so the frame is cut into tiles that are resized to the
model's input: scale 3.75 = the whole frame (ball ~2 px), 2 = 2 x 2 tiles (~4 px), 1 = native tiles
(~8 px), with >= 32 px overlap. Per tile the sigmoid heatmap of the target frame gives blobs
(connected components above 0.1): confidence = the blob's peak, position = its weighted centre;
blobs of all tiles are merged within 8 px. Strict fp32 (mtah.gpu).
  quick   WASB at the three scales on 300 validation frames
  val     the P3f validation frames (every 5th 10-minute block, public excluded, every 10th frame),
          scored exactly as ball_v1 (analysis/p3f_train_ball.scores), with ball_v1's own boxes
          from the strict-fp32 Blackwell pass (cache/ball_v1_pass_blackwell) alongside
  kinks   two contiguous 2-minute stretches of those blocks: kinks per minute of seen ball
          (mtah.timing.kinks; real tracking 30-47, ball_v1 111), and frames per second
"""

import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import asset_paths  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.gpu import strict_fp32  # noqa: E402
from mtah.timing import kinks  # noqa: E402
from p3a_yolo_ball import on_pitch, on_spot, to_pitch  # noqa: E402
from p3e_ball_pseudolabels import BLOCK, PUBLIC  # noqa: E402

MODELS = ("wasb", "tracknetv2", "restracknetv2", "monotrack", "ballseg")
CFG = {"wasb": "wasb"}                         # model name -> configs/model/<file>.yaml
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
BLOB_MIN, MERGE_PX, OVERLAP = 0.1, 8.0, 32
EVAL_STEP, KINK_LEN = 10, 3000
PASS = os.path.join(ROOT, "cache", "ball_v1_pass_blackwell")
OUT = os.path.join(ROOT, "eda_out", "reports", "P5l_wasb_{}.md")


def load(src, weights, name):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, src)
    from models import build_model
    # HRNet reads its config by attribute (cfg.MODEL.EXTRA), as WASB's hydra configs allow
    cfg = OmegaConf.create({"model": OmegaConf.load(os.path.join(src, "configs", "model", CFG.get(name, name) + ".yaml"))})
    model = build_model(cfg)
    ck = torch.load(os.path.join(weights, "{}_soccer_best.pth.tar".format(name)), map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(sd)
    return model.cuda().eval(), cfg["model"]


def tiles(scale, iw, ih, W=1920, H=1080):
    tw, th = min(W, int(round(iw * scale))), min(H, int(round(ih * scale)))
    def starts(full, t):
        n = 1 if t >= full else int(np.ceil((full - OVERLAP) / (t - OVERLAP)))
        return np.linspace(0, full - t, n).round().astype(int) if n > 1 else np.array([0])
    return [(x, y, tw, th) for y in starts(H, th) for x in starts(W, tw)]


def blobs(hm):
    m = (hm > BLOB_MIN).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        w = hm[ys, xs]
        out.append((float(w.max()), float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())))
    return out


def merge(cands):
    cands = sorted(cands, reverse=True)
    keep = []
    for c, x, y in cands:
        if all(np.hypot(x - kx, y - ky) > MERGE_PX for _, kx, ky in keep):
            keep.append((c, x, y))
    return keep


class Runner:
    def __init__(self, model, cfg, scale):
        self.model, self.fin, self.fout = model, cfg["frames_in"], cfg["frames_out"]
        self.iw, self.ih = cfg["inp_width"], cfg["inp_height"]
        self.ow, self.oh = cfg.get("out_width", self.iw), cfg.get("out_height", self.ih)
        self.tiles = tiles(scale, self.iw, self.ih)
        # which input frames feed the target, and which output channel is the target
        self.offsets = list(range(-(self.fin // 2), self.fin - self.fin // 2)) if self.fin == 3 else list(range(1 - self.fin, 1))
        self.target = self.fin // 2 if self.fout == 3 else 0

    def prep(self, img):
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        out = []
        for x, y, w, h in self.tiles:
            t = cv2.resize(rgb[y:y + h, x:x + w], (self.iw, self.ih), interpolation=cv2.INTER_LINEAR)
            out.append(((t - MEAN) / STD).transpose(2, 0, 1))
        return np.stack(out).astype(np.float32)                 # (tiles, 3, ih, iw)

    def __call__(self, imgs):
        """imgs: the frames_in images around (or ending at) the target -> [(conf, x, y)] in frame pixels."""
        import torch
        x = np.concatenate([self.prep(i) for i in imgs], axis=1)   # (tiles, 3 * frames_in, ih, iw)
        with torch.no_grad():
            hm = torch.sigmoid(self.model(torch.from_numpy(x).cuda())[0][:, self.target]).cpu().numpy()
        cands = []
        for (tx, ty, tw, th), h in zip(self.tiles, hm):
            for c, bx, by in blobs(h):
                cands.append((c, tx + (bx + 0.5) * tw / self.ow - 0.5, ty + (by + 0.5) * th / self.oh - 0.5))
        return merge(cands)


def read(frames, need):
    """{frame: BGR image} for every frame in need, read sequentially."""
    cap = cv2.VideoCapture(asset_paths()["video"])
    want, out = set(int(f) for f in need), {}
    first = min(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for f in range(first, max(want) + 1):
        if not cap.grab():
            break
        if f in want:
            out[f] = cap.retrieve()[1]
    return out


def run(runner, frames):
    rows, t0, n = [], time.time(), 0
    chunk = 400
    for i in range(0, len(frames), chunk):
        part = frames[i:i + chunk]
        imgs = read(part, sorted({f + o for f in part for o in runner.offsets}))
        for f in part:
            for c, x, y in runner([imgs[f + o] for o in runner.offsets]):
                rows.append((f, c, x, y, 8.0, 8.0))
            n += 1
    secs = time.time() - t0
    return pd.DataFrame(rows, columns=["frame", "conf", "x", "y", "w", "h"]), n / secs


def val_frames(m):
    g = np.flatnonzero(m["graded"])
    return g[((g // BLOCK) % 5 == 4) & ((g < PUBLIC[0]) | (g > PUBLIC[1]))]


def pass_boxes(frames):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(PASS, "chunk_*.npz")))]
    b = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "w", "h")})
    return b[(b.cls == 0) & b.frame.isin(set(int(f) for f in frames))].drop(columns="cls")


def score_table(named, frames):
    from p3f_train_ball import scores
    from mtah.frames import load_frames
    m, df = load_tactical(), load_frames()
    v0 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")).label.values
    lines = ["| model | ölçek | kare/s | conf | ATD topu recall | precision | oyun içi boşluk doldurma | nokta yanlışı |",
             "|---|---|---|---|---|---|---|---|"]
    for name, scale, fps, boxes in named:
        s, n_seen, n_gap = scores(boxes, frames, m, df, v0)
        for c, r in s.items():
            lines.append("| {} | {} | {} | {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(
                name, scale, "{:.1f}".format(fps) if fps else "—", c, r["recall"], r["precision"], r["gap_fill"], r["spot"]))
        print(lines[-3:], flush=True)
    lines.append("")
    lines.append("{} kare; ATD topunun görülüp yavaş olduğu kare {}, oyun içi ATD boşluğu (v0) {}.".format(len(frames), n_seen, n_gap))
    return lines


def kink_rate(boxes, frames, hom, conf=0.3):
    b = boxes[boxes.conf >= conf].copy()
    if not len(b):
        return np.nan, 0
    b = b[on_pitch(b, hom) & ~on_spot(b, hom)]
    best = b.sort_values("conf").groupby("frame").tail(1).set_index("frame")
    idx = np.arange(frames[0], frames[-1] + 1)
    bx, by = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    have = best.index.values[(best.index.values >= idx[0]) & (best.index.values <= idx[-1])]
    X, Y = to_pitch(best.loc[have].reset_index(), hom)
    bx[have - idx[0]], by[have - idx[0]] = X / PITCH[0], Y / PITCH[1]
    seen = ~np.isnan(bx)
    k = kinks(pd.Series(bx).interpolate(limit=5).values, pd.Series(by).interpolate(limit=5).values, PITCH)
    k = k[seen[k]]
    return len(k) / (seen.sum() / 25 / 60), seen.mean()


def main():
    cmd, src, weights = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd != "score":
        strict_fp32()
    m = load_tactical()
    val = val_frames(m)
    if cmd == "quick":
        frames = val[::EVAL_STEP][:300]
        model, cfg = load(src, weights, "wasb")
        named = []
        for scale in (3.75, 2.0, 1.0):
            r = Runner(model, cfg, scale)
            boxes, fps = run(r, frames)
            named.append(("wasb ({} karo)".format(len(r.tiles)), scale, fps, boxes))
        named.append(("ball_v1", "—", None, pass_boxes(frames)))
        lines = ["# P5l: WASB ölçek denemesi (300 doğrulama karesi)", ""] + score_table(named, frames)
    elif cmd == "val":
        scale, names = float(sys.argv[4]), sys.argv[5:] or ["wasb"]
        frames = val[::EVAL_STEP]
        named = []
        for name in names:
            model, cfg = load(src, weights, name)
            r = Runner(model, cfg, scale)
            boxes, fps = run(r, frames)
            boxes.to_parquet(os.path.join(ROOT, "eda_out", "reports", "P5l_{}_{}.parquet".format(name, scale)))
            named.append(("{} ({} karo)".format(name, len(r.tiles)), scale, fps, boxes))
        named.append(("ball_v1", "—", None, pass_boxes(frames)))
        lines = ["# P5l: WASB ve model deposu, P3f doğrulama kareleri", ""] + score_table(named, frames)
    elif cmd == "score":
        scale, names = float(sys.argv[4]), sys.argv[5:]
        frames = val[::EVAL_STEP]
        named = [("{}".format(n), scale, None, pd.read_parquet(os.path.join(ROOT, "eda_out", "reports", "P5l_{}_{}.parquet".format(n, scale))))
                 for n in names]
        named.append(("ball_v1", "—", None, pass_boxes(frames)))
        lines = ["# P5l: WASB ve model deposu, P3f doğrulama kareleri (kayıtlı adaylardan)", ""] + score_table(named, frames)
    else:
        scale, names = float(sys.argv[4]), sys.argv[5:] or ["wasb"]
        blocks = sorted(set(val // BLOCK))[:2]
        segs = [np.arange(b * BLOCK, b * BLOCK + KINK_LEN) for b in blocks]
        lines = ["# P5l: titreme (dakikada kırılma, conf ≥ 0.3, sahada, nokta değil)", "",
                 "| model | parça | top görülen kare | dakikada kırılma | kare/s |", "|---|---|---|---|---|"]
        for name in names + ["ball_v1"]:
            for seg in segs:
                if name == "ball_v1":
                    boxes, fps = pass_boxes(seg), None
                else:
                    model, cfg = load(src, weights, name)
                    boxes, fps = run(Runner(model, cfg, scale), seg)
                rate, cover = kink_rate(boxes, seg, m["hom"])
                lines.append("| {} | {}–{} | {:.0%} | {:.1f} | {} |".format(name, seg[0], seg[-1], cover, rate,
                                                                          "{:.1f}".format(fps) if fps else "—"))
                print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.format(cmd), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
