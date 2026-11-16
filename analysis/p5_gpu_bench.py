"""Same 200 frames, same detector, different GPUs: how fast, and are the boxes the same?

    uv run python analysis/p5_gpu_bench.py frames DIR                  # the 200 frames as PNG (+ manifest)
    uv run python analysis/p5_gpu_bench.py run DIR WEIGHTS OUT.npz [--batch 4] [--classes 0]
    uv run python analysis/p5_gpu_bench.py compare A.npz B.npz [...]   # pairwise against the first

The frames are the ones analysis/p3d_determinism.py uses (100 per half, evenly spaced), written as
lossless PNG so every machine reads exactly the same pixels (no video decoder in the comparison).
`run` is strict fp32: TF32 off for matmul and cuDNN, cuDNN deterministic, no autotuning, a fixed
batch, one warm-up batch excluded from the timing. It stores every box with the run's environment
(GPU, torch, CUDA, cuDNN, ultralytics, Python). `compare` reuses p3d's matching: box count change,
unmatched boxes, largest centre and confidence change, and the share of frames whose ball set
(conf >= 0.3) is identical; ball class is taken from the file. `compare --quantized` puts every
box on mtah.gpu.quantize_boxes' grid first and counts exactly equal frames and values, the check
for the submission chain (p3_yolo_pass.py writes its boxes on that grid).
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

IMGSZ, CONF = 1920, 0.10


def cmd_frames(out):
    from p3d_determinism import frames
    os.makedirs(out, exist_ok=True)
    fs, imgs = frames()
    manifest = []
    for f, img in zip(fs, imgs):
        path = os.path.join(out, "f{:06d}.png".format(f))
        cv2.imwrite(path, img)
        manifest.append({"frame": int(f), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=0)
    print(len(manifest), "frames ->", out)


def env_info():
    import torch
    import ultralytics
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "ultralytics": ultralytics.__version__,
            "python": platform.python_version(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def cmd_run(src, weights, out, batch, classes, device="0"):
    from mtah.gpu import strict_fp32
    strict_fp32()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    paths = sorted(glob.glob(os.path.join(src, "f*.png")))
    assert len(paths) == len(manifest)
    for p, m in zip(paths, manifest):
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == m["sha256"], p
    imgs = [cv2.imread(p) for p in paths]
    model = YOLO(weights)
    kw = dict(imgsz=IMGSZ, conf=CONF, half=False, device=device, verbose=False)
    if classes is not None:
        kw["classes"] = classes
    model.predict(imgs[:batch], **kw)                       # warm-up, not timed
    torch.cuda.synchronize()
    boxes, t0 = [], time.time()
    for i in range(0, len(imgs), batch):
        for r in model.predict(imgs[i:i + batch], **kw):
            boxes.append(np.c_[r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xywh.cpu().numpy()])
    torch.cuda.synchronize()
    secs = time.time() - t0
    info = dict(env_info(), batch=batch, weights=os.path.basename(weights), frames=len(imgs),
                seconds=secs, fps=len(imgs) / secs)
    if device == "cpu":
        info["gpu"] = "cpu"
    np.savez_compressed(out, info=json.dumps(info), frame=np.array([m["frame"] for m in manifest]),
                        **{"b{}".format(k): b for k, b in enumerate(boxes)})
    print(json.dumps(info))


def load(path):
    d = np.load(path)
    info = json.loads(str(d["info"]))
    return info, [d["b{}".format(k)] for k in range(len(d["frame"]))]


def cmd_compare_quantized(paths):
    from mtah.gpu import quantize_boxes

    def q(boxes):
        out = []
        for b in boxes:
            conf, xywh = quantize_boxes(b[:, 1], b[:, 2:6])
            out.append(np.c_[b[:, 0], conf, xywh])
        return out
    _, ref = load(paths[0])
    ref = q(ref)
    print("| run | GPU | torch | kare aynı | değer aynı |")
    print("|---|---|---|---|---|")
    for p in paths:
        info, other = load(p)
        other = q(other)
        same = sum(r.shape == o.shape and np.array_equal(r, o) for r, o in zip(ref, other))
        vals = sum(int((r == o).sum()) if r.shape == o.shape else 0 for r, o in zip(ref, other))
        total = sum(r.size for r in ref)
        print("| {} | {} | {} | {}/{} | {}/{} |".format(os.path.basename(p), info["gpu"], info["torch"], same, len(ref), vals, total))


def cmd_compare(paths, ball_cls):
    from p3d_determinism import iou
    ref_info, ref = load(paths[0])
    print("| run | GPU | torch | batch | kare/s | kutu sayısı farkı | eşleşmeyen | en büyük merkez farkı px | en büyük güven farkı | top kümesi aynı |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        info, other = load(p)
        dcount = unmatched = 0
        dpos = dconf = 0.0
        same = 0
        for r, o in zip(ref, other):
            dcount += abs(len(r) - len(o))
            for box in r:
                cand = o[o[:, 0] == box[0]]
                if not len(cand) or iou(box, cand).max() < 0.5:
                    unmatched += 1
                    continue
                j = np.argmax(iou(box, cand))
                dpos = max(dpos, float(np.hypot(*(box[2:4] - cand[j, 2:4]))))
                dconf = max(dconf, float(abs(box[1] - cand[j, 1])))
            rb = {tuple(np.round(b[2:4]).astype(int)) for b in r if b[0] == ball_cls and b[1] >= 0.3}
            ob = {tuple(np.round(b[2:4]).astype(int)) for b in o if b[0] == ball_cls and b[1] >= 0.3}
            same += rb == ob
        print("| {} | {} | {} | {} | {:.1f} | {} | {} | {:.3f} | {:.4f} | {:.0%} |".format(
            os.path.basename(p), info["gpu"], info["torch"], info["batch"], info["fps"], dcount, unmatched, dpos, dconf, same / len(ref)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("frames")
    a.add_argument("out")
    b = sub.add_parser("run")
    b.add_argument("src")
    b.add_argument("weights")
    b.add_argument("out")
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--device", default="0", help="0 for the first GPU, cpu for the CPU")
    b.add_argument("--classes", type=int, nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    c.add_argument("--ball-cls", type=int, default=0)
    c.add_argument("--quantized", action="store_true")
    args = ap.parse_args()
    if args.cmd == "frames":
        cmd_frames(args.out)
    elif args.cmd == "run":
        cmd_run(args.src, args.weights, args.out, args.batch, args.classes, args.device)
    else:
        cmd_compare_quantized(args.paths) if args.quantized else cmd_compare(args.paths, args.ball_cls)


if __name__ == "__main__":
    main()
