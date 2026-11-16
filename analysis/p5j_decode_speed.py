"""Open item 15: is reading the video the bottleneck of the detector pass?

    uv run python analysis/p5j_decode_speed.py [N]      # N frames from the start of each half (default 2000)

Times cv2's sequential read of the tactical video alone (one thread, as analysis/p3_yolo_pass.py
reads it), from the first graded frame of each half. Compare with the model's own speed on frames
already in memory (analysis/p5_gpu_bench.py run) and with the pass's printed fps.
"""

import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cap = cv2.VideoCapture(asset_paths()["video"])
    print("cv2", cv2.__version__, "threads", cv2.getNumThreads())
    for half, first, _ in half_video_blocks(GAME):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))
        t0 = time.time()
        for _ in range(n):
            ok, _img = cap.read()
            assert ok
        secs = time.time() - t0
        print("half {}: {} frames in {:.1f} s = {:.0f} fps".format(half, n, secs, n / secs))


if __name__ == "__main__":
    main()
