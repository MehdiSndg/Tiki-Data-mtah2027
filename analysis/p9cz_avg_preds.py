"""P9cz: average the dense video predictions of several models (P9ci's pred_<match>_<half>.npz): Q as the normalised
geometric mean (so the team log ratio is the mean of the members'), the other arrays from the first directory. Only
halves present in every directory are written.

    python analysis/p9cz_avg_preds.py <out dir> <dir> <dir> [...]
"""
import glob
import os
import sys

import numpy as np


def main(out, *dirs):
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(dirs[0], "pred_*.npz"))):
        name = os.path.basename(path)
        if not all(os.path.exists(os.path.join(d, name)) for d in dirs):
            continue
        z = dict(np.load(path))
        lq = np.mean([np.log(np.clip(np.load(os.path.join(d, name))["Q"].astype(np.float64), 1e-6, 1)) for d in dirs], axis=0)
        q = np.exp(lq)
        z["Q"] = (q / q.sum(1, keepdims=True)).astype(np.float32)
        np.savez_compressed(os.path.join(out, name), **z)
        print(name, "members", len(dirs))


if __name__ == "__main__":
    main(*sys.argv[1:])
