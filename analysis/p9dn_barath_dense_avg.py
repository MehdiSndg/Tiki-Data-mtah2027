"""P9dn: the mean of several dense video models' outputs on our match (P9cm's barath_dense.npz: e, q_dead, q_left).

    python analysis/p9dn_barath_dense_avg.py A B F A3 F2      # cache/barath_<k>/barath_dense.npz -> cache/barath_dense_avg<n>.npz
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(*keys):
    Z = [np.load(os.path.join(ROOT, "cache", "barath_{}".format(k), "barath_dense.npz")) for k in keys]
    out = {name: np.mean([z[name].astype(np.float64) for z in Z], 0).astype(np.float32) for name in ("e", "q_dead", "q_left")}
    path = os.path.join(ROOT, "cache", "barath_dense_avg{}.npz".format(len(keys)))
    np.savez_compressed(path, **out)
    print("->", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
