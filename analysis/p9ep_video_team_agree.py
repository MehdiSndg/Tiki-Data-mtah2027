"""P9ep: third judge for recipe C's team members, on our match and label-free. The six members' team probability before
any fusion (production A vs recipe C) against the video's own team evidence (the seven SoccerTrack-trained video
models' mean, cache/barath_dense_avg7.npz "e", log-odds of home, smoothed over 151 frames as the submission uses it):
on graded frames v28 calls live where the video is confident (|smoothed e| > 1), the share where the members' call
(p > 0.5) agrees with the video's sign. No ground truth is read; the video models never saw our match's labels.

    .venv/bin/python analysis/p9ep_video_team_agree.py

Registered before the run (14 Nov): C's agreement must not be lower than A's (and is reported at |e| > 0.5 and > 2 too).
Inference only: a missing model stops the run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.members_c import external_gru_c  # noqa: E402
from p9bw_st_chain_base import ARCHES, DRAWS, no_training  # noqa: E402


def main():
    G.train = no_training
    m = from_tactical()
    m.pop("labels", None)                                # never needed here
    g = m["half"] > 0
    members = {"A": lambda a, mt: G.external_gru(swap=True, metrica=True, short_k=2.0, arch=a, multitask=mt, seeds=tuple(range(6)), draws=DRAWS.get((a, mt), 1)),
               "C": lambda a, mt: external_gru_c(arch=a, multitask=mt, seeds=tuple(range(6)), draws=DRAWS.get((a, mt), 1))}
    p = {k: np.mean([G.predict(f(a, mt), dict(m, labels=np.zeros(len(g), np.int8))) for a in ARCHES for mt in (False, True)], axis=0) for k, f in members.items()}
    e = np.convolve(np.load(os.path.join(ROOT, "cache", "barath_dense_avg7.npz"))["e"].astype(np.float64), np.ones(151) / 151, "same")
    codes = np.load(os.path.join(ROOT, "cache", "v28_hsmm.npz"))["codes"]
    live = g & np.isin(codes, (pm.H, pm.A))
    for thr in (0.5, 1.0, 2.0):
        k = live & (np.abs(e) > thr)
        print("|e| > {}: {} frames;".format(thr, int(k.sum())), " ".join("{} {:.4f}".format(a, np.mean((p[a][k] > 0.5) == (e[k] > 0))) for a in p))
    print("members' calls differ on {:.2%} of live frames".format(np.mean((p["A"][live] > 0.5) != (p["C"][live] > 0.5))))
    k = live & (np.abs(e) > 1.0)
    a, c = (np.mean((p[x][k] > 0.5) == (e[k] > 0)) for x in ("A", "C"))
    print("rule (|e| > 1): C {:.4f} vs A {:.4f} -> {}".format(c, a, "holds" if c >= a else "FAILS"))


if __name__ == "__main__":
    main()
