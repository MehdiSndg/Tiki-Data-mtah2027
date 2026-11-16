"""P9cu: the dead ball of our match with the video head's dead-ball probability as a third model (inference only; no
label is read). Rebuilds dead ball v8 from its parts (must equal the stored chain's dead ball exactly), then the same
with to_blocks((p + q + v) / 3), and checks the gate registered in notes/BULGULAR.md (12 Nov): dead share within
+-3 points, dead blocks within 0.8-1.3 x, frame agreement >= 93 %. Saves cache/barath_dead_video.npz (dead).

    python analysis/p9cu_barath_video_dead.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import p4_common as P4  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import broadcast_still_dead, features_v2  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402


def nblocks(x):
    return int(np.sum(np.diff(np.r_[0, x.astype(np.int8)]) == 1))


def build(mix=False, mix_gru=False):
    """(v8 rebuilt, v8 with the video probability as a third model, p, q, g). mix / mix_gru: the classifier / the GRU
    trained on the production + realistic-dead copies (P9dw / P9dx; v27)."""
    clf = P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, mix=mix)
    m = from_tactical()
    g = m["half"] > 0
    F = features_v2(m)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(F[g])[:, 1]
    gm = P4.dead_gru(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, nseeds=3, mix=mix_gru)    # as v17-v19c were built
    q = np.where(g, np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0), 0)
    fr = load_frames()
    rest = lambda blocks: P4.close_short_gaps(((blocks | np.isin(fr["period"].values, NOT_PLAYED)) & g) | broadcast_still_dead(
        fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz")), g)
    d0 = rest(to_blocks((p + q) / 2))
    v = np.load(os.environ.get("BARATH_DENSE", os.path.join(ROOT, "cache", "barath_dense.npz")))["q_dead"].astype(np.float64)
    d1 = rest(to_blocks((p + q + np.where(g, v, 0)) / 3))
    return d0, d1, p, q, g


def main():
    d0, d1, p, q, g = build()
    ref = np.load(os.path.join(ROOT, "cache", "v19c_dump.npz"))["dead"]
    print("rebuilt v8 dead ball differs from the stored chain's on", int(np.sum(d0 != ref)), "frames")
    share0, share1, agree, ratio = ref[g].mean(), d1[g].mean(), np.mean(ref[g] == d1[g]), nblocks(d1 & g) / nblocks(ref & g)
    ok = abs(share1 - share0) <= 0.03 and 0.8 <= ratio <= 1.3 and agree >= 0.93
    print("dead share {:.3f} -> {:.3f}; blocks {} -> {} (x {:.2f}); frame agreement {:.3f} -> {}".format(
        share0, share1, nblocks(ref & g), nblocks(d1 & g), ratio, agree, "holds" if ok else "FAILS"))
    np.savez_compressed(os.path.join(ROOT, "cache", "barath_dead_video.npz"), dead=d1, p=p.astype(np.float32), q=q.astype(np.float32))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
