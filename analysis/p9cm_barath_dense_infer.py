"""P9cm: the dense video possession model of P9ci (trained on SoccerTrack v2 only) on our match, inference only: the
virtual-camera frames of P9cl (cache/barath_vcam) -> per even frame left / right / dead probabilities, mapped to home /
away per half (left = the team whose players' mean image x is smaller in that half, as SoccerTrack's sides: checked on
its GSR rows, image x against pitch x r = 0.96 and side 0 on the left) and spread to every video frame.

    SNTS=third_party/sn-teamspotting TDEED_CKPT=models/tdeed/checkpoint_best.pt \
        python analysis/p9cm_barath_dense_infer.py <model.pt> [<out npz>]
    # -> cache/barath_dense.npz: e (log q_home - log q_away per video frame, 0 outside the halves), q_dead, q_left
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import p9ci_tdeed_dense_colab as M  # noqa: E402


def main(model, out=os.path.join(ROOT, "cache", "barath_dense.npz")):
    dev = "cuda"
    plan = np.load(os.path.join(ROOT, "cache", "barath_vcam_plan.npz"))
    half, pix, team = plan["half"], plan["pix"].astype(np.float64), plan["slot_team"]
    net = M.Dense(M.Z.build(dev)).to(dev)
    if model != "-":                                               # "-": the SoccerNet checkpoint with an untrained head (smoke test)
        net.load_state_dict(torch.load(model, map_location=dev, weights_only=True))
    n = len(half)
    e, qd, ql = np.zeros(n, np.float32), np.zeros(n, np.float32), np.full(n, 0.5, np.float32)
    for h, hname in ((1, "1st"), (2, "2nd")):
        d = os.path.join(ROOT, "cache", "barath_vcam", "BARATH_{}".format(hname))
        H = M.Half(d)
        if os.environ.get("SMOKE"):
            H.m = 400
        P, T, D, Q = M.predict(net, H, dev)
        np.savez_compressed(os.path.join(os.path.dirname(out), "barath_dense_{}.npz".format(hname)), P=P.astype(np.float32),
                            T=T.astype(np.float32), D=D.astype(np.float32), Q=Q.astype(np.float32))
        a = int(np.load(os.path.join(d, "meta.npz"))["vid0"])
        k = np.flatnonzero(half == h)
        mx = [np.nanmean(pix[k][:, team == t, 0]) for t in (0, 1)]
        home_left = mx[0] < mx[1]
        Qf = np.clip(np.repeat(Q, 2, axis=0), 1e-4, 1)[:len(k)]
        lr = np.log(Qf[:, 0]) - np.log(Qf[:, 1])
        m = len(Qf)
        e[a:a + m] = lr if home_left else -lr
        qd[a:a + m] = Qf[:, 2]
        ql[a:a + m] = Qf[:, 0]
        print(hname, "frames", m, "home on the left" if home_left else "home on the right",
              "argmax shares left / right / dead", np.round(np.bincount(Q.argmax(1), minlength=3) / len(Q), 3), flush=True)
    np.savez_compressed(out, e=e, q_dead=qd, q_left=ql)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
