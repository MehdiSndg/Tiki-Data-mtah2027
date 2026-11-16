"""P9dd: a team member trained on real broadcast tracking only: the BigGRU of P9am on the PFF World Cup matches
(mtah.external.from_pff, no degradation, team-swap copies, short spells x2 as v10), three seeds. It never sees DFL or
Metrica, so one model serves every held-out match: its P(home) on the degraded copies the other out-of-fold members
are scored on (P9at's test copies) -> <out>/oof_<match>.npz (pp, labels, g), and on PFF matches held out here
(<out>/pff_<id>.npz) for its own accuracy. The model -> <out>/model.pt (nets, mu, sd). Colab GPU (the training table
does not fit the local 8 GB limit).

    [PFF_TRAIN_N=24] python analysis/p9dd_pff_member_colab.py <out dir> [n PFF test = 6] [seeds = 3]
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.pff import from_pff, pff_ids  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p9am_big_gru import big_net  # noqa: E402


def main(out, ntest=6, nseeds=3):
    os.makedirs(out, exist_ok=True)
    ids = pff_ids()
    tr, te = ids[:-ntest], ids[-ntest:]
    if os.environ.get("PFF_TRAIN_N"):                # local run under the 8 GB limit: every k-th training match
        tr = tr[::max(1, len(tr) // int(os.environ["PFF_TRAIN_N"]))][:int(os.environ["PFF_TRAIN_N"])]
    print("PFF train", len(tr), "test", te, flush=True)
    data = []
    for i in tr:
        m = from_pff(i)
        for mm in (m, swap_teams(m)):
            lab = mm["labels"]
            w = np.isin(lab, (pm.H, pm.A)).astype(np.float32) * np.where(G.short_mask(lab), 2.0, 1.0).astype(np.float32)
            data.append((features(mm).astype(np.float32), (lab == pm.H).astype(np.float32), w, mm["half"] > 0))
        print("features", i, data[-1][0].shape, flush=True)
    n = sum(len(d[0]) for d in data)
    s1, s2 = sum(np.nansum(d[0], 0, dtype=np.float64) for d in data), sum(np.nansum(d[0].astype(np.float64) ** 2, 0) for d in data)
    cnt = sum(np.sum(~np.isnan(d[0]), 0) for d in data)
    mu = s1 / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 0)) + 1e-6
    mod = G.GRUEnsemble([], mu.astype(np.float32), sd.astype(np.float32), "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in data]
    print("rows", n, "training", flush=True)
    nets = [G.train(data, s, "cuda", make_net=big_net) for s in range(3, 3 + nseeds)]
    torch.save({"nets": [x.state_dict() for x in nets], "mu": mod.mu, "sd": mod.sd}, os.path.join(out, "model.pt"))
    del data
    pred = lambda m: np.mean([G._predict_one(x, mod.norm(features(m).astype(np.float32)), m["half"] > 0, "cuda") for x in nets], axis=0)
    for i in te:
        m = from_pff(i)
        p, lab = pred(m), m["labels"]
        live = np.isin(lab, (pm.H, pm.A))
        print("pff", i, "team accuracy on live frames {:.4f}".format(np.mean((p[live] > 0.5) == (lab[live] == pm.H))), flush=True)
        np.savez_compressed(os.path.join(out, "pff_{}.npz".format(i)), pp=p.astype(np.float32), labels=lab, g=m["half"] > 0)
    dfl = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    for raw, k in [(from_dfl(i), i) for i in dfl] + [(from_metrica_sample(gm), "metrica") for gm in (1, 2, 3)]:
        m = degrade(raw, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)          # P9at's test copy
        p, lab = pred(m), m["labels"]
        live = np.isin(lab, (pm.H, pm.A))
        print(m["name"], "team accuracy on live frames {:.4f}".format(np.mean((p[live] > 0.5) == (lab[live] == pm.H))), flush=True)
        np.savez_compressed(os.path.join(out, "oof_{}.npz".format(m["name"].split("+")[0])), pp=p.astype(np.float32), labels=lab, g=m["half"] > 0, name=m["name"])
    print("MEMBER_DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 6, int(a[2]) if len(a) > 2 else 3)
