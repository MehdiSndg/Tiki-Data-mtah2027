"""P9dl: PFF as pre-training for the BigGRU member (P9dd's PFF-only member failed on the label definition; here the
definition comes from DFL + Metrica): each net starts from P9dd's PFF-trained weights (cache/oof_pff/model.pt, same
standardisation) and is fine-tuned with the member's usual recipe on the nine other matches (P9at's data, team-swap
copies, short spells x2). Against the same recipe from scratch, same seeds (3, 4, 5). First on three DFL folds.
Score: the member swapped into v19c's six-member mean, v25's chain (semi-Markov decoder, our two dead models), 1.6 s /
3 s; and the member's own frame team accuracy. (P9bf: SoccerTrack pre-training of this member was null.) Local GPU.

    python analysis/p9dl_pff_pretrain.py [fold matches = J03WMX,J03WN1,J03WOH]
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
from mtah import semimarkov  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import features  # noqa: E402
from p9am_big_gru import big_net  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

SEEDS = (3, 4, 5)
C = lambda *p: os.path.join(ROOT, "cache", *p)


def main(folds="J03WMX,J03WN1,J03WOH"):
    pre = torch.load(C("oof_pff", "model.pt"), map_location="cuda", weights_only=False)          # our own file (P9dd, this machine); mu / sd are numpy arrays
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    To, Tso = [features(m).astype(np.float32) for m in old], [features(swap_teams(m)).astype(np.float32) for m in old]
    Sh = [G.short_mask(m["labels"]) for m in old]
    labs = [(m["labels"], swap_teams(m)["labels"], m["half"] > 0) for m in old]
    print("features done", flush=True)
    R = []
    for t, (raw, k) in enumerate(raws):
        if not any(f in raw["name"] for f in folds.split(",")):
            continue
        m = degrade(raw, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, Tt, name = m["half"] > 0, features(m).astype(np.float32), m["name"].split("+")[0]
        ks = [j for j in range(len(old)) if j != t]
        out = {}
        for arm in ("scratch", "pff"):
            if arm == "pff":
                mod = G.GRUEnsemble([], pre["mu"], pre["sd"], "cuda")
            else:
                allF = np.concatenate([F for j in ks for F in (To[j], Tso[j])])
                mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
                del allF
            data = []
            for j in ks:
                for F, lab in ((To[j], labs[j][0]), (Tso[j], labs[j][1])):
                    w = np.isin(lab, (pm.H, pm.A)).astype(np.float32) * np.where(G.short_mask(lab), 2.0, 1.0).astype(np.float32)
                    data.append((mod.norm(F), (lab == pm.H).astype(np.float32), w, labs[j][2]))
            nets = []
            for i, s in enumerate(SEEDS):
                def make(nf, i=i):
                    n = big_net(nf)
                    if arm == "pff":
                        n.load_state_dict(pre["nets"][i])
                    return n
                nets.append(G.train(data, s, "cuda", make_net=make))
            out[arm] = np.mean([G._predict_one(n, mod.norm(Tt), g, "cuda") for n in nets], axis=0)
            del data, nets
            torch.cuda.empty_cache()
        f = "oof_{}.npz".format(name)
        z, big, t1, d3 = (dict(np.load(C(d, f), allow_pickle=True)) for d in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
        lab = z["labels"]
        assert len(lab) == len(g)
        l3, b, dead = merged(lab, 3.0), np.nan_to_num(z["b"]), (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"])) / 2
        live = g & np.isin(lab, (pm.H, pm.A))
        res = {}
        for arm, pg in out.items():
            p = np.nan_to_num((pg + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
            c = np.where(g, semimarkov.decode_halves(p, dead, g, b), pm.X).astype(np.int8)
            res[arm] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)] + [np.mean((pg[live] > 0.5) == (lab[live] == pm.H))]
        np.savez_compressed(C("oof_pffpre_{}.npz".format(name)), scratch=out["scratch"].astype(np.float32), pff=out["pff"].astype(np.float32))
        R.append(np.array(res["pff"]) - np.array(res["scratch"]))
        print(name, "scratch {:.4f} {:.4f} acc {:.4f} | PFF pre-trained {:.4f} {:.4f} acc {:.4f} | gain {:+.4f} {:+.4f} acc {:+.4f}".format(*res["scratch"], *res["pff"], *R[-1]), flush=True)
    R = np.array(R)
    print("MEAN gain of PFF pre-training: chain {:+.4f} / {:+.4f} ({} / {} ahead), member frame accuracy {:+.4f}".format(*R.mean(0)[:2], int((R[:, 0] > 0).sum()), len(R), R.mean(0)[2]))


if __name__ == "__main__":
    main(*sys.argv[1:2])
