"""P9fa: the production boundary GRU (mtah.grumodel.external_gru target="boundary", swap, Metrica; 3 seeds 0-2) with
twelve seeds: seeds 3-11 trained with the same recipe (same data, so the same standardisation) and pooled with 0-2 into
one ensemble. Twelve seeds for the team members were ahead on SoccerTrack (8/8) and PFF (P9ev, P9ew).
  train  (GPU, Kaggle) seeds 3-11 -> models/boundary_gru_swap_met_s345678910_11_<hash>.pt (G's own cache path)
  st     SoccerTrack full chain as P9ei with the v30 team members (twelve seeds), boundary 3 vs 12 seeds
  pff    PFF World Cup (P9ew's judge: team call only, dead from labels), boundary 3 vs 12 seeds

    .venv/bin/python analysis/p9fa_boundary_seeds.py train | st | pff

Registered before the run (15 Nov): the twelve-seed boundary goes into the build only if it is not behind the
three-seed one on SoccerTrack and on PFF at either referee.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402

NEW = tuple(range(3, 12))


def pooled(dev=None):
    """The production three-seed boundary model and the nine-seed one pooled (same recipe, same mu / sd)."""
    a = G.external_gru(target="boundary", swap=True, metrica=True)
    b = G.external_gru(target="boundary", swap=True, metrica=True, seeds=NEW)
    assert np.allclose(a.mu, b.mu) and np.allclose(a.sd, b.sd), "the two boundary models were not trained on the same data"
    return a, G.GRUEnsemble(a.nets + b.nets, a.mu, a.sd, a.dev, getattr(a, "arch", "gru"), getattr(a, "heads", 1))


def train():
    G.external_gru("cuda", target="boundary", swap=True, metrica=True, seeds=NEW)
    print("trained boundary seeds", NEW, flush=True)


def st():
    from p9bw_st_chain_base import no_training
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9b_oof_sweeps import HI, LO
    from p9cs_st_video_dead import TEST
    from p9ei_st_members_c import logit
    from p9ev_seeds12 import members, team
    from p9l_rescore_calibrated import merged
    b3, b12 = pooled()
    G.train = no_training
    ms = members()
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, dead = z["g"], z["lab"], z["dead3"].astype(np.float64)
        e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
        p = 1 / (1 + np.exp(-(logit(team(ms, m)[1]) + e)))
        l3 = merged(lab, 3.0)
        r = {}
        for k, bm in (("b3", b3), ("b12", b12)):
            b = np.clip((HI - np.asarray(G.switch_costs(bm, m), np.float64)) / (HI - LO), 0, 1)
            c = np.where(g, sm.decode_halves(p, dead, g, b), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[mid] = r
        print(mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("SoccerTrack", rows)


def pff():
    from mtah.pff import from_pff, pff_ids
    from mtah.teammodel import decode
    from p9bw_st_chain_base import no_training
    from p9ev_seeds12 import members, team
    from p9l_rescore_calibrated import merged
    b3, b12 = pooled()
    G.train = no_training
    ms = members()
    rows = {}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, merge_s=1.6, ball="visible")
        g, lab = m["half"] > 0, m["labels"]
        l3 = merged(lab, 3.0)
        alive = g & np.isin(lab, (pm.H, pm.A))
        p = team(ms, m)[1]
        r = {}
        for k, bm in (("b3", b3), ("b12", b12)):
            c = np.where(g, np.where(alive, decode(p, alive, G.switch_costs(bm, m)), pm.D), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[gid] = r
        print(gid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("PFF", rows)


def report(name, rows):
    a, c = np.array([rows[k]["b3"] for k in rows]), np.array([rows[k]["b12"] for k in rows])
    d = (c - a).mean(0)
    print("{} {}: b3 {:.4f} / {:.4f}, b12 {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s) -> {}".format(
        name, len(rows), *a.mean(0), *c.mean(0), *d, int(((c - a)[:, 0] > 0).sum()), len(rows), "holds" if (d >= 0).all() else "FAILS"))


if __name__ == "__main__":
    {"train": train, "st": st, "pff": pff}[sys.argv[1]]()
