"""P9fq: the dead-ball classifier and GRU trained with PFF World Cup games made to look like our ATD added to their
production copies (p4_common.pff_dead_copies: PFF's seen-or-estimated ball, our degradation, PFF's event-derived dead
ball). Why: on SoccerTrack a perfect dead ball is worth +0.072 (16 Nov oracle); PFF's labels are the closest to the
organisers'; every earlier PFF dead-ball attempt (P9dc, P9dx) used its raw broadcast ball.
PFF games: pff_ids()[12:64] for training (52); pff_ids()[:12] stay held out (the PFF judge).

  train   fit the production classifier and the +PFF classifier (models/dead_clf_{base,pff}.joblib), train the +PFF GRU
          (p4_common.dead_gru(..., pff=...)); the production GRU file must be in models/
  st      SoccerTrack full chain as v33 (P9di inputs: team, boundary, video dead; P(dead) = video / 2 + (clf + GRU) / 4,
          D weight 1.0), four arms: base, clf+PFF, GRU+PFF, both
  pff     held-out PFF 12 (same ATD-like copies as training, seed 7): dead frame accuracy of to_blocks((clf + GRU) / 2)
          and spellF1 with the labels' team, same four arms
  ours    our match, label-free (run locally): dead share, agreement with the video's dead calls, set-piece recall

Registered before the runs (16 Nov): an arm replaces production only if on SoccerTrack it is ahead by >= +0.003 at both
referees and on >= 5/8 matches at 1.6 s, on held-out PFF its dead accuracy is not lower, and on our match its
agreement with the video's dead calls is not lower. If several arms pass, the one with the higher SoccerTrack mean.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p4_common as P4  # noqa: E402

CLF = {k: os.path.join(ROOT, "models", "dead_clf_{}.joblib".format(k)) for k in ("base", "pff")}
ARMS = ("base", "clf+pff", "gru+pff", "both")


def pff_train():
    from mtah.pff import pff_ids
    return tuple(pff_ids()[12:64])


def kw():
    return dict(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True)


def train():
    import joblib
    from mtah.deadball import features_v2
    assert os.path.exists(P4.dead_gru_path(nseeds=3, **kw())), "upload the production dead GRU"
    for k, pff in (("base", ()), ("pff", pff_train())):
        if not os.path.exists(CLF[k]):
            joblib.dump(P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, mix=True, pff=pff, **kw()), CLF[k])
            print("fitted classifier", k, flush=True)
    P4.dead_gru(nseeds=3, pff=pff_train(), **kw())
    print("trained", os.path.basename(P4.dead_gru_path(nseeds=3, pff=pff_train(), **kw())), flush=True)


def models():
    import joblib
    from mtah import grumodel as G
    from p9bw_st_chain_base import no_training
    G.train = no_training
    clf = {k: joblib.load(CLF[k]) for k in CLF}
    gru = {"base": P4.dead_gru(nseeds=3, **kw()), "pff": P4.dead_gru(nseeds=3, pff=pff_train(), **kw())}
    return clf, gru


def pq(clf, gru, m):
    """{arm: (classifier p, GRU q)} on match m's halves."""
    from mtah import grumodel as G
    from mtah.deadball import features_v2
    g = m["half"] > 0
    F = features_v2(m)
    p = {k: np.where(g, 0.0, 0.0) for k in clf}
    for k, c in clf.items():
        p[k][g] = c.predict_proba(F[g])[:, 1]
    q = {k: np.where(g, np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0), 0) for k, gm in gru.items()}
    return {"base": (p["base"], q["base"]), "clf+pff": (p["pff"], q["base"]), "gru+pff": (p["base"], q["pff"]), "both": (p["pff"], q["pff"])}


def st():
    import copy
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9cs_st_video_dead import TEST
    from p9l_rescore_calibrated import merged
    clf, gru = models()
    cfg = copy.deepcopy(sm.CONFIG)
    cfg["decoder"]["emission_weights"][2] = 1.0
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab = z["g"], z["lab"]
        v = np.clip(3 * z["dead3"].astype(np.float64) - 2 * z["dead2"].astype(np.float64), 0, 1)
        l3 = merged(lab, 3.0)
        r = {}
        for arm, (p, q) in pq(clf, gru, m).items():
            c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), 0.5 * v + 0.25 * (p + q), g, z["b"].astype(np.float64), cfg=cfg), pm.X).astype(np.int8)
            r[arm] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[mid] = r
        print(mid, {k: np.round(x, 4).tolist() for k, x in r.items()}, flush=True)
    S = np.array([[rows[m][a] for a in ARMS] for m in rows])          # match x arm x referee
    for i, a in enumerate(ARMS[1:], 1):
        d = (S[:, i] - S[:, 0]).mean(0)
        n = int((S[:, i, 0] > S[:, 0, 0]).sum())
        print("SoccerTrack {:8s} {:.4f} / {:.4f} vs base {:.4f} / {:.4f}: {:+.4f} / {:+.4f} ({}/8) -> {}".format(
            a, *S[:, i].mean(0), *S[:, 0].mean(0), *d, n, "holds" if (d >= 0.003).all() and n >= 5 else "FAILS"))


def pff():
    from m1_dead_classifier import to_blocks
    from mtah.degrade import degrade, rates_for
    from mtah.pff import from_pff, pff_ids
    clf, gru = models()
    acc, sp = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for gid in pff_ids()[:12]:
        m = degrade(from_pff(gid, ball="all", merge_s=1.6), dict(rates_for("pff"), loss_per_min=0.0, smooth_frames=25), seed=7)
        g, lab = m["half"] > 0, m["labels"]
        truth = lab == pm.D
        team = np.where(np.isin(lab, (pm.H, pm.A)), lab, pm.H)
        for arm, (p, q) in pq(clf, gru, m).items():
            d = to_blocks((p + q) / 2) & g
            acc[arm].append(np.mean(d[g] == truth[g]))
            c = np.where(g, np.where(d, pm.D, team), pm.X).astype(np.int8)
            sp[arm].append(pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"])
        print(gid, {a: round(acc[a][-1], 4) for a in ARMS}, flush=True)
    for a in ARMS:
        print("PFF held out {:8s} dead accuracy {:.4f}  spellF1 with the labels' team {:.4f}".format(a, np.mean(acc[a]), np.mean(sp[a])))


def ours():
    from m1_dead_classifier import to_blocks
    from mtah.match import from_tactical
    clf, gru = models()
    m = from_tactical()
    m["labels"] = np.zeros(len(m["half"]), np.int8)          # never read
    g = m["half"] > 0
    vd = np.load(os.path.join(ROOT, "cache", "barath_dense_avg7.npz"))["q_dead"].astype(np.float64)[g] > 0.5
    for arm, (p, q) in pq(clf, gru, m).items():
        d = to_blocks((p + q) / 2) & g
        print("ours {:8s} dead share {:.3f}  agreement with the video's dead calls {:.4f}".format(arm, d[g].mean(), np.mean(d[g] == vd)))


if __name__ == "__main__":
    {"train": train, "st": st, "pff": pff, "ours": ours}[sys.argv[1]]()
