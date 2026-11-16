"""Step 3c: the possessor crops (p6d) as evidence in the learned team call, weighted GT-free. Colab.

    python analysis/p6f_fuse_crops.py SWITCH [booster|gru]      # -> eda_out/reports/P6f_fuse_crops.md, evaluation/results/v3_crops.csv

The team model is the production one (mtah.teammodel.external_model, features), the dead ball v3.
Evidence per frame: the person box with the highest "has the ball" probability, if >= TOP_MIN; its team
is the ATD slot whose position is within MATCH_M of the box's foot point and clearly the nearest
(mtah.kits.slot_labels); boxes without one give no evidence. The evidence is calibrated on ball_v1's
validation blocks without any ground truth: where the ATD sees the ball slowly and one player is
clearly nearest it (the p6c positives' rule), how often the evidence's team is that player's team, per
probability bin -> log-odds. It enters the Viterbi as scale x log-odds added to the model's log-odds.

scale is chosen by a ball-dropout test on our own match, also GT-free: on the validation blocks the
ATD ball is removed in contiguous stretches (lengths drawn like our in-play gaps, DROP_SHARE of the
time); the reference is the learned call WITH the ATD ball; each scale's call without it is scored
against that reference with the official spell metric. The public window is outside the blocks
(p3e) and is printed only as a correctness check.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.data import PITCH  # noqa: E402
from mtah.possessor import BINS, PROB, TOP_MIN, calibrate, crop_evidence, evidence_logodds  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402
from p3a_yolo_ball import to_pitch  # noqa: E402
from p3e_ball_pseudolabels import BLOCK  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6f_fuse_crops.md")
SUB = os.path.join(ROOT, "evaluation", "results", "v3_crops.csv")
SCALES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
DROP_SHARE, SEED = 0.4, 0


def drop_ball(m, val):
    """A copy of m with the ATD ball removed on DROP_SHARE of the validation frames, in contiguous stretches."""
    rng = np.random.default_rng(SEED)
    bx, by = m["bx"].copy(), m["by"].copy()
    gaps = []
    seen = ~np.isnan(m["bx"])
    e = np.diff(np.r_[0, (~seen & (m["half"] > 0)).astype(np.int8), 0])
    lens = (np.flatnonzero(e == -1) - np.flatnonzero(e == 1))
    lens = lens[lens >= 25]
    drop = np.zeros(len(bx), bool)
    target = DROP_SHARE * len(val)
    vs = set(val.tolist())
    while drop[val].sum() < target:
        a = int(rng.choice(val))
        ln = int(rng.choice(lens))
        span = np.arange(a, min(a + ln, len(bx)))
        span = span[[s in vs for s in span]]
        drop[span] = True
    bx[drop], by[drop] = np.nan, np.nan
    out = dict(m)
    out.update(bx=bx, by=by)
    return out, drop


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main(switch, model="booster"):
    switch = float(switch)
    from mtah.data import load_tactical
    from mtah.match import from_tactical
    from mtah.teammodel import external_model, features
    from p4_common import dead_v3, ours
    m = from_tactical()
    m["hom"] = load_tactical()["hom"]
    _, v1 = ours()
    dead = dead_v3()
    g = m["half"] > 0
    alive = g & ~dead
    val = np.flatnonzero(g & ((np.arange(len(g)) // BLOCK) % 5 == 4))
    t, p = crop_evidence(m)
    lo = calibrate(m, t, p, val)
    E = evidence_logodds(t, p, lo)
    if model == "gru":
        from mtah.grumodel import external_gru, predict
        gm = external_gru()
        prob = lambda mm: predict(gm, mm)
    else:
        clf = external_model()
        prob = lambda mm: clf.predict_proba(features(mm))[:, 1]
    base = logit(prob(m))
    md, drop = drop_ball(m, val)
    based = logit(prob(md))
    ref = decode(1 / (1 + np.exp(-base)), alive, switch)
    vmask = np.zeros(len(g), bool)
    vmask[val] = True
    gt_ref = np.where(vmask, ref, pm.X).astype(np.int8)
    rows = []
    for s in SCALES:
        c = decode(1 / (1 + np.exp(-(based + s * E))), alive, switch)
        c = np.where(vmask, c, pm.X).astype(np.int8)
        r = pm.spell_scores(gt_ref, c, pm.valid_runs(gt_ref))["mean_f1"]
        acc = np.mean(c[drop & alive] == ref[drop & alive])
        rows.append((s, r, acc))
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r[1])[0]
    v2 = np.where(g, ref, pm.X).astype(np.int8)
    v3 = np.where(g, decode(1 / (1 + np.exp(-(base + best * E))), alive, switch), pm.X).astype(np.int8)
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(v3)), "label": np.array(pm.LABELS)[v3]}).to_csv(SUB, index=False)
    pd.DataFrame({"frame": np.arange(len(v2)), "label": np.array(pm.LABELS)[v2]}).to_csv(SUB.replace("v3_crops", "v2_team"), index=False)
    pub = {k: pm.evaluate(m["labels"], c, bootstrap_n=0)["spell_f1_mean"] for k, c in (("v1", v1), ("v2", v2), ("v3", v3))}
    minutes = g.sum() / 25 / 60
    runs = pm.valid_runs(np.where(g, pm.D, pm.X).astype(np.int8))
    per90 = {k: len(pm.possession_spells(c, runs)) / minutes * 90 for k, c in (("v1", v1), ("v2", v2), ("v3", v3))}
    lines = ["# P6f: kırpıntı kanıtı öğrenen takım kararında (GT'siz)", "",
             "`analysis/p6f_fuse_crops.py {:g}`. Kanıtın kalibrasyonu (doğrulama blokları, ATD topu yavaş ve tek oyuncu açıkça en yakın):".format(switch), "",
             "| olasılık | kare | takım doğru | log-odds |", "|---|---|---|---|"]
    lines += ["| {:.2f}–{:.2f} | {} | {:.1%} | {:.2f} |".format(a, min(z, 1), n, acc, w) for a, z, n, acc, w in lo]
    lines += ["", "Top bırakma testi: doğrulama bloklarında ATD topu karelerin {:.0%}'inde bitişik parçalar hâlinde silindi; referans topla çalışan model.".format(DROP_SHARE), "",
              "| ölçek | spellF1 (referansa göre) | silinen oyun içi karelerde uyum |", "|---|---|---|"]
    lines += ["| {:g} | {:.3f} | {:.1%} |".format(s, r, a) for s, r, a in rows]
    lines += ["", "Seçilen ölçek {:g}.".format(best), "",
              "| | v1 (kontrol) | v2 (model) | v3 (model + kırpıntı) |", "|---|---|---|---|",
              "| 90 dk'da possession | {:.0f} | {:.0f} | {:.0f} |".format(per90["v1"], per90["v2"], per90["v3"]),
              "| public spellF1 (sadece kontrol, seçim değil) | {:.3f} | {:.3f} | {:.3f} |".format(pub["v1"], pub["v2"], pub["v3"])]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
