"""Step 3b: the learned team call in the full chain, on DFL (LOMO) and on our match (GT-free).

    uv run python analysis/p6e_full_chain.py SWITCH     # -> eda_out/reports/P6e_full_chain.md,
                                                        #    evaluation/results/v2_team.csv

SWITCH is the Viterbi switch cost, fixed beforehand from P6b's nested choice (never from our match).
DFL, degraded with the ball smoothed over 25 frames: for each test match the team model and the
dead-ball classifier (analysis/m1_dead_classifier.py features) are trained on the other 6 DFL
matches plus degraded Metrica 1-2; the control rule gets the same dead ball.
Our match: both models trained on all 7 DFL + Metrica 1-2, the dead ball and periods exactly as the
v1 chain (analysis/p4_common.ours); GT-free checks: possessions per 90 graded minutes, agreement with
Smart Tagging outside the public window, and the kit-colour judge (analysis/p5m_kit_team.kit_team_at)
on frames where the ATD sees the ball and the two calls differ. The public window is printed as a
correctness check only.
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
from m1_dead_classifier import SEED, features, table, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6e_full_chain.md")
SUB = os.path.join(ROOT, "evaluation", "results", "v2_team.csv")
SMOOTH = 25


def dead_clf(ms):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(ms))


def dead_of(clf, m):
    g = m["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(m)[g])[:, 1]
    return to_blocks(p) & g


def spell(m, codes):
    return pm.spell_scores(m["labels"], codes, pm.valid_runs(m["labels"]))["mean_f1"]


def team_data(m):
    F, _ = team_features(m)
    alive = np.isin(m["labels"], (pm.H, pm.A))
    return F, alive, (m["labels"] == pm.H).astype(int)


def main(switch):
    switch = float(switch)
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    dfl = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=SMOOTH), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=SMOOTH), seed=0) for g in (1, 2)]
    data = [team_data(m) for m in dfl + met]
    print("features done", flush=True)
    rows = []
    for t, m in enumerate(dfl):
        others = [k for k in range(len(dfl) + len(met)) if k != t]
        clf = fit([data[k][0][data[k][1]] for k in others], [data[k][2][data[k][1]] for k in others])
        dead = dead_of(dead_clf([(dfl + met)[k] for k in others]), m)
        alive = (m["half"] > 0) & ~dead
        model = decode(clf.predict_proba(data[t][0])[:, 1], alive, switch)
        rule = control(m, dead=dead)
        rows.append({"match": m["name"], "control": spell(m, rule), "model": spell(m, model)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)

    # our match
    from mtah.frames import load_frames
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    from p4_common import ours
    from p5m_kit_team import kit_team_at
    from mtah.video_match import _hom
    ours_m = from_tactical()
    clf = fit([d[0][d[1]] for d in data], [d[2][d[1]] for d in data])
    dead, v1 = ours()
    g = ours_m["half"] > 0
    F, _ = team_features(ours_m)
    p = clf.predict_proba(F)[:, 1]
    v2 = np.where(g, decode(p, g & ~dead, switch), pm.X).astype(np.int8)
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(v2)), "label": np.array(pm.LABELS)[v2]}).to_csv(SUB, index=False)
    fr = load_frames()
    played = g & ~np.isin(fr["period"].values, NOT_PLAYED)
    minutes = played.sum() / 25 / 60
    runs = pm.valid_runs(np.where(g, pm.D, pm.X).astype(np.int8))
    per90 = {k: len(pm.possession_spells(c, runs)) / minutes * 90 for k, c in (("v1", v1), ("v2", v2))}
    tag = np.where(fr["tag_H"].values > fr["tag_A"].values, pm.H, np.where(fr["tag_A"].values > fr["tag_H"].values, pm.A, -1))
    outside = np.ones(len(v2), bool)
    outside[11185:18685] = False
    alive = played & ~dead
    judged = alive & outside & (tag >= 0)
    st = {k: np.mean(c[judged] == tag[judged]) for k, c in (("v1", v1), ("v2", v2))}
    has = ~np.isnan(ours_m["bx"])
    ch = (v1 != v2) & alive & has & outside
    kit = kit_team_at(ch, np.c_[ours_m["bx"], ours_m["by"]], ours_m, _hom())
    k = ch & (kit >= 0)
    kv1, kv2 = np.mean(np.where(kit[k] == 0, pm.H, pm.A) == v1[k]), np.mean(np.where(kit[k] == 0, pm.H, pm.A) == v2[k])
    pub = {key: pm.evaluate(ours_m["labels"], c, bootstrap_n=0)["spell_f1_mean"] for key, c in (("v1", v1), ("v2", v2))}
    d = df.model - df.control
    lines = ["# P6e: öğrenen takım kararı tam zincirde", "",
             "`analysis/p6e_full_chain.py {:g}`. DFL bozulmuş (top {} kare yumuşatılmış); takım modeli ve ölü top sınıflandırıcısı test maçı dışarıda eğitildi.".format(switch, SMOOTH), "",
             "| maç | kontrol (tam zincir) | model (tam zincir) |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["control"], r["model"]) for r in rows]
    lines += ["", "DFL ortalama: kontrol {:.3f}, model {:.3f}, fark {:+.3f} ({:+.3f} .. {:+.3f}; {} / {} maçta model önde).".format(
        df.control.mean(), df.model.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d)), "",
        "## Bizim maç (GT'siz; aynı ölü top ve dönemler)", "",
        "| | v1 (kontrol) | v2 (model) |", "|---|---|---|",
        "| 90 dk'da possession | {:.0f} | {:.0f} |".format(per90["v1"], per90["v2"]),
        "| Smart Tagging uyumu (public dışı, oyun içi) | {:.1%} | {:.1%} |".format(st["v1"], st["v2"]),
        "| forma hakemi, kararların ayrıştığı ve ATD topunun olduğu {} kare | {:.1%} | {:.1%} |".format(int(k.sum()), kv1, kv2),
        "| public spellF1 (sadece kontrol, seçim değil) | {:.3f} | {:.3f} |".format(pub["v1"], pub["v2"])]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
