"""Step 3: the learned team call (mtah.teammodel) against the control rule, leave-one-match-out on DFL.

    uv run python analysis/p6b_team_model.py [smooth_frames]   # -> eda_out/reports/P6b_team_model_s{N}.md

7 DFL matches, degraded like our ATD (mtah.degrade, seed 0; the ball smoothed over smooth_frames,
default 25, the closest width to our ATD's kinks). For each test match the model is trained on the
other 6; the switch cost of the Viterbi is chosen on those 6 too (nested: fit on 5, score on the
6th, every rotation), so the test match never picks anything. Team call only: the dead ball comes
from the labels for both labelers, so the difference is the team call. Our match is not used.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402

SWITCHES = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6b_team_model_s{}_{}.md")


def spell(m, codes):
    return pm.spell_scores(m["labels"], codes, pm.valid_runs(m["labels"]))["mean_f1"]


def main(smooth=25, variant="base"):
    """variant: "base", or any of "touch" (mtah.teammodel.touch_features) and "aug" (training also on
    the same matches degraded with an unsmoothed ball), "seeds" (and with degradation seeds 1, 2), "big" (a
    larger booster), "motion" (mtah.teammodel.motion_features), "met" (Metrica games 1-3 added to training), "reach" (mtah.teammodel.reach_features), "carry" (mtah.teammodel.carry_features), "ens" (3 boosters averaged), e.g. "touch_aug_seeds"."""
    touches, aug, seeds, big = "touch" in variant, "aug" in variant, "seeds" in variant, "big" in variant
    motion, met, reach, carry = "motion" in variant, "met" in variant, "reach" in variant, "carry" in variant
    ens = 3 if "ens" in variant else 0
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raw = [from_dfl(i) for i in ids]
    ms = [degrade(r, dict(rates_for(i), smooth_frames=smooth), seed=0) for r, i in zip(raw, ids)]
    data, extra = [], []
    for k, m in enumerate(ms):
        F, _ = team_features(m, touches, motion, reach, carry)
        alive = np.isin(m["labels"], (pm.H, pm.A))
        data.append((F, alive, (m["labels"] == pm.H).astype(int)))
        copies = ([(0, 0)] if aug else []) + ([(smooth, 1), (smooth, 2)] if seeds else [])
        Xa, ya = [], []
        for sm, sd in copies:
            a = degrade(raw[k], dict(rates_for(ids[k]), smooth_frames=sm), seed=sd)
            Fa, _ = team_features(a, touches, motion, reach, carry)
            al = np.isin(a["labels"], (pm.H, pm.A))
            Xa.append(Fa[al])
            ya.append((a["labels"][al] == pm.H).astype(int))
        extra.append((Xa, ya))
        print(m["name"], F.shape, flush=True)

    mX, my = [], []
    if met:                                   # Metrica sample games 1-3 in every training set, never tested
        from mtah.external import from_metrica_sample
        for gm in (1, 2, 3):
            e = degrade(from_metrica_sample(gm), dict(rates_for("metrica"), smooth_frames=smooth), seed=0)
            al = np.isin(e["labels"], (pm.H, pm.A))
            mX.append(team_features(e, touches, motion, reach, carry)[0][al])
            my.append((e["labels"][al] == pm.H).astype(int))

    def fit_on(ks):
        X = [data[k][0][data[k][1]] for k in ks] + [x for k in ks for x in extra[k][0]] + mX
        y = [data[k][2][data[k][1]] for k in ks] + [z for k in ks for z in extra[k][1]] + my
        return fit(X, y, big, ens)
    rows = []
    for t, m in enumerate(ms):
        train = [k for k in range(len(ms)) if k != t]
        # nested choice of the switch cost on the 6 training matches
        inner = {s: [] for s in SWITCHES}
        for v in train:
            sub = [k for k in train if k != v]
            p = fit_on(sub).predict_proba(data[v][0])[:, 1]
            for s in SWITCHES:
                inner[s].append(spell(ms[v], decode(p, data[v][1], s)))
        best = max(SWITCHES, key=lambda s: np.mean(inner[s]))
        p = fit_on(train).predict_proba(data[t][0])[:, 1]
        alive = data[t][1]
        model = decode(p, alive, best)
        frame = np.where(alive, np.where(p >= 0.5, pm.H, pm.A), pm.D).astype(np.int8)
        rule = control(m, dead="truth")
        rows.append({"match": m["name"], "switch": best, "control": spell(m, rule), "model": spell(m, model),
                     "frame_argmax": spell(m, frame),
                     "acc_model": np.mean(model[alive] == m["labels"][alive]),
                     "acc_control": np.mean(rule[alive] == m["labels"][alive]),
                     "n_gt": len(pm.possession_spells(m["labels"], pm.valid_runs(m["labels"]))),
                     "n_model": len(pm.possession_spells(model, pm.valid_runs(m["labels"]))),
                     "n_control": len(pm.possession_spells(rule, pm.valid_runs(m["labels"])))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6b: öğrenen takım kararı vs kontrol kuralı (DFL, maç dışarıda bırakarak)", "",
             "`analysis/p6b_team_model.py {} {}`. Bozulmuş DFL, top {} kare yumuşatılmış; ölü top etiketten (sadece takım kararı).".format(smooth, variant, smooth),
             "Değişim cezası her test maçı için diğer 6 maçta iç döngüyle seçildi.", "",
             "| maç | ceza | kontrol spellF1 | model spellF1 | model (Viterbi'siz) | oyun içi doğruluk kontrol / model | possession GT / kontrol / model |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {:g} | {:.3f} | {:.3f} | {:.3f} | {:.3f} / {:.3f} | {} / {} / {} |".format(
            r["match"], r["switch"], r["control"], r["model"], r["frame_argmax"], r["acc_control"], r["acc_model"],
            r["n_gt"], r["n_control"], r["n_model"]))
    d = df.model - df.control
    lines += ["", "Ortalama: kontrol {:.3f}, model {:.3f}; fark {:+.3f} (maçlarda {:+.3f} .. {:+.3f}, {} / {} maçta model önde).".format(
        df.control.mean(), df.model.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT.format(smooth, variant), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25, *sys.argv[2:])
