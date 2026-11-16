"""SkillCorner open data (20 A-League matches) added to the team GRU's training. DFL LOMO, team call only. GPU.

    python analysis/p7a_skillcorner.py      # -> eda_out/reports/P7a_skillcorner.md

Per DFL test match (degraded as production, smooth 25), the team GRU (mtah.grumodel recipe, 3 seeds) trained
on the other six DFL matches alone, or with the 20 SkillCorner matches added: A, SkillCorner's detected
positions only, as they are (broadcast view, ~4.5 players a team); B, its extrapolated full tracking through
our degradation, like DFL. Decoded with the flat switch, dead ball from the labels. Also the reverse check:
leave one SkillCorner match out (B), trained on DFL + the other 19, to see that its labels can be learnt.
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_skillcorner, skillcorner_ids  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p6v_gru_dead import gru_fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7a_skillcorner.md")
N_SC_CHECK = 4          # SkillCorner matches left out in the reverse check


def item(m, T):
    lab = m["labels"]
    return (T, (lab == pm.H).astype(np.float32), np.isin(lab, (pm.H, pm.A)).astype(np.float32), m["half"] > 0)


def score(m, p):
    alive = np.isin(m["labels"], (pm.H, pm.A))
    c = np.where(m["half"] > 0, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    dfl = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    scs = skillcorner_ids()
    sa = [from_skillcorner(i, detected_only=True) for i in scs]
    sb = [degrade(from_skillcorner(i, detected_only=False), dict(rates_for("sc"), smooth_frames=25), seed=0) for i in scs]
    Td, Ta, Tb = [features(m) for m in dfl], [features(m) for m in sa], [features(m) for m in sb]
    print("features done", len(scs), "SkillCorner matches", flush=True)
    rows = []
    for t in range(len(ids)):
        base = [item(dfl[k], Td[k]) for k in range(len(ids)) if k != t]
        test = (Td[t], dfl[t]["half"] > 0)
        r = {"match": dfl[t]["name"],
             "dfl": score(dfl[t], gru_fit_predict(base, test, seeds=G.SEEDS)),
             "dfl+A": score(dfl[t], gru_fit_predict(base + [item(m, T) for m, T in zip(sa, Ta)], test, seeds=G.SEEDS)),
             "dfl+B": score(dfl[t], gru_fit_predict(base + [item(m, T) for m, T in zip(sb, Tb)], test, seeds=G.SEEDS))}
        rows.append(r)
        print(r, flush=True)
    rev = []
    for t in range(N_SC_CHECK):
        tr = [item(m, T) for m, T in zip(dfl, Td)] + [item(sb[k], Tb[k]) for k in range(len(scs)) if k != t]
        only = [item(m, T) for m, T in zip(dfl, Td)]
        test = (Tb[t], sb[t]["half"] > 0)
        rr = {"match": sb[t]["name"], "dfl only": score(sb[t], gru_fit_predict(only, test, seeds=G.SEEDS)),
              "dfl+sc": score(sb[t], gru_fit_predict(tr, test, seeds=G.SEEDS))}
        rev.append(rr)
        print(rr, flush=True)
    df, dr = pd.DataFrame(rows), pd.DataFrame(rev)
    lines = ["# P7a: SkillCorner açık verisi GRU eğitiminde (DFL, maç dışarıda bırakarak, 3 tohum, sadece takım kararı)", "",
             "`analysis/p7a_skillcorner.py`. A: SkillCorner'ın sadece görülen konumları; B: tam (tahminli) iz + bizim bozma.", "",
             "| maç | sadece DFL | DFL + SC A | DFL + SC B |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["dfl"], r["dfl+A"], r["dfl+B"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} |".format(df["dfl"].mean(), df["dfl+A"].mean(), df["dfl+B"].mean()), "",
              "A önde {} / {}, B önde {} / {} maçta.".format(int((df["dfl+A"] > df["dfl"]).sum()), len(df), int((df["dfl+B"] > df["dfl"]).sum()), len(df)), "",
              "## Ters kontrol: SkillCorner maçı dışarıda (B)", "", "| maç | sadece DFL ile | DFL + diğer SC ile |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["dfl only"], r["dfl+sc"]) for r in rev]
    lines += ["| ortalama | {:.3f} | {:.3f} |".format(dr["dfl only"].mean(), dr["dfl+sc"].mean())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
