"""SoccerTrack v2 (10 matches) in the team GRU's training, on top of v10's recipe. Team call only, flat switch,
3 seeds. GPU.

    MTAH_SOCCERTRACK=... python analysis/p7z_soccertrack_gru.py      # -> eda_out/reports/P7z_soccertrack_gru.md

v10's team GRU: DFL + Metrica 1-3, every match also team-swapped, short possessions weighted x2. +ST: the
ten SoccerTrack matches (labels from the ball track's possession status) added, degraded like DFL (their
full-pitch players and event-anchored ball would otherwise be cleaner than our ATD). DFL test (leave one
DFL match out) and, as a check that the labels can be learnt, SoccerTrack test (leave one out, N_ST).
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
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack, soccertrack_ids  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

N_ST = 4
OUT = os.path.join(ROOT, "eda_out", "reports", "P7z_soccertrack_gru.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    dfl = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    st = []
    for i in soccertrack_ids():
        m = from_soccertrack(i)
        g = m["half"] > 0
        seen = [np.isfinite(m["px"][g][:, m["slot_team"] == t]).any() for t in (0, 1)]
        if np.mean(np.isin(m["labels"][g], (pm.H, pm.A))) < 0.3 or not all(seen):
            print("skip", i, "(no usable possession status or a team without players)", flush=True)
            continue
        st.append(degrade(m, dict(rates_for("st"), smooth_frames=25), seed=0))
    ms = dfl + met + st
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    S = [G.short_mask(m["labels"]) for m in ms]
    nd, nm = len(dfl), len(met)
    print("features done", nd, nm, len(st), flush=True)
    def item(k, swapped):
        m = sw[k] if swapped else ms[k]
        w = np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(S[k], 2.0, 1.0).astype(np.float32)
        return ((Ts if swapped else T)[k], (m["labels"] == pm.H).astype(np.float32), w, m["half"] > 0)
    data = lambda ks: [item(k, s) for k in ks for s in (False, True)]
    rows = []
    tests = list(range(nd)) + list(range(nd + nm, nd + nm + N_ST))
    for t in tests:
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        base = [k for k in range(nd + nm) if k != t]
        with_st = base + [k for k in range(nd + nm, len(ms)) if k != t]
        r = {"match": m["name"], "test": "DFL" if t < nd else "SoccerTrack",
             "v10": sc(fit_predict(data(base), (T[t], g))), "v10+st": sc(fit_predict(data(with_st), (T[t], g)))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7z: SoccerTrack v2 takım GRU'sunun eğitiminde (v10 tarifi, sadece takım kararı, 3 tohum)", "", "`analysis/p7z_soccertrack_gru.py`.", "",
             "| maç | v10 | v10 + SoccerTrack |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["v10"], r["v10+st"]) for r in rows]
    for test in ("DFL", "SoccerTrack"):
        d = df[df.test == test]
        lines += ["", "{} testi: {:.3f} → {:.3f} ({:+.3f}, {} / {} maçta önde).".format(test, d.v10.mean(), d["v10+st"].mean(), (d["v10+st"] - d.v10).mean(),
                                                                                    int((d["v10+st"] > d.v10).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
