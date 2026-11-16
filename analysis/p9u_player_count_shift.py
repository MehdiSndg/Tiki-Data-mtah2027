"""The domain shift behind our match's under-splitting (P9t: per team ~10.8 tracked players on our match vs ~9.7 on the
degraded DFL, AUROC 0.72-0.75; nothing else above 0.58). Three team-GRU variants (v11 recipe: DFL + Metrica, swapped,
short x2; 3 seeds):
  A  production: usual degradation, all features;
  B  usual degradation, the player-count features (n_H, n_A) zeroed;
  C  the degradation's camera window leaves the players alone (view_drops_players=False), all features.
Each scored leave one match out (DFL and Metrica, team call, flat switch) on BOTH test degradations (usual, players
kept), and trained on all ten matches to count possessions per 90 min on our match (v14's dead ball; no labels of
our match are read). GPU.

    python analysis/p9u_player_count_shift.py <sub_dir> [only=<match>,...] [final]      # -> eda_out/reports/P9u_player_count_shift.md
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
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical, swap_teams  # noqa: E402
from mtah.teammodel import MOTION, TOUCHES, decode, team_features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
FINAL = "final" in sys.argv[1:]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9u_player_count_shift{}.md".format("_" + "_".join(ONLY) if ONLY else ("_final" if FINAL else "")))


def rates(key, keep):
    return dict(rates_for(key), smooth_frames=25, view_drops_players=not keep)


def feats(m, drop):
    F, names = team_features(m, TOUCHES, MOTION)
    if drop:
        F = F.copy()
        F[:, [names.index("n_H"), names.index("n_A")]] = 0
    return F


VARIANTS = {"A": (False, False), "B": (False, True), "C": (True, False)}      # (train keeps players, drop counts)


def main(sub_dir, *rest):
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    deg = {keep: [degrade(r, rates(k, keep), seed=0) for r, k in raws] for keep in (False, True)}
    for keep in (False, True):
        print("players per team, keep={}: {:.2f}".format(keep, np.mean([np.nanmean(np.sum(~np.isnan(m["px"]), 1)[m["half"] > 0]) / 2 for m in deg[keep][:len(ids)]])), flush=True)
    F = {(keep, drop): [(feats(m, drop), feats(swap_teams(m), drop)) for m in deg[keep]] for keep in (False, True) for drop in (False, True)}
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)

    def train_set(keep, drop, ks):
        out = []
        for k in ks:
            m = deg[keep][k]
            sw, sh = swap_teams(m), G.short_mask(m["labels"])
            for mm, FF in ((m, F[keep, drop][k][0]), (sw, F[keep, drop][k][1])):
                out.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        return out

    rows = []
    if FINAL:
        sub = pd.read_csv(os.path.join(sub_dir, "v14_chain.csv")).label.values
        ours = from_tactical()
        g = ours["half"] > 0
        dead = sub == "D"
        for v, (keep, drop) in VARIANTS.items():
            p = fit_predict(train_set(keep, drop, range(len(raws))), (feats(ours, drop), g), seeds=SEEDS)
            for c in (4.0, G.SWITCH, 20.0):
                codes = np.where(g, decode(p, g & ~dead, c), pm.X).astype(np.int8)
                n = len(pm.possession_spells(codes, pm.valid_runs(codes))) * 90 * 60 * 25 / g.sum()
                rows.append({"variant": v, "cost": c, "possessions_90": n, "mean_conf": float(np.mean(np.abs(p[g & ~dead] - 0.5)))})
                print(rows[-1], flush=True)
        df = pd.DataFrame(rows)
        lines = ["# P9u (bizim maç): varyantların possession sayısı, GT'siz", "", "| varyant | ceza | 90 dk'da possession | ort. |p − 0.5| |", "|---|---|---|---|"]
        lines += ["| {} | {:g} | {:.0f} | {:.3f} |".format(r.variant, r.cost, r.possessions_90, r.mean_conf) for _, r in df.iterrows()]
    else:
        for t in range(len(raws)):
            name = deg[False][t]["name"]
            if ONLY and not any(o in name for o in ONLY):
                continue
            ks = [k for k in range(len(raws)) if k != t]
            r = {"match": name}
            for v, (keep, drop) in VARIANTS.items():
                tr = train_set(keep, drop, ks)
                allF = np.concatenate([d[0] for d in tr])
                mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
                nets = [G.train([(mod.norm(d[0]),) + tuple(d[1:]) for d in tr], s, "cuda") for s in SEEDS]
                for test_keep in (False, True):
                    m = deg[test_keep][t]
                    g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
                    X = mod.norm(feats(m, drop))
                    p = np.mean([G._predict_one(n, X, g, "cuda") for n in nets], axis=0)
                    r["{} on {}".format(v, "kept" if test_keep else "usual")] = pm.spell_scores(
                        m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
            rows.append(r)
            print(r, flush=True)
        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c != "match"]
        lines = ["# P9u: oyuncu sayısı kayması (GRU, sadece takım, tohum 3–5)", "", "`analysis/p9u_player_count_shift.py`.", "",
                 "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
        for src in ("DFL", "metrica"):
            q = df[df.match.str.startswith(src)]
            if len(q):
                lines.append("| ort. {} | ".format(src) + " | ".join("{:.3f}".format(q[c].mean()) for c in cols) + " |")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=") and a != "final"])
