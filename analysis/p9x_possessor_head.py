"""Possessor auxiliary head on the player-set member (research report "Yapısal sıçrama adayları", item 5): besides the
team logit, SetNet gives one logit per player slot, trained to pick the labelled team's player nearest the TRUE ball
(within mtah.setwho.POSS_M, all zero in flight), so the net has to learn from the players where the ball is and who
holds it. Player-set member alone, with and without the head (slot weight 1 / 2K on alive frames, so the head weighs as
much as the team target), DFL leave one match out and Metrica 1-3, team call only, flat switch, fresh seeds (3, 4, 5);
also scored against labels merged at 3.0 s (the baseline-calibrated referee). GPU.

    python analysis/p9x_possessor_head.py [only=<match>,...]      # -> eda_out/reports/P9x_possessor_head.md
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
from mtah.match import swap_teams  # noqa: E402
from mtah.setgru import K, NFP, player_set, set_net  # noqa: E402
from mtah.setwho import possessor_target, set_net_who  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


SEEDS = (3, 4, 5)
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("set", "set+who")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9x_possessor_head{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def fit_predict(train, test, make_net):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net) for s in SEEDS]
    p = np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)
    return p[:, 0] if p.ndim == 2 else p


def inputs(m):
    return np.concatenate([features(m).astype(np.float32), player_set(m)], axis=1)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    data = []
    for m, (r, _) in zip(base, raws):
        sw = swap_teams(m)
        data.append([(mm, inputs(mm), possessor_target(mm, r["bx"], r["by"])) for mm in (m, sw)])
        y = data[-1][0][2]
        alive = np.isin(m["labels"], (pm.H, pm.A))
        print(m["name"], "possessor found on {:.0%} of alive frames".format(y[alive].any(axis=1).mean()), flush=True)
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    nets = {"set": lambda nf: set_net(nf - 2 * K * NFP), "set+who": lambda nf: set_net_who(nf - 2 * K * NFP)}
    rows = []
    names = [m["name"] for m in base]
    for t in range(len(raws)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
        for v in VARIANTS:
            tr = []
            for k, pair in enumerate(data):
                if k == t:
                    continue
                for mm, FF, Y in pair:
                    w = aw(mm) * np.where(G.short_mask(mm["labels"]), 2.0, 1.0).astype(np.float32)
                    y = (mm["labels"] == pm.H).astype(np.float32)
                    if v == "set+who":
                        y = np.c_[y, Y].astype(np.float32)
                        w = np.c_[w, np.repeat(aw(mm)[:, None] / (2 * K), 2 * K, axis=1)].astype(np.float32)
                    tr.append((FF, y, w, mm["half"] > 0))
            m, F, _ = data[t][0]
            g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
            p = fit_predict(tr, (F, g), nets[v])
            pred = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9x: oyuncu kümesi üyesine possessor başı (sadece takım, tohum 3–5)", "", "`analysis/p9x_possessor_head.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for sfx in ("", " @3s"):
                a, b = q["set+who" + sfx], q["set" + sfx]
                lines.append("{}{}: set+who − set {:+.3f}, {} / {}".format(src, sfx, (a - b).mean(), int((a > b).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
