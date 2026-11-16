"""P9aw's test (three degradation draws per training match instead of one, team call only, seeds 3-5) for another
member: MTAH_ARCH=tcn (default), gru (the production 128 x 2 GRU) or big; MTAH_DRAWS=a,b the two draw counts (1,3). P9aw: BigGRU +0.016 DFL; P7g: the 128 x 2 GRU did
not gain from a second draw. GPU.

    MTAH_ARCH=tcn python analysis/p9az_draws_member.py [only=<match>,...]      # -> eda_out/reports/P9az_draws_tcn.md
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
from mtah.teammodel import decode, features  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


SEEDS = (3, 4, 5)
NA, NB = (int(x) for x in os.environ.get("MTAH_DRAWS", "1,3").split(","))     # the two variants' draw counts
DRAWS = tuple(range(max(NA, NB)))
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("draw{}".format(NA), "draw{}".format(NB))
ARCH = os.environ.get("MTAH_ARCH", "tcn")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9az_draws_" + ARCH + ("" if (NA, NB) == (1, 3) else "_{}v{}".format(NA, NB)) + "{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def big_net(nf):
    return G.NETS[ARCH](nf)


def fit_predict(train, test, make_net=None, jitter=False):
    """Mean / std from running float64 sums instead of a concatenated copy of every training feature (WSL ran out
    of memory on 5 Nov); np.nanmean / np.nanstd up to rounding."""
    nf = train[0][0].shape[1]
    n, s1, s2 = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    for F, *_ in train:
        ok = ~np.isnan(F)
        n += ok.sum(0)
        s1 += np.where(ok, F, 0).sum(0, dtype=np.float64)
        s2 += np.where(ok, F.astype(np.float64) ** 2, 0).sum(0)
    mu = s1 / n
    mod = G.GRUEnsemble([], mu, np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net, jitter=jitter) for s in SEEDS]
    del data
    return np.mean([G._predict_one(nt, mod.norm(test[0]), test[1], "cuda") for nt in nets], axis=0)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    draws = {d: [degrade(r, dict(rates_for(k), smooth_frames=25), seed=d) for r, k in raws] for d in DRAWS}
    base = draws[0]
    featd = {d: [(m, features(m), features(swap_teams(m))) for m in draws[d]] for d in DRAWS}
    feats = featd[0]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    names = [m["name"] for m in base]
    for t in range(len(raws)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
        def rows_of(ds):
            out = []
            for d in ds:
                for k, (m, F, Fs) in enumerate(featd[d]):
                    if k == t:
                        continue
                    sw, sh = swap_teams(m), G.short_mask(m["labels"])
                    for mm, FF in ((m, F), (sw, Fs)):
                        out.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            return out
        m, F, _ = feats[t]
        g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
        for v, n in zip(VARIANTS, (NA, NB)):
            p = fit_predict(rows_of(tuple(range(n))), (F, g), make_net=big_net)
            pred = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9az: " + ARCH + " üyesine {} yerine {} bozma kopyası".format(NA, NB) + " (tohum 3–5, sadece takım)", "", "`analysis/p9az_draws_member.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for sfx in ("", " @3s"):
                a, b = VARIANTS
                lines.append("{}{}: {} − {} {:+.3f}, {} / {}".format(src, sfx, b, a, (q[b + sfx] - q[a + sfx]).mean(), int((q[b + sfx] > q[a + sfx]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
