"""Decode and dead-ball post-processing studied on P9a's out-of-fold probabilities (no retraining). CPU, parallel.

    python analysis/p9b_oof_sweeps.py <oof_dir>      # -> eda_out/reports/P9b_oof_sweeps.md

A. Team decode grid (dead ball = v11's): team source (v11 members, multi-task heads, both), boundary source (boundary
   GRU, multi-task heads, both), switch cost flat or bounded hi (1 - b) + lo b, probability clip eps. Each config's
   mean on DFL and on Metrica, and the nested choice (each DFL match decoded with the config best on the other six).
B. Segment voting (ASRF): cut each alive run at boundary peaks above a threshold, each piece takes its mean team.
C. Where the dead ball loses (team decode = v11): spurious blocks removed, missed blocks added, edges of matched
   blocks taken from the labels, each alone.
D. Dead-ball blocking: median window, hysteresis (enter / leave), and dropping a weak block that sits inside one team's
   possession.
"""

import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import MIN_BLOCK, _roll, to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P9b_oof_sweeps.md")
HI, LO = 12.0, 0.5                        # production


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "oof_*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        z["name"] = str(z["name"])
        out[z["name"]] = z
    return out


def team_src(z, k):
    v11, mt = (z["pg"] + z["pt"] + z["ps"]) / 3, (z["mg"] + z["mt"] + z["ms"]) / 3
    return {"v11": v11, "mt": mt, "both": (v11 + mt) / 2}[k]


def bnd_src(z, k):
    mb = (z["mbg"] + z["mbt"] + z["mbs"]) / 3
    return {"b": z["b"], "mb": mb, "both": (z["b"] + mb) / 2}[k]


def score(z, codes):
    g = z["g"]
    return pm.spell_scores(z["labels"], np.where(g, codes, pm.X).astype(np.int8), pm.valid_runs(z["labels"]))["mean_f1"]


def v11_dead(z):
    return to_blocks((z["qh"] + z["qg"]) / 2) & z["g"]


def cost_of(z, bk, hi, lo):
    return hi if bk is None else hi * (1 - bnd_src(z, bk)) + lo * bnd_src(z, bk)


def run_a(args):
    name, tk, bk, hi, lo, eps = args
    z = Z[name]
    p = np.clip(team_src(z, tk), eps, 1 - eps)
    dead = v11_dead(z)
    return dict(match=name, team=tk, bnd=bk or "flat", hi=hi, lo=lo, eps=eps, f1=score(z, decode(p, z["g"] & ~dead, cost_of(z, bk, hi, lo))))


def segment_vote(p, alive, b, thr):
    out = np.full(len(p), pm.D, np.int8)
    peak = (b >= thr) & (b >= np.r_[-1, b[:-1]]) & (b > np.r_[b[1:], -1])
    e = np.diff(np.r_[0, alive.astype(np.int8), 0])
    for a, z_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        cuts = [a] + [i for i in np.flatnonzero(peak[a:z_]) + a if i > a] + [z_]
        for s, t in zip(cuts[:-1], cuts[1:]):
            out[s:t] = pm.H if p[s:t].mean() >= 0.5 else pm.A
    return out


def run_b(args):
    name, tk, bk, thr = args
    z = Z[name]
    dead = v11_dead(z)
    return dict(match=name, team=tk, bnd=bk, thr=thr, f1=score(z, segment_vote(team_src(z, tk), z["g"] & ~dead, bnd_src(z, bk), thr)))


def blocks(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def run_c(name):
    z = Z[name]
    g, truth = z["g"], (z["labels"] == pm.D) & z["g"]
    pred = v11_dead(z)
    p, cost = team_src(z, "v11"), cost_of(z, "b", HI, LO)
    f = lambda dead: score(z, decode(p, g & ~dead, cost))
    no_spur, add_missed, edges = pred.copy(), pred.copy(), pred.copy()
    for a, b in blocks(pred):
        if not truth[a:b].any():
            no_spur[a:b] = False
    for a, b in blocks(truth):
        if not pred[a:b].any():
            add_missed[a:b] = True
    for a, b in blocks(pred):
        if truth[a:b].any():
            edges[a:b] = False
            for c, d in blocks(truth):
                if d > a and c < b:
                    edges[c:d] = True
    n_sp = sum(not truth[a:b].any() for a, b in blocks(pred))
    n_mi = sum(not pred[a:b].any() for a, b in blocks(truth))
    return dict(match=name, v11=f(pred), no_spurious=f(no_spur), add_missed=f(add_missed), true_edges=f(edges), truth=f(truth),
                n_pred=len(blocks(pred)), n_true=len(blocks(truth)), n_spurious=n_sp, n_missed=n_mi)


def hyst(q, med, enter, leave):
    r = _roll(q, med)
    dead = np.zeros(len(q), bool)
    on = False
    for i, v in enumerate(r):
        on = v > enter if not on else v > leave
        dead[i] = on
    e = np.diff(np.r_[0, (~dead).astype(np.int8), 0])
    for a, z_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z_ - a < MIN_BLOCK and a > 0 and z_ < len(dead):
            dead[a:z_] = True
    for a, z_ in blocks(dead):
        if z_ - a < MIN_BLOCK:
            dead[a:z_] = False
    return dead


def run_d(args):
    name, med, enter, leave, veto = args
    z = Z[name]
    g, q = z["g"], (z["qh"] + z["qg"]) / 2
    p, cost = team_src(z, "v11"), cost_of(z, "b", HI, LO)
    dead = hyst(q, med, enter, leave) & g
    codes = decode(p, g & ~dead, cost)
    if veto > 0:
        for a, b in blocks(dead):
            if a > 0 and b < len(codes) and codes[a - 1] == codes[b] and codes[b] in (pm.H, pm.A) and q[a:b].mean() < veto:
                dead[a:b] = False
        codes = decode(p, g & ~dead, cost)
    return dict(match=name, med=med, enter=enter, leave=leave, veto=veto, f1=score(z, codes))


def nested(df, keys):
    """Each DFL match scored with the config best on the other DFL matches (mean), plus that mean on Metrica."""
    d = df[df.match.str.startswith("DFL")]
    rows = []
    for m in sorted(d.match.unique()):
        best = d[d.match != m].groupby(keys).f1.mean().idxmax()
        sel = d[(d.match == m)].set_index(keys).f1.loc[best]
        rows.append((m, best, float(sel)))
    return rows


def summary(df, keys, title, top=8):
    dfl, met = df[df.match.str.startswith("DFL")], df[df.match.str.startswith("metrica")]
    t = pd.concat([dfl.groupby(keys).f1.mean().rename("DFL"), met.groupby(keys).f1.mean().rename("Metrica")], axis=1)
    t["both"] = (t.DFL + t.Metrica.fillna(t.DFL)) / 2
    lines = ["## " + title, "", "| " + " | ".join(keys) + " | DFL | Metrica |", "|" + "---|" * (len(keys) + 2)]
    for idx, r in t.sort_values("both", ascending=False).head(top).iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        lines.append("| " + " | ".join(map(str, idx)) + " | {:.3f} | {:.3f} |".format(r.DFL, r.Metrica))
    ne = nested(df, keys)
    lines += ["", "İç içe (DFL): " + ", ".join("{} {:.3f} {}".format(m.split("-")[-1][:6], s, b) for m, b, s in ne),
              "İç içe ortalama {:.3f}.".format(np.mean([s for _, _, s in ne])), ""]
    return lines


def main(d):
    global Z
    Z = load(d)
    names = sorted(Z)
    print("matches", names, flush=True)
    grid_a = [(n, tk, bk, hi, lo, eps) for n in names for tk in ("v11", "mt", "both") for eps in (1e-4, 1e-3, 1e-2)
              for bk, hi, lo in [(None, c, 0) for c in (2.0, 3.0, 4.0, 6.0, 12.0)] + [(bk, hi, lo) for bk in ("b", "mb", "both") for hi in (3.0, 4.0, 6.0, 8.0, 12.0) for lo in (0.5, 1.0, 2.0)]]
    grid_b = [(n, tk, bk, thr) for n in names for tk in ("v11", "mt", "both") for bk in ("b", "mb", "both") for thr in (0.2, 0.3, 0.5, 0.7)]
    grid_d = [(n, med, en, lv, veto) for n in names for med in (25, 50, 100) for en, lv in ((0.5, 0.5), (0.6, 0.4), (0.7, 0.5), (0.6, 0.5), (0.5, 0.3))
              for veto in (0.0, 0.6, 0.7)]
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        A = pd.DataFrame(pool.map(run_a, grid_a, chunksize=4))
        print("A done", flush=True)
        B = pd.DataFrame(pool.map(run_b, grid_b, chunksize=4))
        C = pd.DataFrame(pool.map(run_c, names))
        Dd = pd.DataFrame(pool.map(run_d, grid_d, chunksize=4))
    A.to_csv(OUT.replace(".md", "_A.csv"), index=False)
    lines = ["# P9b: OOF üzerinde kod çözme ve ölü top taramaları", "", "`analysis/p9b_oof_sweeps.py`.", ""]
    prod = A[(A.team == "v11") & (A.bnd == "b") & (A.hi == HI) & (A.lo == LO) & (A.eps == 1e-4)]
    lines += ["Üretim (v11, sınır GRU'su 12 / 0.5, eps 1e-4): DFL {:.3f}, Metrica {:.3f}.".format(
        prod[prod.match.str.startswith("DFL")].f1.mean(), prod[prod.match.str.startswith("metrica")].f1.mean()), ""]
    lines += summary(A, ["team", "bnd", "hi", "lo", "eps"], "A. Takım kod çözme ızgarası", top=12)
    lines += summary(B, ["team", "bnd", "thr"], "B. Segment oylaması")
    lines += ["## C. Ölü top kaybı nerede (takım v11, ceza üretim)", "", "| maç | v11 | sahte bloklar yok | kaçan bloklar eklendi | eşleşen bloklarda gerçek kenar | ölü top etiketten | tahmin / gerçek / sahte / kaçan blok |",
              "|---|---|---|---|---|---|---|"]
    for _, r in C.iterrows():
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} / {} / {} / {} |".format(r.match, r.v11, r.no_spurious, r.add_missed, r.true_edges, r.truth,
                                                                                              r.n_pred, r.n_true, r.n_spurious, r.n_missed))
    for part in ("DFL", "metrica"):
        c = C[C.match.str.startswith(part)]
        lines.append("| ort. {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | |".format(part, c.v11.mean(), c.no_spurious.mean(), c.add_missed.mean(), c.true_edges.mean(), c.truth.mean()))
    lines += [""] + summary(Dd, ["med", "enter", "leave", "veto"], "D. Ölü top bloklama")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
