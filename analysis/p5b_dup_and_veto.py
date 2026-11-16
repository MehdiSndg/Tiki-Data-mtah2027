"""Open items 8 and 9: do double-counted players and the ATD's drifting ball change the team call?

    uv run python analysis/p5b_dup_and_veto.py      # -> eda_out/reports/P5b_dup_and_veto.md

Part A, double counting, external data (spellF1, dead ball from the labels so only the team call
is judged): every DFL and Metrica match degraded twice with the same seed, once as in P2 and once
with dup_share = 0 (the same random stream: only the duplicate copies differ).
Part B, double counting, our match (GT-free): on frames where a team shows more than 10 outfield
slots, a slot within DUP_M of a same-team slot with a lower index is dropped; labels before and
after, agreement with the Smart Tagging side outside the public window.
Part C, the drifting ATD ball, our match (GT-free): where the self-trained detector
(cache/ball_v1_pass) has a confident on-pitch box (conf >= VETO_CONF, not a painted spot) more
than VETO_M from the ATD ball, backed by its own boxes on >= 3 of the +/- 3 neighbouring frames,
the ATD ball is replaced by the detector's. Same comparisons as part B.
The dead ball comes from the raw-ATD classifier (p4_common.ours) throughout; the public window is
only a check.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.video_match import detector_ball  # noqa: E402
from p4_common import ours  # noqa: E402

FPS = 25
DUP_M = 1.5
VETO_CONF, VETO_M, NEIGH, NEIGH_M = 0.5, 5.0, 3, 2.0
AERIAL_DX, AERIAL_UP = 60, 400      # px: box within 60 px sideways and up to 400 px above the ATD point
PUBLIC = (11185, 18685)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5b_dup_and_veto.md")


def spell_f1(truth, pred):
    return pm.spell_scores(truth, pred, pm.valid_runs(truth))["mean_f1"]


def dedupe(m):
    L, W = m["pitch"]
    px, py = m["px"].copy(), m["py"].copy()
    gk = set(GK_SLOTS.values())
    for t in (0, 1):
        slots = [k for k in np.flatnonzero(m["slot_team"] == t) if k not in gk]
        X, Y = px[:, slots] * L, py[:, slots] * W
        crowded = np.flatnonzero((~np.isnan(X)).sum(1) > 10)
        for f in crowded:
            for j in range(len(slots)):
                if np.isnan(X[f, j]):
                    continue
                d = np.hypot(X[f, :j] - X[f, j], Y[f, :j] - Y[f, j])
                if np.any(d < DUP_M):
                    X[f, j] = Y[f, j] = np.nan
        px[:, slots], py[:, slots] = X / L, Y / W
    return dict(m, px=px, py=py)


def veto(m, hom):
    xy, conf, spot = detector_ball(hom, len(m["bx"]), os.path.join(ROOT, "cache", "ball_v1_pass"), 0, conf_min=0.1)
    L, W = PITCH
    has = ~np.isnan(xy[:, 0])
    d = np.hypot((xy[:, 0] - m["bx"]) * L, (xy[:, 1] - m["by"]) * W)
    cand = np.flatnonzero(has & ~spot & (conf >= VETO_CONF) & ~np.isnan(m["bx"]) & (d > VETO_M))
    # an aerial ball: in the image the box sits right above the ATD's ground point, so its ground
    # projection lands far away while the ATD is right (EDA A-5, overnight 22 Oct: 61% of mismatches)
    def px(f, x, y):
        h = hom[f]; q = h @ np.array([x, y, 1.0]); return q[:2] / q[2]
    aerial = np.array([(lambda a, b: abs(b[0] - a[0]) < AERIAL_DX and -AERIAL_UP < b[1] - a[1] < 0)(
        px(f, m["bx"][f], m["by"][f]), px(f, xy[f, 0], xy[f, 1])) for f in cand], bool)
    n_aerial = int(aerial.sum())
    cand = cand[~aerial]
    keep = []
    for f in cand:
        lo, hi = max(0, f - NEIGH), min(len(has), f + NEIGH + 1)
        nb = [g for g in range(lo, hi) if g != f and has[g]]
        close = sum(np.hypot((xy[g, 0] - xy[f, 0]) * L, (xy[g, 1] - xy[f, 1]) * W) < NEIGH_M * (1 + abs(g - f)) for g in nb)
        if close >= 3:
            keep.append(f)
    keep = np.array(keep, int)
    bx, by = m["bx"].copy(), m["by"].copy()
    bx[keep], by[keep] = xy[keep, 0], xy[keep, 1]
    return dict(m, bx=bx, by=by), keep, int((~np.isnan(m["bx"]) & (m["half"] > 0)).sum()), n_aerial


def compare(name, base, other, alive, sel, tag, gt):
    changed = np.mean(base[alive] != other[alive])
    n = lambda c: len(pm.possession_spells(c, pm.valid_runs(np.where(alive | (c == pm.D), 0, pm.X))))  # noqa: E731
    r = pm.evaluate(gt, other)
    return "| {} | {:.2%} | {} | {:.1%} | {:.3f} |".format(name, changed, n(other), np.mean(other[sel] == tag[sel]), r["spell_f1_mean"])


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    lines = ["# P5b: çift sayım ve ATD'nin kayan topu takım kararını değiştiriyor mu?", "",
             "`analysis/p5b_dup_and_veto.py`. Dış veride ölü top etiketten; bizim maçta ölü top ham ATD sınıflandırıcısından.", "",
             "## A. Çift sayım, dış veri (spellF1, takım kararı)", "",
             "| kaynak | çift sayımlı (P2 bozma) | çift sayımsız (aynı tohum) | fark |", "|---|---|---|---|"]
    for name, ms, keys in (("DFL (7)", [from_dfl(i) for i in ids], ids), ("Metrica (2)", [from_metrica_sample(g) for g in (1, 2)], ["metrica", "metrica"])):
        a, b = [], []
        for m, k in zip(ms, keys):
            r = dict(rates_for(k))
            d1 = degrade(m, r, seed=SEED)
            r["dup_share"] = 0.0
            d0 = degrade(m, r, seed=SEED)
            a.append(spell_f1(d1["labels"], control(d1, dead="truth")))
            b.append(spell_f1(d0["labels"], control(d0, dead="truth")))
        lines.append("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(name, np.mean(a), np.mean(b), np.mean(b) - np.mean(a)))
        print(lines[-1], flush=True)
    dead, base = ours()
    v1 = from_tactical()
    g = v1["half"] > 0
    alive = g & ~dead
    df = load_frames()
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(g), bool)
    outside[PUBLIC[0]:PUBLIC[1]] = False
    sel = alive & outside & (tag >= 0)
    gt = v1["labels"]
    label = lambda m: np.where(g, control(m, dead=dead), pm.X).astype(np.int8)  # noqa: E731
    lines += ["", "## B ve C. Bizim maç (GT'siz; public sadece kontrol)", "",
              "| sürüm | değişen oyun içi kare | possession sayısı | Smart Tagging takım uyumu (public dışı) | public spellF1 (kontrol) |",
              "|---|---|---|---|---|", compare("v1 (şu anki)", base, base, alive, sel, tag, gt)]
    lines.append(compare("çift sayım ayıklandı", base, label(dedupe(v1)), alive, sel, tag, gt))
    print(lines[-1], flush=True)
    vm, kept, n_atd, n_aerial = veto(v1, load_tactical()["hom"])
    lines.append(compare("kayan top dedektörle düzeltildi", base, label(vm), alive, sel, tag, gt))
    print(lines[-1], flush=True)
    lines += ["", "Kayan top vetosu: ATD topunun olduğu {} puanlanan karenin {}'inde ({:.2%}) dedektör topu "
              "ATD'den > {:.0f} m uzakta, güveni ≥ {} ve komşu karelerde tutarlıydı. Ayrıca {} aday havadaki top "
              "diye ayıklandı (görüntüde kutu ATD noktasının hemen üstünde).".format(n_atd, len(kept), len(kept) / n_atd, VETO_M, VETO_CONF, n_aerial)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    main()
