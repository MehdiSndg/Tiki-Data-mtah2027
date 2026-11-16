"""Open item 10: second, independent checks for the findings that rested on a single source.

    uv run python analysis/p5f_verification.py      # -> eda_out/reports/P5f_verification.md

A. P3 said: where the ATD loses the ball, giving the detector's ball to the nearest ATD player is
   worse than holding the last team (judged only by Smart Tagging agreement). The same experiment
   with real labels: degraded external data, the ball put back from the clean data in the alive
   frames the degradation emptied (a perfect detector), players left degraded; and with the clean
   players put back too, to test "the players are the cause". Dead ball from the labels.
B. "The ATD ball is > 5 m off on ~2-3% of its frames" came from one tool (off-the-shelf YOLO, 1000
   frames); P5b's ball_v1 veto gave 1.4%. A third source: the broadcast ATD on the same frames
   (another camera, synchronised by analysis/h_broadcast_sync.py), alone and with ball_v1.
C. The degradation rates were each measured once over the whole match. (1) The same measures per
   half and per quarter of our match, GT-free. (2) Do the conclusions survive other rates? Every
   rate scaled by 0.5 and 1.5, and the provider's smoothing on: the control / touch / control+touch
   ranking on degraded external data. Also the per-match breakdown of the touch gain (DFL is its
   only honest source).
D. Metrica's "truth" is our derivation and DFL's merge (1.6 s) our choice. The labeler ranking
   against other derivations: merge 0 / 1.0 / 3.0 s, and every dead-ball block widened by 1 s.
A2. How good must the detector be for A's gain to survive? The ball is put back only on a share of
   the emptied frames (1 s blocks) and with a constant offset per block (an aerial ball projected to
   the ground, a box centre off the ball).
B2. The drifting / offset ATD ball, external data: degraded with drift and offsets off, as in P2,
   and doubled; the spellF1 difference bounds what fixing it can bring.
A3. P3's explanation was "the players near the lost ball are missing too". The ATD loses ball and
   players together in crowds; the degradation loses them independently. Here the degraded players
   within NEAR_M of the true ball are also removed on the emptied frames (all, or half of the 1 s
   blocks), then A is repeated.
B2 is repeated with two more seeds to see its noise.
A4. A GT-free test of the detector fill on our match, with no Smart Tagging: artificial gaps (5 s
   blocks, 10% of the alive frames where the ATD has the ball) are cut into the ATD ball and filled
   with ball_v1's box; the reference is the control rule's call with the ATD ball itself. Caveat:
   these gaps fall where the ATD sees the ball, the easier moments.
A5. The real gaps of our match, judged by Smart Tagging as P3 was: filled with the detector, with the
   broadcast ATD, and only where the two agree within 3 m (a ball almost surely right). If Smart
   Tagging still prefers holding there, the judge is suspect rather than the ball. Last row: the
   ghost-keeper hypothesis (a filled ball handed to a frozen keeper slot) -- far keeper slots dropped.
A6. The ATD loses the ball in crowds and duels; the degradation loses it anywhere. External data,
   real labels: artificial gaps cut around contested moments (both teams within CONTEST_M of the
   ball) against the same number of random gap frames, each filled with the clean ball.
The public window is not read anywhere.
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.broadcast import broadcast_ball_on_tactical  # noqa: E402
from mtah.data import GK_SLOTS, PITCH  # noqa: E402
from mtah.degrade import DEFAULT_RATES, degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample, merge_short  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.video_match import _hom, detector_ball  # noqa: E402
from p4_common import ours  # noqa: E402
from p4f_touch_labeler import SEED, control_touch, touch_only  # noqa: E402

FPS = 25
NEAR_M = 3.0
CONTEST_M = 2.0
SCALED = ("loss_per_min", "piece_drop", "drift_p", "jitter_share", "dup_share", "gk_freeze_share")
SEED = int(os.environ.get("MTAH_SEED", SEED))      # other degradation seeds: MTAH_SEED=1, 2 (analysis/seed_combine.py)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5f_verification{}.md".format("" if SEED == 0 else "_seed{}".format(SEED)))


def spell_f1(truth, pred):
    return pm.spell_scores(truth, pred, pm.valid_runs(truth))["mean_f1"]


def runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def external():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    return [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]


def src_of(name):
    return "Metrica" if name.startswith("metrica") else "DFL"


def labelers():
    return {"control": lambda m: control(m, dead="truth"), "touch": touch_only, "control+touch": control_touch}


# ---------------------------------------------------------------- A
def part_a(ext):
    rows = []
    for clean, key in ext:
        deg = degrade(clean, rates_for(key), seed=SEED)
        alive = np.isin(clean["labels"], (pm.H, pm.A))
        gap = alive & np.isnan(deg["bx"]) & ~np.isnan(clean["bx"])
        ball = dict(deg, bx=np.where(gap, clean["bx"], deg["bx"]), by=np.where(gap, clean["by"], deg["by"]))
        both = dict(ball, px=np.concatenate([np.where(gap[:, None], np.nan, deg["px"]),
                                             np.where(gap[:, None], clean["px"], np.nan)], axis=1),
                    py=np.concatenate([np.where(gap[:, None], np.nan, deg["py"]),
                                       np.where(gap[:, None], clean["py"], np.nan)], axis=1),
                    slot_team=np.r_[deg["slot_team"], clean["slot_team"]])
        r = {"src": src_of(clean["name"]), "gap": gap.sum() / alive.sum()}
        for name, m in (("hold", deg), ("ball", ball), ("ball+players", both)):
            p = control(m, dead="truth")
            r[name] = spell_f1(clean["labels"], p)
            r[name + "_acc"] = np.mean(p[gap] == clean["labels"][gap])
        rows.append(r)
        print("A", clean["name"], {k: round(v, 3) for k, v in r.items() if k != "src"}, flush=True)
    t = pd.DataFrame(rows).groupby("src").mean()
    lines = ["## A. P3 kararının dış veride karşılığı: kaybolan topu doldurmak", "",
             "Bozulmuş veride oyun içinde topun silindiği karelere temiz verinin topu konuyor (kusursuz dedektör). "
             "Kontrol kuralı, ölü top etiketten.", "",
             "| kaynak | oyun içi karelerde boşluk | son takımı tut | topu doldur (oyuncular bozuk) | topu + oyuncuları doldur | "
             "boşluk karelerinde takım doğruluğu: tut / top / top+oyuncu |", "|---|---|---|---|---|---|"]
    for s, r in t.iterrows():
        lines.append("| {} | {:.1%} | {:.3f} | {:.3f} | {:.3f} | {:.1%} / {:.1%} / {:.1%} |".format(
            s, r["gap"], r["hold"], r["ball"], r["ball+players"], r["hold_acc"], r["ball_acc"], r["ball+players_acc"]))
    return lines


def part_a2(ext):
    rows = []
    for clean, key in ext:
        deg = degrade(clean, rates_for(key), seed=SEED)
        alive = np.isin(clean["labels"], (pm.H, pm.A))
        gap = alive & np.isnan(deg["bx"]) & ~np.isnan(clean["bx"])
        L, W = clean["pitch"]
        for cover in (1.0, 0.5):
            for off_m in (0, 2, 5):
                rng = np.random.RandomState(SEED)
                n = len(gap)
                blocks = np.arange(n) // FPS
                keep = rng.uniform(size=blocks[-1] + 1) < cover
                ang = rng.uniform(0, 2 * np.pi, blocks[-1] + 1)
                r = np.abs(rng.normal(0, off_m, blocks[-1] + 1)) if off_m else np.zeros(blocks[-1] + 1)
                fill = gap & keep[blocks]
                bx = np.where(fill, clean["bx"] + r[blocks] * np.cos(ang[blocks]) / L, deg["bx"])
                by = np.where(fill, clean["by"] + r[blocks] * np.sin(ang[blocks]) / W, deg["by"])
                p = control(dict(deg, bx=bx.astype(np.float32), by=by.astype(np.float32)), dead="truth")
                rows.append({"src": src_of(clean["name"]), "cover": cover, "off": off_m,
                             "f1": spell_f1(clean["labels"], p)})
        rows.append({"src": src_of(clean["name"]), "cover": 0.0, "off": 0,
                     "f1": spell_f1(clean["labels"], control(deg, dead="truth"))})
    t = pd.DataFrame(rows).groupby(["src", "cover", "off"]).f1.mean().unstack(0)
    lines = ["", "## A2. Kusurlu dedektörle doldurmak: kazanç ne zaman kayboluyor?", "",
             "Boşluk karelerinin bir payı 1 s'lik bloklar halinde dolduruluyor; her bloğa sabit bir kayma "
             "(|N(0, σ)| m, rastgele yön). Kapsam 0 = son takımı tut.", "",
             "| kapsam | kayma σ m | DFL | Metrica |", "|---|---|---|---|"]
    for (c, o), r in t.iterrows():
        lines.append("| {:.0%} | {} | {:.3f} | {:.3f} |".format(c, o, r["DFL"], r["Metrica"]))
    return lines


def part_b2(ext):
    variants = {"kayma ve sapma yok": 0.0, "P2 (drift 2.5%, sapma 6%)": 1.0, "iki katı": 2.0}
    rows = []
    for clean, key in ext:
        for name, f in variants.items():
            r = dict(rates_for(key))
            r["drift_p"], r["jitter_share"] = r["drift_p"] * f, r["jitter_share"] * f
            deg = degrade(clean, r, seed=SEED)
            rows.append({"src": src_of(clean["name"]), "v": name, "f1": spell_f1(clean["labels"], control(deg, dead="truth"))})
    t = pd.DataFrame(rows).groupby(["v", "src"], sort=False).f1.mean().unstack()
    lines = ["", "## B2. ATD'nin kayan topu dış veride: düzeltmenin tavanı", "",
             "Bozma aynı, sadece kayma (drift_p) ve 1–5 m sapma (jitter_share) değişiyor. Kontrol kuralı, ölü top etiketten.", "",
             "| bozma | DFL | Metrica |", "|---|---|---|"]
    for v, r in t.iterrows():
        lines.append("| {} | {:.3f} | {:.3f} |".format(v, r["DFL"], r["Metrica"]))
    return lines


def part_a3(ext):
    rows = []
    for clean, key in ext:
        deg = degrade(clean, rates_for(key), seed=SEED)
        alive = np.isin(clean["labels"], (pm.H, pm.A))
        gap = alive & np.isnan(deg["bx"]) & ~np.isnan(clean["bx"])
        L, W = clean["pitch"]
        near = np.hypot((deg["px"] - clean["bx"][:, None]) * L, (deg["py"] - clean["by"][:, None]) * W) <= NEAR_M
        blocks = np.arange(len(gap)) // FPS
        for q in (0.0, 0.5, 1.0):
            hit = gap & (np.random.RandomState(SEED).uniform(size=blocks[-1] + 1) < q)[blocks]
            gone = near & hit[:, None]
            m = dict(deg, px=np.where(gone, np.nan, deg["px"]), py=np.where(gone, np.nan, deg["py"]))
            fill = dict(m, bx=np.where(gap, clean["bx"], m["bx"]), by=np.where(gap, clean["by"], m["by"]))
            rows.append({"src": src_of(clean["name"]), "q": q, "hold": spell_f1(clean["labels"], control(m, dead="truth")),
                         "fill": spell_f1(clean["labels"], control(fill, dead="truth"))})
    t = pd.DataFrame(rows).groupby(["src", "q"])[["hold", "fill"]].mean()
    lines = ["", "## A3. Top kaybolurken yanındaki oyuncular da kaybolursa doldurmak zarar verir mi?", "",
             "Boşluk karelerinde gerçek topun {:.0f} m yakınındaki bozulmuş oyuncular da siliniyor (1 s'lik blokların payı).".format(NEAR_M), "",
             "| kaynak | silinen blok payı | son takımı tut | topu doldur (kusursuz) |", "|---|---|---|---|"]
    for (s_, q), r in t.iterrows():
        lines.append("| {} | {:.0%} | {:.3f} | {:.3f} |".format(s_, q, r["hold"], r["fill"]))
    rows = []
    for clean, key in ext:
        for seed in (1, 2):
            for name, f in (("kayma ve sapma yok", 0.0), ("P2", 1.0), ("iki katı", 2.0)):
                r = dict(rates_for(key))
                r["drift_p"], r["jitter_share"] = r["drift_p"] * f, r["jitter_share"] * f
                deg = degrade(clean, r, seed=seed)
                rows.append({"src": src_of(clean["name"]), "seed": seed, "v": name,
                             "f1": spell_f1(clean["labels"], control(deg, dead="truth"))})
    t = pd.DataFrame(rows).groupby(["seed", "v", "src"], sort=False).f1.mean().unstack()
    lines += ["", "### B2, başka tohumlarla", "", "| tohum | bozma | DFL | Metrica |", "|---|---|---|---|"]
    for (seed, v), r in t.iterrows():
        lines.append("| {} | {} | {:.3f} | {:.3f} |".format(seed, v, r["DFL"], r["Metrica"]))
    return lines


def part_a4():
    m = from_tactical()
    dead, _ = ours()
    period = load_frames()["period"].values
    alive = (m["half"] > 0) & ~np.isin(period, NOT_PLAYED) & ~dead
    xy, conf, spot = detector_ball(_hom(), len(m["bx"]), os.path.join(ROOT, "cache", "ball_v1_pass"), 0, conf_min=0.3)
    ref = control(m, dead=~alive)
    blocks = np.arange(len(alive)) // (5 * FPS)
    lines = ["", "## A4. Bizim maçta GT'siz deneme: yapay boşluğu dedektörle doldurmak", "",
             "Referans: ATD topuyla kontrol kuralı. Yapay boşluk: ATD topunun olduğu oyun içi karelerde 5 s'lik blokların %10'u. "
             "3 tohum ortalaması.", "",
             "| boşlukta | dedektörün kapsadığı boşluk payı | boşluk karelerinde referansla uyum | referansa göre spellF1 |", "|---|---|---|---|"]
    res = {"son takımı tut": [], "dedektör topu (güven ≥ 0.3)": [], "dedektör topu (güven ≥ 0.5)": []}
    cov = {k: [] for k in res}
    for seed in (0, 1, 2):
        cut = (np.random.RandomState(seed).uniform(size=blocks[-1] + 1) < 0.1)[blocks] & alive & ~np.isnan(m["bx"])
        hold = dict(m, bx=np.where(cut, np.nan, m["bx"]).astype(np.float32), by=np.where(cut, np.nan, m["by"]).astype(np.float32))
        truth = np.where(alive, ref, np.where(m["half"] > 0, pm.D, pm.X)).astype(np.int8)
        for name in res:
            c = 0.5 if "0.5" in name else 0.3
            use = cut & ~np.isnan(xy[:, 0]) & ~spot & (conf >= c) if "dedektör" in name else np.zeros(len(cut), bool)
            mm = dict(hold, bx=np.where(use, xy[:, 0], hold["bx"]).astype(np.float32),
                      by=np.where(use, xy[:, 1], hold["by"]).astype(np.float32))
            p = control(mm, dead=~alive)
            res[name].append((np.mean(p[cut] == ref[cut]), spell_f1(truth, np.where(m["half"] > 0, p, pm.X))))
            cov[name].append(use.sum() / cut.sum())
    for name, v in res.items():
        v = np.mean(v, axis=0)
        lines.append("| {} | {:.0%} | {:.1%} | {:.3f} |".format(name, np.mean(cov[name]), v[0], v[1]))
    return lines


def part_a5():
    m = from_tactical()
    dead, _ = ours()
    df = load_frames()
    period = df["period"].values
    alive = (m["half"] > 0) & ~np.isin(period, NOT_PLAYED) & ~dead
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(alive), bool)
    outside[11185:18685] = False
    xy, conf, spot = detector_ball(_hom(), len(m["bx"]), os.path.join(ROOT, "cache", "ball_v1_pass"), 0, conf_min=0.3)
    bbx, bby = broadcast_ball_on_tactical()
    L, W = PITCH
    gap = alive & np.isnan(m["bx"])
    det = gap & ~np.isnan(xy[:, 0]) & ~spot
    bro = gap & ~np.isnan(bbx)
    agree = det & bro & (np.hypot((xy[:, 0] - bbx) * L, (xy[:, 1] - bby) * W) <= 3)
    ref = control(m, dead=~alive)
    judged = alive & outside & (tag >= 0)
    lines = ["", "## A5. Bizim maçın gerçek boşlukları, Smart Tagging hakemliğinde (P3'ün ölçüsü)", "",
             "Oyun içinde ATD topunun olmadığı kareler: {} ({:.1%}).".format(gap.sum(), gap.sum() / alive.sum()), "",
             "| doldurma | doldurulan kare | kararı değişen kare | değişen karelerde Smart Tagging uyumu: tut → doldur | "
             "doldurulan topa en yakın ATD oyuncusu medyan m | genel uyum | değişen aralıklar: medyan s, < 2 s payı |",
             "|---|---|---|---|---|---|---|"]
    lines.append("| son takımı tut | 0 | 0 | — | — | {:.1%} | — |".format(np.mean(ref[judged] == tag[judged])))
    for name, use, X, Y in (("dedektör (güven ≥ 0.3)", det, xy[:, 0], xy[:, 1]), ("yayın ATD'si", bro, bbx, bby),
                            ("ikisi ≤ 3 m uyumlu (dedektörün konumu)", agree, xy[:, 0], xy[:, 1]),
                            ("dedektör, uzaktaki kaleci izi yok sayılarak", det, xy[:, 0], xy[:, 1])):
        mm = dict(m, bx=np.where(use, X, m["bx"]).astype(np.float32), by=np.where(use, Y, m["by"]).astype(np.float32))
        if "kaleci" in name:            # a ghost keeper: believed only within 25 m of the ball, near a goal line
            px, py = m["px"].copy(), m["py"].copy()
            bxm = mm["bx"] * L
            for k in GK_SLOTS.values():
                far = use & ~((np.minimum(bxm, L - bxm) < 25) & (np.abs(bxm - m["px"][:, k] * L) < 25))
                px[far, k] = py[far, k] = np.nan
            mm = dict(mm, px=px, py=py)
        p = control(mm, dead=~alive)
        ch = (p != ref) & judged
        d = np.sqrt(np.nanmin((m["px"][use] - X[use][:, None]) ** 2 * L * L + (m["py"][use] - Y[use][:, None]) ** 2 * W * W, axis=1))
        lens = np.array([(b - a) / FPS for a, b in runs(p != ref)])
        lines.append("| {} | {} | {} | {:.1%} → {:.1%} | {:.1f} | {:.1%} | {:.1f}, {:.0%} |".format(
            name, use.sum(), ch.sum(), np.mean(ref[ch] == tag[ch]), np.mean(p[ch] == tag[ch]), np.nanmedian(d),
            np.mean(p[judged] == tag[judged]), np.median(lens), np.mean(lens < 2)))
    side = np.where(tag >= 0, tag, -9)
    tl = np.array([(b - a) / FPS for t in (pm.H, pm.A) for a, b in runs((side == t) & outside & (m["half"] > 0))])
    lines += ["", "Smart Tagging'in kodları taktik evreler (BUILD UP, PROGRESSION, ...): bir takımın kesintisiz "
              "aralıklarının medyanı {:.1f} s, {:.0%}'i 2 s'den kısa (public dışı). Kısa possession'ları görmeyen bir hakem.".format(
                  np.median(tl), np.mean(tl < 2))]
    return lines


def part_a6(ext):
    from mtah.labelers import _team_distances
    rows = []
    for clean, key in ext:
        deg = degrade(clean, rates_for(key), seed=SEED)
        alive = np.isin(clean["labels"], (pm.H, pm.A))
        dH, dA = _team_distances(clean)
        contested = alive & (dH <= CONTEST_M) & (dA <= CONTEST_M)
        near = np.convolve(contested.astype(float), np.ones(2 * FPS + 1), "same") > 0
        has = alive & ~np.isnan(deg["bx"]) & ~np.isnan(clean["bx"])
        blocks = np.arange(len(alive)) // FPS
        rng = np.random.RandomState(SEED)
        cut_c = has & near & (rng.uniform(size=blocks[-1] + 1) < 0.5)[blocks]
        pool = np.flatnonzero(has)
        cut_r = np.zeros(len(alive), bool)
        starts = rng.choice(pool, size=int(cut_c.sum() / FPS), replace=False)
        for s0 in starts:
            cut_r[s0:s0 + FPS] = True
        cut_r &= has
        r = {"src": src_of(clean["name"]), "contested_share": contested.sum() / alive.sum()}
        for name, cut in (("çekişmeli", cut_c), ("rastgele", cut_r)):
            hold = dict(deg, bx=np.where(cut, np.nan, deg["bx"]).astype(np.float32), by=np.where(cut, np.nan, deg["by"]).astype(np.float32))
            fill = dict(hold, bx=np.where(cut, clean["bx"], hold["bx"]).astype(np.float32), by=np.where(cut, clean["by"], hold["by"]).astype(np.float32))
            ph, pf = control(hold, dead="truth"), control(fill, dead="truth")
            r[name + "_n"] = cut.sum() / alive.sum()
            r[name + "_hold"], r[name + "_fill"] = spell_f1(clean["labels"], ph), spell_f1(clean["labels"], pf)
            r[name + "_acc_hold"], r[name + "_acc_fill"] = np.mean(ph[cut] == clean["labels"][cut]), np.mean(pf[cut] == clean["labels"][cut])
        rows.append(r)
        print("A6", clean["name"], {k: round(v, 3) for k, v in r.items() if k != "src"}, flush=True)
    t = pd.DataFrame(rows).groupby("src").mean()
    lines = ["", "## A6. Çekişmeli anlarda kesilen boşluk: doldurmak orada da işe yarıyor mu? (dış veri, gerçek etiket)", "",
             "Çekişmeli: iki takımdan da bir oyuncu topa ≤ {:.0f} m. Boşluklar bu anların ±1 s çevresinde, ya da aynı miktarda "
             "rastgele yerde. Kontrol kuralı, ölü top etiketten.".format(CONTEST_M), "",
             "| kaynak | boşluk yeri | oyun içi payı | spellF1 tut → doldur | boşluk karelerinde doğruluk tut → doldur |", "|---|---|---|---|---|"]
    for s_, r in t.iterrows():
        for name in ("çekişmeli", "rastgele"):
            lines.append("| {} | {} | {:.1%} | {:.3f} → {:.3f} | {:.1%} → {:.1%} |".format(
                s_, name, r[name + "_n"], r[name + "_hold"], r[name + "_fill"], r[name + "_acc_hold"], r[name + "_acc_fill"]))
    return lines


# ---------------------------------------------------------------- B, C1
def our_measures():
    m = from_tactical()
    period = load_frames()["period"].values
    dead, _ = ours()
    played = (m["half"] > 0) & ~np.isin(period, NOT_PLAYED)
    tbx, tby = m["bx"], m["by"]
    bbx, bby = broadcast_ball_on_tactical()
    xy, conf, spot = detector_ball(_hom(), len(tbx), os.path.join(ROOT, "cache", "ball_v1_pass"), 0, conf_min=0.5)
    L, W = PITCH
    d_tb = np.hypot((tbx - bbx) * L, (tby - bby) * W)
    d_td = np.hypot((tbx - xy[:, 0]) * L, (tby - xy[:, 1]) * W)
    d_bd = np.hypot((bbx - xy[:, 0]) * L, (bby - xy[:, 1]) * W)
    has_t = ~np.isnan(tbx) & played
    both = has_t & ~np.isnan(bbx)
    det = ~np.isnan(xy[:, 0]) & ~spot
    three = both & det

    lines = ["## B. ATD topunun yanlış yerde olması: üçüncü kaynak (yayın ATD'si)", "",
             "Bizim maç, oynanan puanlanan kareler (yarı sonu ve 2. yarı öncesi hariç). Mesafe metre.", "",
             "| karşılaştırma | kare | ≤ 1 m | 1–5 m | > 5 m |", "|---|---|---|---|---|"]
    for name, mask, d in (("taktik ATD – yayın ATD", both, d_tb), ("taktik ATD – ball_v1 (güven ≥ 0.5)", has_t & det, d_td),
                          ("yayın ATD – ball_v1 (güven ≥ 0.5)", ~np.isnan(bbx) & played & det, d_bd)):
        x = d[mask]
        lines.append("| {} | {} | {:.1%} | {:.1%} | {:.1%} |".format(name, mask.sum(), np.mean(x <= 1),
                                                                       np.mean((x > 1) & (x <= 5)), np.mean(x > 5)))
    off = three & (d_tb > 5) & (d_td > 5) & (d_bd <= 3)
    lines += ["", "Üç kaynağın da olduğu {} karede, yayın ATD'si ile dedektör birbirine ≤ 3 m yakınken taktik ATD "
              "ikisinden de > 5 m uzak: **{:.2%}** ({} kare). Taktik ATD ile yayın ATD'si birbirine ≤ 3 m yakınken "
              "dedektör ikisinden de > 5 m uzak (dedektör yanılıyor ya da top havada): {:.2%}.".format(
                  three.sum(), off.sum() / three.sum(), off.sum(),
                  np.mean((d_tb <= 3) & (d_td > 5) & (d_bd > 5) & three) / np.mean(three))]
    for h in (1, 2):
        mk = three & (m["half"] == h)
        lines.append("- {}. yarı: taktik ATD tek başına yanlış {:.2%}; taktik – yayın > 5 m {:.1%}.".format(
            h, np.mean(off[mk]), np.mean(d_tb[both & (m["half"] == h)] > 5)))

    # C1: the degradation measures per half and per quarter
    alive = played & ~dead
    gk = set(GK_SLOTS.values())
    out_slots = {t: [k for k in np.flatnonzero(m["slot_team"] == t) if k not in gk] for t in (0, 1)}
    count = {t: (~np.isnan(m["px"][:, s])).sum(1) for t, s in out_slots.items()}
    parts = {}
    for h in (1, 2):
        idx = np.flatnonzero(played & (m["half"] == h))
        mid = idx[len(idx) // 2]
        parts["{}. yarı".format(h)] = played & (m["half"] == h)
        parts["{}. çeyrek".format(2 * h - 1)] = played & (m["half"] == h) & (np.arange(len(played)) < mid)
        parts["{}. çeyrek".format(2 * h)] = played & (m["half"] == h) & (np.arange(len(played)) >= mid)
    parts["maçın tamamı"] = played
    lines += ["", "## C1. Bozma modelinin ölçüleri parça parça (bizim maç, GT'siz)", "",
              "Oyun içi = sınıflandırıcının ölü topu dışı. Boşluk: oynanan karelerde ATD topunun kesintisiz yokluğu.", "",
              "| parça | top yok (hepsi) | top yok (oyun içi) | boşluk medyan s | boşluk p90 s | saha oyuncusu medyan H / A | "
              "p5 H / A | > 10 saha oyuncusu H / A | taktik–yayın > 5 m |", "|---|---|---|---|---|---|---|---|---|"]
    order = ["1. çeyrek", "2. çeyrek", "1. yarı", "3. çeyrek", "4. çeyrek", "2. yarı", "maçın tamamı"]
    for name in order:
        p = parts[name]
        miss = np.isnan(tbx) & p
        gaps = [(b - a) / FPS for a, b in runs(miss)]
        lines.append("| {} | {:.1%} | {:.1%} | {:.1f} | {:.1f} | {:.0f} / {:.0f} | {:.0f} / {:.0f} | {:.1%} / {:.1%} | {:.1%} |".format(
            name, miss.sum() / p.sum(), np.mean(np.isnan(tbx)[p & alive]), np.median(gaps), np.percentile(gaps, 90),
            np.median(count[0][p]), np.median(count[1][p]), np.percentile(count[0][p], 5), np.percentile(count[1][p], 5),
            np.mean(count[0][p] > 10), np.mean(count[1][p] > 10), np.mean(d_tb[both & p] > 5)))
    return lines


# ---------------------------------------------------------------- C2, D
def score_all(pairs, truth_fn=None):
    """{(src, labeler): mean spellF1}, and the per-match rows; truth_fn(m) gives other labels."""
    rows = []
    for m, raw in pairs:
        mm = m if truth_fn is None else dict(m, labels=truth_fn(raw))
        for name, f in labelers().items():
            rows.append({"src": src_of(raw["name"]), "match": raw["name"], "labeler": name,
                         "f1": spell_f1(mm["labels"], f(mm))})
    return pd.DataFrame(rows)


def ranking(df):
    t = df.groupby(["src", "labeler"]).f1.mean().unstack()
    return t[["control", "touch", "control+touch"]]


def part_c2(ext):
    variants = {"oranlar ×1 (P2)": {}, "oranlar ×0.5": {k: 0.5 for k in SCALED}, "oranlar ×1.5": {k: 1.5 for k in SCALED},
                "×1 + yumuşatma 25 kare": {"smooth": 25}}
    lines = ["", "## C2. Sonuçlar başka bozma oranlarında da geçerli mi? (spellF1, ölü top etiketten)", "",
             "Ölçeklenen oranlar: {}. Metrica'nın ek top kaybı zaten 0.".format(", ".join(SCALED)), "",
             "| bozma | kaynak | control | touch | control+touch | sıralama |", "|---|---|---|---|---|---|"]
    per_match = None
    for vname, v in variants.items():
        pairs = []
        for m, key in ext:
            r = dict(rates_for(key))
            for k, f in v.items():
                if k == "smooth":
                    r["smooth_frames"] = f
                else:
                    r[k] = r[k] * f
            pairs.append((degrade(m, r, seed=SEED), m))
        df = score_all(pairs)
        if vname == "oranlar ×1 (P2)":
            per_match = df
        for s, r in ranking(df).iterrows():
            order = " > ".join(r.sort_values(ascending=False).index)
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(vname, s, r["control"], r["touch"], r["control+touch"], order))
            print("C2", vname, s, order, flush=True)
    clean = score_all([(m, m) for m, _ in ext])
    lines += ["", "### Dokunuş kazancı maç maç (touch − control)", "",
              "| maç | temiz | bozulmuş |", "|---|---|---|"]
    gains = []
    for name in clean.match.unique():
        g = []
        for df in (clean, per_match):
            x = df[df.match == name].set_index("labeler").f1
            g.append(x["touch"] - x["control"])
        gains.append(g)
        lines.append("| {} | {:+.3f} | {:+.3f} |".format(name, *g))
    gains = np.array(gains)
    dfl = np.array([not n.startswith("metrica") for n in clean.match.unique()])
    lines.append("")
    lines.append("DFL'de touch, kontrolü temizde {}/{} maçta, bozulmuşta {}/{} maçta geçiyor.".format(
        (gains[dfl, 0] > 0).sum(), dfl.sum(), (gains[dfl, 1] > 0).sum(), dfl.sum()))
    return lines


def part_d(ext):
    def merged(s):
        def f(raw):
            base = raw["raw"] if "raw" in raw else raw["labels"]
            out = raw["labels"].copy()
            for h in (1, 2):
                idx = np.flatnonzero(raw["half"] == h)
                out[idx] = merge_short(base[idx], s)
            return out
        return f

    def widen(raw):
        out = raw["labels"].copy()
        d = (out == pm.D).astype(float)
        wide = np.convolve(d, np.ones(2 * FPS + 1), "same") > 0
        out[wide & (raw["half"] > 0)] = pm.D
        return out

    pairs = [(degrade(m, rates_for(key), seed=SEED), m) for m, key in ext]
    for (m, _), (d, raw) in zip(ext, pairs):
        if not m["name"].startswith("metrica"):
            from mtah.external import _dfl_raw
            raw["raw"] = _dfl_raw(m["name"])["raw"]
    variants = {"bizim türetme (DFL 1.6 s birleştirme)": None, "birleştirme 0 s": merged(0), "birleştirme 1.0 s": merged(1.0),
                "birleştirme 3.0 s": merged(3.0), "ölü top blokları ±1 s genişletilmiş": widen}
    lines = ["", "## D. \"Doğru cevap\" başka türetilseydi sıralama değişir miydi? (bozulmuş veri, spellF1)", "",
             "Metrica'da birleştirme bizim olaylardan türettiğimiz etiketin üstüne uygulanıyor; DFL'de ham BallPossession'a.", "",
             "| hedef | kaynak | control | touch | control+touch | sıralama |", "|---|---|---|---|---|---|"]
    for vname, f in variants.items():
        df = score_all(pairs, f)
        for s, r in ranking(df).iterrows():
            order = " > ".join(r.sort_values(ascending=False).index)
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(vname, s, r["control"], r["touch"], r["control+touch"], order))
            print("D", vname, s, order, flush=True)
    return lines


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    ext = external()
    if sys.argv[1:] in (["extra"], ["extra2"], ["extra3"], ["extra4"], ["extra5"]):
        lines = {"extra": lambda: part_a2(ext) + part_b2(ext), "extra2": lambda: part_a3(ext), "extra3": part_a4, "extra4": part_a5, "extra5": lambda: part_a6(ext)}[sys.argv[1]]()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    lines = ["# P5f: tek kaynağa dayanan bulgular için ikinci kontroller (madde 10)", "",
             "`analysis/p5f_verification.py`. Public pencere hiçbir yerde okunmadı.", ""]
    lines += part_a(ext) + [""] + our_measures() + part_c2(ext) + part_d(ext)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
