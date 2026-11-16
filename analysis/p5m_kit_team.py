"""Open items 16 and 2: where the ATD lost the ball, is the nearest ATD player of the right team? A
third judge from the video: the kit colour of the person nearest to the detector's ball. GT-free.

    uv run python analysis/p5m_kit_team.py [broadcast]      # -> eda_out/reports/P5m_kit_team[_broadcast].md

With "broadcast" the filled ball is the broadcast ATD's (P3 found it hurt too, by Smart Tagging).

P3 kept the detector's ball out of the team call because the Smart Tagging side disagreed;
external data (P5f) said the opposite. Hypothesis (b): in those moments the ATD's players are
wrong. Test, on our match (alive = outside the raw-ATD classifier's dead ball):
  gap   alive, no ATD ball, a ball_v1 box (conf >= 0.3, not a painted spot; strict-fp32 pass)
  easy  alive, ATD ball and ball_v1 box within 1 m of each other (calibration), same count
For each frame: the YOLO person (cache/yolo, conf >= 0.3) whose foot point is nearest to the
detector ball, within NEAR_M, its team from the kit model (mtah.kits, trained on slot-labelled
boxes; P_MIN), and the team of the nearest ATD player within NEAR_M.
  1. agreement of the two teams, easy vs gap frames
  2. how often the ATD has nobody within NEAR_M while the video has a person
  3. on the frames where filling the ball changes the control rule's call (P5f A5), which call the
     kit team sides with: the held one or the filled one
"""

import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.kits import fit_kit_model, kit_proba, slot_labels, torso_features  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.video_match import _hom, detector_ball  # noqa: E402
from p3h_missing_carrier import to_pitch  # noqa: E402
from p4_common import ours  # noqa: E402

NEAR_M, P_MIN, SEED = 2.0, 0.8, 0
PASS = os.path.join(ROOT, "cache", "ball_v1_pass_blackwell")
OUT = os.path.join(ROOT, "eda_out", "reports", "P5m_kit_team{}.md")


def persons():
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "cache", "yolo", "chunk_*.npz")))]
    c = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "cls", "conf", "x", "y", "w", "h")}
    keep = (c["cls"] == 0) & (c["conf"] >= 0.3)
    return {k: v[keep] for k, v in c.items()}


def kit_team_at(want, xy, m, hom):
    """Per frame, the kit team (0 home, 1 away; -1 unknown) of the YOLO person nearest to the ball xy
    (normalised pitch), within NEAR_M, on the frames in want. The kit model as P3c / P3h train it."""
    L, W = PITCH
    n = len(want)
    d = np.load(os.path.join(ROOT, "cache", "kit_features.npz"))
    X, Y = to_pitch(hom[d["frame"]], d["x"], d["y"] + d["h"] / 2)
    lab = slot_labels(np.c_[X, Y], m["px"][d["frame"]] * L, m["py"][d["frame"]] * W, m["slot_team"], GK_SLOTS)
    model = fit_kit_model(d["feat"], lab)
    B = persons()
    sel = np.isin(B["frame"], np.flatnonzero(want))
    B = {k: v[sel] for k, v in B.items()}
    fx, fy = to_pitch(hom[B["frame"]], B["x"], B["y"] + B["h"] / 2)
    dist = np.hypot(fx - xy[B["frame"], 0] * L, fy - xy[B["frame"], 1] * W)
    nearest = {}
    for i in np.argsort(dist):
        f = int(B["frame"][i])
        if dist[i] <= NEAR_M and f not in nearest:
            nearest[f] = i
    print("frames asked", int(want.sum()), "with a person within", NEAR_M, "m:", len(nearest), flush=True)
    kit = np.full(n, -1)
    frames = sorted(nearest)
    if not frames:
        return kit
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    k = 0
    for fr in range(frames[0], frames[-1] + 1):
        if not cap.grab():
            break
        if fr != frames[k]:
            continue
        img = cap.retrieve()[1]
        i = nearest[fr]
        ft = torso_features(img, B["x"][i], B["y"][i], B["w"][i], B["h"][i])
        if ft is not None:
            pr = kit_proba(model, ft[None])[0]
            pt = np.array([pr[0] + pr[2], pr[1] + pr[3]])
            if pt.max() >= P_MIN:
                kit[fr] = int(np.argmax(pt))
        k += 1
        if k == len(frames):
            break
    return kit


def main():
    m = from_tactical()
    hom = _hom()
    dead, _ = ours()
    df = load_frames()
    n = len(m["bx"])
    alive = (m["half"] > 0) & ~np.isin(df["period"].values, NOT_PLAYED) & ~dead
    source = sys.argv[1] if len(sys.argv) > 1 else "detector"
    if source == "broadcast":
        from mtah.broadcast import broadcast_ball_on_tactical
        xy = np.c_[broadcast_ball_on_tactical()]
        det = ~np.isnan(xy[:, 0])
    else:
        xy, conf, spot = detector_ball(hom, n, PASS, 0, conf_min=0.3)
        det = ~np.isnan(xy[:, 0]) & ~spot
    L, W = PITCH
    gap = alive & np.isnan(m["bx"]) & det
    close = np.hypot((xy[:, 0] - m["bx"]) * L, (xy[:, 1] - m["by"]) * W) <= 1.0
    easy_all = np.flatnonzero(alive & ~np.isnan(m["bx"]) & det & close)
    easy = np.zeros(n, bool)
    easy[np.random.RandomState(SEED).choice(easy_all, size=min(len(easy_all), gap.sum()), replace=False)] = True

    kit = kit_team_at(gap | easy, xy, m, hom)

    D = np.hypot((m["px"] - xy[:, :1]) * L, (m["py"] - xy[:, 1:2]) * W)
    D = np.where(np.isnan(D), np.inf, D)
    j = np.argmin(D, axis=1)
    atd = np.where(D[np.arange(n), j] <= NEAR_M, m["slot_team"][j], -1)

    lines = ["# P5m: boşluklarda takım, forma renginden (madde 16 + 2, GT'siz); doldurulan top: {}".format(source), "",
             "`analysis/p5m_kit_team.py`. Dedektör topuna en yakın kişi (≤ {:.0f} m) ve forma rengi takımı (olasılık ≥ {}),"
             " en yakın ATD oyuncusunun takımıyla. Kolay: ATD topu ile dedektör topu ≤ 1 m.".format(NEAR_M, P_MIN), "",
             "| kareler | kare | formayla takımı belli | ATD'de ≤ 2 m oyuncu yok | ikisi de var: aynı takım |", "|---|---|---|---|---|"]
    for name, mask in (("kolay (kalibrasyon)", easy), ("boşluk", gap)):
        kv = mask & (kit >= 0)
        both = kv & (atd >= 0)
        lines.append("| {} | {} | {:.0%} | {:.0%} | {:.1%} ({} kare) |".format(
            name, mask.sum(), kv.sum() / mask.sum(), np.mean(atd[kv] < 0), np.mean(kit[both] == atd[both]), both.sum()))

    ref = control(m, dead=~alive)
    filled = control(dict(m, bx=np.where(det & gap, xy[:, 0], m["bx"]).astype(np.float32),
                          by=np.where(det & gap, xy[:, 1], m["by"]).astype(np.float32)), dead=~alive)
    ch = (filled != ref) & alive & (kit >= 0)
    lines += ["", "Doldurmanın kararı değiştirdiği karelerde ({} karede forma belli): forma takımı son takımı tutanla "
              "aynı {:.0%}, doldurulanla aynı {:.0%}. Bütün boşluk karelerinde forma takımıyla uyum: tut {:.0%}, doldur {:.0%}.".format(
                  ch.sum(), np.mean(kit[ch] == ref[ch]), np.mean(kit[ch] == filled[ch]),
                  np.mean(kit[gap & (kit >= 0)] == ref[gap & (kit >= 0)]), np.mean(kit[gap & (kit >= 0)] == filled[gap & (kit >= 0)]))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.format("" if source == "detector" else "_" + source), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
