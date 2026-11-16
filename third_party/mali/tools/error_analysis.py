"""Where does the current model still fail? (Gün 2 -> Gün 3 research brief)

Uses the DFL leave-one-match-out probabilities (outputs/decoder/<config>/lomo_*.npy, real
labels) and, as a diagnostic only, the public window. Breaks frame errors down by ball
visibility, distance to the nearest true transition and true state; spell errors by spell
length; and measures start/end offsets of matched spells.

    .venv\\Scripts\\python.exe tools\\error_analysis.py [config] [decoder_json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah import config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.features import build_features  # noqa: E402
from mtah.decoding import post  # noqa: E402
from mtah.io.broadcast import bar_data  # noqa: E402
from mtah.io.gt import public_gt  # noqa: E402
from mtah.labels.masking import gap_lengths  # noqa: E402
from mtah.paths import OUTPUTS, ensure  # noqa: E402
from mtah.pipeline import dfl_referee, engine, m0  # noqa: E402


def spells(lab):
    out = []
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for s, e in zip(np.r_[0, b], np.r_[b, len(lab)]):
        if lab[s] in (C.H, C.A):
            out.append((int(s), int(e), int(lab[s])))
    return out


def dist_to_transition(gt):
    n = len(gt)
    t = np.flatnonzero((gt[1:] != gt[:-1]) & (gt[1:] != C.X) & (gt[:-1] != C.X)) + 1
    if len(t) == 0:
        return np.full(n, 10 ** 6)
    idx = np.arange(n)
    pos = np.searchsorted(t, idx)
    left = np.where(pos > 0, idx - t[np.clip(pos - 1, 0, len(t) - 1)], 10 ** 6)
    right = np.where(pos < len(t), t[np.clip(pos, 0, len(t) - 1)] - idx, 10 ** 6)
    return np.minimum(np.abs(left), np.abs(right))


def analyse(gt, pred, vis):
    known = gt != C.X
    err = known & (pred != gt)
    dt = dist_to_transition(gt)
    out = {"frames": int(known.sum()), "frame_error": float(err.sum() / known.sum())}
    out["error_share_by_ball"] = {k: float((err & m).sum() / max(err.sum(), 1)) for k, m in
                                  (("visible", vis), ("hidden", ~vis))}
    out["error_rate_by_ball"] = {k: float((err & m).sum() / max((known & m).sum(), 1)) for k, m in
                                 (("visible", vis), ("hidden", ~vis))}
    bins = [(0, 12), (12, 50), (50, 125), (125, 10 ** 7)]
    out["error_share_by_dist_to_true_transition"] = {
        f"{a}-{b if b < 10 ** 6 else 'inf'}": float((err & (dt >= a) & (dt < b)).sum() / max(err.sum(), 1)) for a, b in bins}
    conf = {}
    for g in (C.H, C.A, C.D):
        for p in (C.H, C.A, C.D):
            if g != p:
                conf[f"{'HAD'[g]}->{'HAD'[p]}"] = float((known & (gt == g) & (pred == p)).sum() / max(err.sum(), 1))
    out["error_share_by_confusion"] = conf
    gs, ps = spells(np.where(known, gt, C.X)), spells(np.where(known, pred, C.X))
    # greedy IoU matching (same team) for diagnostics
    used, offs_s, offs_e, matched = set(), [], [], np.zeros(len(gs), bool)
    for i, (a, b, k) in enumerate(gs):
        best, bj = 0.0, None
        for j, (c, d, kk) in enumerate(ps):
            if kk != k or j in used or d <= a or c >= b:
                continue
            iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou > best:
                best, bj = iou, j
        if bj is not None and best >= 0.5:
            used.add(bj)
            matched[i] = True
            offs_s.append(ps[bj][0] - a)
            offs_e.append(ps[bj][1] - b)
    lens = np.array([b - a for a, b, _ in gs]) / 25
    out["gt_spells"] = len(gs)
    out["pred_spells"] = len(ps)
    out["recall_iou50_by_gt_length_s"] = {
        f"{lo}-{hi}": float(matched[(lens >= lo) & (lens < hi)].mean()) if ((lens >= lo) & (lens < hi)).any() else None
        for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9))}
    out["gt_spell_share_by_length_s"] = {f"{lo}-{hi}": float(((lens >= lo) & (lens < hi)).mean())
                                         for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9))}
    if offs_s:
        out["matched_start_offset_frames_p10_50_90"] = np.percentile(offs_s, [10, 50, 90]).tolist()
        out["matched_end_offset_frames_p10_50_90"] = np.percentile(offs_e, [10, 50, 90]).tolist()
        out["matched_abs_start_offset_median"] = float(np.median(np.abs(offs_s)))
    unmatched_pred = len(ps) - len(used)
    out["false_pred_spells"] = unmatched_pred
    return out


def main(config="main", decoder_json=None):
    cfg = config_mod.load(config)
    if decoder_json:
        cfg = json.loads(Path(decoder_json).read_text())
    folder = OUTPUTS / "decoder" / config
    lengths = gap_lengths(bar_data(cfg))
    res = {"dfl": {}}
    for path in sorted(folder.glob("lomo_*.npy")):
        m = path.stem.split("_", 1)[1]
        prep = dfl_referee.prepare(m, cfg, lengths)
        probs = np.load(path)
        pred = engine.decode_runs(probs, prep["runs"], cfg)
        if cfg.get("post"):
            kfe = build_features(prep["corrupted"], prep["cfg"], None, groups=["kin"], cache_key=prep["key"])
            pred = post.apply(pred, prep["runs"], cfg, kfe["kin_kick"].to_numpy() > 0,
                              prep["corrupted"].ball, prep["data"].pitch)
        gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        res["dfl"][m] = analyse(gt, pred, ~np.isnan(prep["corrupted"].ball[:, 0]))
    bar = bar_data(cfg)
    probs = np.load(OUTPUTS / f"{cfg['name']}_lgbm" / "probs.npy")
    lab = m0.decode_match(probs, cfg)
    res["public_diagnostic"] = analyse(public_gt(), lab, ~np.isnan(bar.ball[:, 0]))
    agg = {}
    for key in ("frame_error",):
        agg[key] = float(np.mean([v[key] for v in res["dfl"].values()]))
    for key in ("error_share_by_ball", "error_rate_by_ball", "error_share_by_dist_to_true_transition",
                "error_share_by_confusion", "recall_iou50_by_gt_length_s", "gt_spell_share_by_length_s"):
        ks = list(next(iter(res["dfl"].values()))[key])
        agg[key] = {k: float(np.mean([v[key][k] for v in res["dfl"].values() if v[key][k] is not None]))
                    for k in ks}
    agg["matched_abs_start_offset_median"] = float(np.mean([v.get("matched_abs_start_offset_median", np.nan)
                                                            for v in res["dfl"].values()]))
    res["dfl_mean"] = agg
    out = ensure(OUTPUTS / "analysis") / f"errors_{config}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"dfl_mean": agg, "public_diagnostic": res["public_diagnostic"]}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
