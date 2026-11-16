"""Gün 2 veri madenciliği (rapor açık soruları). GT okumaz.

1) Broadcast ATD -> tactical zaman hizalaması ve tactical'da NaN olan topun kurtarılma oranı
2) Devre sonları: puanlanan aralığın son saniyelerinde tempo / top / tahmin profili
3) GOALS / SET PIECES tag'lerinden gerçek yeniden başlatmaya ofset (izlemeden kestirim)
Çıktı: outputs/mining/day2.json (+ broadcast_ball_on_tactical.npy) ve ekrana özet.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah import config, constants as C  # noqa: E402
from mtah.features import build_features  # noqa: E402
from mtah.io.atd import cached_barath, read_metadata, read_raw  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.labels.weak import player_median_speed  # noqa: E402
from mtah.paths import OFFICIAL, OUTPUTS, ensure  # noqa: E402

warnings.simplefilter("ignore")
BROAD = OFFICIAL / "data" / "broadcast"
WIN = 1_500


def lag_error(pb, tp, idx, lag):
    """Median one-sided chamfer distance (m): every broadcast-visible player to the nearest
    tactical player at frame i + lag. Permutation-invariant (slots are tracklets, not people)
    and robust to the broadcast camera showing only part of the pitch."""
    errs = []
    for i in idx:
        j = i + lag
        if j < 0 or j >= len(tp):
            continue
        b = pb[i][np.isfinite(pb[i, :, 0])]
        t = tp[j][np.isfinite(tp[j, :, 0])]
        if len(b) < 5 or len(t) < 10:
            continue
        errs.append(np.median(np.linalg.norm(b[:, None] - t[None], axis=-1).min(1)))
    return float(np.median(errs)) if len(errs) >= 8 else np.inf


def best_lag(pb, tp, idx, lags):
    return min((lag_error(pb, tp, idx, lag), lag) for lag in lags)[::-1]


def broadcast_fusion(tac):
    slot_team, _, _ = read_metadata(BROAD / "BAR-ATH_broadcast_atd.xml")
    _, players, ball = read_raw(BROAD / "BAR-ATH_broadcast_atd.txt", len(slot_team))
    scale = np.array(C.PITCH, np.float32)
    pb, bb = players[:, :, :2] * scale, ball[:, :2] * scale
    tp = tac.players
    rows, lag_of, prev = [], np.full(len(pb), np.nan), None
    for a in range(0, len(pb) - WIN, WIN):
        idx = np.arange(a, a + WIN, 75)
        if np.isfinite(pb[idx, :, 0]).sum(1).mean() < 5:
            continue
        coarse = range(-20_000, 50_000, 100) if prev is None else range(prev - 1_500, prev + 1_501, 50)
        lag, err = best_lag(pb, tp, idx, coarse)
        if err > 2.5 and prev is not None:                      # a jump (half-time, cut): wide search
            lag, err = best_lag(pb, tp, idx, range(-20_000, 50_000, 100))
        lag, err = best_lag(pb, tp, np.arange(a, a + WIN, 25), range(lag - 60, lag + 61, 4))
        lag, err = best_lag(pb, tp, np.arange(a, a + WIN, 10), range(lag - 4, lag + 5))
        rows.append((a, int(lag), err))
        if err < 1.0:
            lag_of[a:a + WIN] = lag
            prev = lag
    fused = np.full((tac.n_frames, 2), np.nan, np.float32)
    for i in np.flatnonzero(np.isfinite(lag_of) & np.isfinite(bb[:, 0])):
        j = i + int(lag_of[i])
        if 0 <= j < tac.n_frames:
            fused[j] = bb[i]
    graded = C.graded_mask()
    tvis = ~np.isnan(tac.ball[:, 0])
    have = np.isfinite(fused[:, 0])
    both = have & tvis & graded
    dist = np.linalg.norm(fused[both] - tac.ball[both], axis=1)
    rec = have & ~tvis & graded
    n_nan = int((~tvis & graded).sum())
    gap_len = np.zeros(tac.n_frames, int)
    e = np.diff(np.r_[0, (~tvis).astype(np.int8), 0])
    for s_, t_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        gap_len[s_:t_] = t_ - s_
    np.save(ensure(OUTPUTS / "mining") / "broadcast_ball_on_tactical.npy", fused)
    good = [r for r in rows if r[2] < 1.0]
    return {"windows_aligned": len(good), "windows_total": len(rows),
            "lags_distinct": sorted({r[1] for r in good}),
            "align_err_m_median": round(float(np.median([r[2] for r in good])), 3) if good else None,
            "broadcast_ball_visible_share": round(float(np.isfinite(bb[:, 0]).mean()), 3),
            "tactical_nan_graded_frames": n_nan, "recovered_frames": int(rec.sum()),
            "recovered_share": round(float(rec.sum() / max(n_nan, 1)), 4),
            "recovered_in_gaps_over_10s": int((rec & (gap_len > 250)).sum()),
            "agreement_frames": int(both.sum()),
            "agreement_dist_m_p50_p90": [round(float(np.percentile(dist, q)), 2) for q in (50, 90)] if len(dist) else None,
            "agreement_within_2m": round(float(np.mean(dist < 2.0)), 3) if len(dist) else None}


def half_ends(tac, labels):
    med = player_median_speed(tac)
    vis = ~np.isnan(tac.ball[:, 0])
    out = {}
    for name, (a, b) in C.HALVES.items():
        rows = []
        for sec in (60, 30, 20, 10, 5, 2):
            s = slice(b - sec * 25, b)
            rows.append({"last_s": sec, "median_player_speed": round(float(np.nanmedian(med[s])), 2),
                         "ball_visible": round(float(vis[s].mean()), 2),
                         "pred_D_share": None if labels is None else round(float(np.mean(labels[s] == C.D)), 2)})
        after = slice(b, min(b + 750, tac.n_frames))
        live = np.flatnonzero(np.nan_to_num(med[a:b]) > 1.5)
        out[name] = {"profile": rows,
                     "after_end_30s_median_speed": round(float(np.nanmedian(med[after])), 2),
                     "frames_after_last_live_tempo": int(b - (a + live[-1])) if len(live) else None,
                     "pred_last_state_run_frames": None if labels is None else int(
                         b - 1 - np.flatnonzero(labels[a:b] != labels[b - 1])[-1] - a) if np.any(labels[a:b] != labels[b - 1]) else None,
                     "pred_last_state": None if labels is None else "HADX"[int(labels[b - 1])]}
    return out


def tag_offsets(tac, feats):
    tags = load_tags()
    med = player_median_speed(tac)
    kick = feats["kin_kick"].to_numpy() > 0
    res = {}
    for code in ("GOALS", "SET PIECES"):
        t = tags[tags["code"] == code]
        offs_start, offs_end = [], []
        for _, r in t.iterrows():
            a, b = int(r["start"]), int(r["end"])
            lo, hi = max(0, a - 750), min(tac.n_frames, b + 750)
            slow = np.nan_to_num(med[lo:hi], nan=99) < 1.2
            run = np.convolve(slow.astype(float), np.ones(50), "full")[:len(slow)] >= 49   # 2 s slow just before
            cand = np.flatnonzero(kick[lo:hi] & run)
            if code == "GOALS":
                cand = cand[cand + lo > a]          # kick-off after the goal
            if len(cand) == 0:
                continue
            restart = lo + int(cand[0])
            offs_start.append((restart - a) / 25)
            offs_end.append((restart - b) / 25)
        pct = lambda v: np.percentile(v, [10, 50, 90]).round(1).tolist() if v else None  # noqa: E731
        res[code] = {"n_tags": int(len(t)), "n_estimated": len(offs_start),
                     "restart_minus_tag_start_s_p10_50_90": pct(offs_start),
                     "restart_minus_tag_end_s_p10_50_90": pct(offs_end),
                     "tag_length_s_p10_50_90": pct(list((t["end"] - t["start"]) / 25)),
                     "types": t["type"].value_counts().to_dict(),
                     "team_na": int((t["team"].isna() | (t["team"] < 0)).sum()) if "team" in t else None}
    return res


def main():
    tac = cached_barath()
    cfg = config.load("main")
    feats = build_features(tac, cfg, load_tags(), groups=["kin"], cache_key="barath")
    sub = OUTPUTS / "main_lgbm" / "submission.csv"
    labels = (pd.read_csv(sub)["label"].map({"H": 0, "A": 1, "D": 2, "X": 3}).to_numpy()
              if sub.exists() else None)
    out = {"half_ends": half_ends(tac, labels), "tag_offsets": tag_offsets(tac, feats)}
    out["broadcast"] = broadcast_fusion(tac)
    path = ensure(OUTPUTS / "mining") / "day2.json"
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
