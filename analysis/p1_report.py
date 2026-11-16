"""P1 preprocessing report: what every step did, measured without any ground truth.

    uv run python analysis/p1_report.py        # writes eda_out/reports/P1.md

The numbers are regenerated on every run; the interpretation under each section is written by
hand afterwards and kept between the AUTO markers' neighbours, so re-running only refreshes the
measured blocks.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st  # noqa: E402
from games import GAME, asset_paths, half_video_blocks  # noqa: E402
from mtah.ball import INTERP, MASKED, MISSING, OBSERVED, OUTLIER  # noqa: E402
from mtah.data import FPS, GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.periods import K2_BALL, attack_sign, detect_kickoff, periods  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "reports", "P1.md")
AUDIT = os.path.join(ROOT, "eda_out", "B", "atd_ball_audit.csv")
STATUS = {OBSERVED: "OBSERVED", INTERP: "INTERP", OUTLIER: "OUTLIER", MISSING: "MISSING", MASKED: "MASKED"}


def kickoff_row(half, detected, published):
    """One table row; a kickoff that was not found says so instead of failing."""
    if detected is None:
        return "| {}. yarı kickoff | **bulunamadı**: bütün yarı oyun sayıldı | {} | — |".format(half, published)
    return "| {}. yarı kickoff | {} | {} | {:+d} kare |".format(half, detected, published, detected - published)


def section_kickoffs(m, meta):
    _, kick = periods(m, meta)
    published = {1: half_video_blocks(GAME)[0][1], 2: 113742}
    lines = ["| | detected | published | difference |", "|---|---|---|---|"]
    for h in (1, 2):
        lines.append(kickoff_row(h, kick[h], published[h]))
    sign = attack_sign(meta)
    spot = np.hypot((m["bx"] - 0.5) * PITCH[0], (m["by"] - 0.5) * PITCH[1])
    lines += ["", "Gol sonrası kickoff'lar (Smart Tagging aralıkları; hiçbir parametre bunlara göre seçilmedi):", "",
              "| aralık | topun orta noktada görüldüğü | tespit |", "|---|---|---|"]
    for e in st.load():
        if (e["tags"].get("Type") or "").lower() != "kick off (after goal)":
            continue
        s, t = e["start_frame"], e["end_frame"]
        h = 1 if s < 100000 else 2
        seen = bool(np.any(spot[s - 250:t + 250] <= K2_BALL))
        k = detect_kickoff(m["px"], m["py"], m["slot_team"], m["bx"], m["by"], s - 250, t, sign[h],
                           exclude=tuple(GK_SLOTS.values()))
        lines.append("| {}–{} | {} | {} |".format(s, t, "evet" if seen else "**hayır**",
                                                  "—" if k is None else ("{} (aralık içinde)".format(k) if s <= k <= t else "{} (dışında)".format(k))))
    return lines


def section_ball(df, m):
    graded = m["graded"]
    st_ = df["ball_status"].values
    lines = ["| durum | puanlanan kare | oran |", "|---|---|---|"]
    for v, name in STATUS.items():
        k = int((st_[graded] == v).sum())
        lines.append("| {} | {} | {:.2%} |".format(name, k, k / graded.sum()))
    raw = np.full(len(m["bx"]), np.nan)
    raw[1:] = np.hypot(np.diff(m["bx"]) * PITCH[0], np.diff(m["by"]) * PITCH[1]) * FPS
    good = graded & np.isin(st_, (OBSERVED, INTERP))
    sm = df["ball_speed"].values
    lines += ["", "Top hızı (m/s), puanlanan ve kullanılabilir kareler:", "",
              "| | p50 | p90 | p99 | p99.9 |", "|---|---|---|---|---|",
              "| ham (kareden kareye fark) | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(*np.nanpercentile(raw[good], [50, 90, 99, 99.9])),
              "| Savitzky–Golay sonrası | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(*np.nanpercentile(sm[good], [50, 90, 99, 99.9]))]
    lines += ["", "- hareketsiz top: kullanılabilir karelerin {:.1%}'i".format(df["ball_stationary"].values[good].mean()),
              "- saha dışı top: kullanılabilir karelerin {:.1%}'i".format(df["ball_off_pitch"].values[good].mean())]
    # missing gaps: was the ball stationary right before it vanished? (GT-free hint for dead balls)
    status = st_
    starts = np.flatnonzero((status[1:] == MISSING) & np.isin(status[:-1], (OBSERVED, INTERP))) + 1
    before = df["ball_stationary"].values[starts - 1]
    lens = df["ball_gap_len"].values[starts]
    long_ = lens >= 5 * FPS
    lines += ["- ATD topunun kaybolduğu {} anın {:.0%}'inde top kaybolmadan hemen önce hareketsizdi; "
              "5 s'den uzun boşluklarda bu oran {:.0%}, kısa boşluklarda {:.0%}.".format(
                  len(starts), before.mean(), before[long_].mean(), before[~long_].mean())]
    return lines


def section_audit(df):
    if not os.path.exists(AUDIT):
        return ["(denetim dosyası yok: `analysis/b_atd_ball_audit.py` çalıştırılmamış)"]
    a = pd.read_csv(AUDIT)
    a["p1_status"] = df["ball_status"].values[a["frame"].values]
    far = a[(a["status"] == "contradicted") & (a["other_dist_m"] > 5)]
    conf = a[a["status"] == "confirmed"]
    return ["ATD denetimindeki kareler (`eda_out/B/atd_ball_audit.csv`, 1000 kare):", "",
            "| denetim sonucu | kare | P1'in OUTLIER dediği |", "|---|---|---|",
            "| çelişki, > 5 m (ATD büyük ihtimalle yanlış) | {} | {} |".format(len(far), int((far["p1_status"] == OUTLIER).sum())),
            "| doğrulandı (ATD doğru) | {} | {} |".format(len(conf), int((conf["p1_status"] == OUTLIER).sum()))]


def section_players(df, m):
    g = m["graded"]
    lines = ["| | H | A |", "|---|---|---|"]
    for name, fn in (("görünen saha oyuncusu, medyan", lambda s: "{:.0f}".format(np.median(s))),
                     ("p5", lambda s: "{:.0f}".format(np.percentile(s, 5))),
                     ("10'dan fazla (çift sayım) kare oranı", lambda s: "{:.1%}".format((s > 10).mean()))):
        lines.append("| {} | {} | {} |".format(name, fn(df["H_n"].values[g]), fn(df["A_n"].values[g])))
    lines += ["", "- topa en yakın saha oyuncusu, kullanılabilir top karelerinde: H medyan {:.1f} m, A medyan {:.1f} m".format(
        np.nanmedian(df["H_near_d"].values[g]), np.nanmedian(df["A_near_d"].values[g])),
        "- saha oyuncularının medyan hızı: {:.2f} m/s".format(np.nanmedian(df["med_speed"].values[g]))]
    return lines


def section_nans(df, m):
    g = m["graded"]
    lines = ["| sütun | puanlanan karelerde NaN |", "|---|---|"]
    for c in df.columns:
        if df[c].dtype.kind == "f":
            lines.append("| `{}` | {:.1%} |".format(c, np.isnan(df[c].values[g]).mean()))
    return lines


def main():
    meta = asset_paths()["atd_meta"]
    m = load_tactical()
    df = load_frames()
    parts = [("Kickoff tespiti", section_kickoffs(m, meta)), ("Top", section_ball(df, m)),
             ("Top temizliği ve ATD denetimi", section_audit(df)), ("Oyuncular", section_players(df, m)),
             ("Eksik değerler", section_nans(df, m))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        handle.write("# P1 ön işleme raporu\n\n`analysis/p1_report.py` ile üretildi; `cache/frames_v1.parquet` "
                     "üzerinden, GT kullanılmadan. Sayılar her çalıştırmada yenilenir.\n\n")
        for title, lines in parts:
            handle.write("## {}\n\n{}\n\n".format(title, "\n".join(lines)))
    print(open(OUT).read())


if __name__ == "__main__":
    main()
