"""Do our DFL possession labels switch team when the ball is received? ELASTIC (Kim et al., CIKM 2026; CC BY 4.0) gives,
for J03WMX, J03WN1 and J03WPY, every event's frame and its reception frame, the median of three annotators. A team
timeline is built from it (the actor's team from the event frame, the receiver's team from the reception frame; out
of play and unknown receivers end nothing) and compared with DFL's raw BallPossession and with our merged labels:
agreement on alive frames, and the offset of each of our team switches to the nearest ELASTIC switch (ours minus
theirs, within 3 s). Research report item 6. CPU, light.

    python analysis/p9g_elastic_labels.py <elastic_repo>      # -> eda_out/reports/P9g_elastic_labels.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import _dfl_raw, from_dfl  # noqa: E402

MATCHES = ("J03WMX", "J03WN1", "J03WPY")
HALF_START = {1: 10000, 2: 100000}        # DFL frame ids of each half's first frame
FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P9g_elastic_labels.md")


def team_of(pid):
    return pm.H if str(pid).startswith("home") else pm.A if str(pid).startswith("away") else None


def timeline(ev, n, half_index):
    """(n,) codes from ELASTIC events: H / A from each touch on, X before the first one of a half."""
    marks = []
    for _, e in ev.iterrows():
        h = int(e.period_id)
        t = team_of(e.player_id)
        if t is not None:
            marks.append((half_index(h, e.frame_id), t))
        r = team_of(e.receiver_id)
        if r is not None and np.isfinite(e.receive_frame_id):
            marks.append((half_index(h, int(e.receive_frame_id)), r))
    out = np.full(n, pm.X, np.int8)
    marks.sort()
    for (a, t), (b, _) in zip(marks, marks[1:] + [(n, None)]):
        if 0 <= a < n:
            out[a:max(a, min(b, n))] = t
    return out


def switches(codes, alive):
    c = np.where(alive, codes, pm.X)
    i = np.flatnonzero((c[1:] != c[:-1]) & np.isin(c[1:], (pm.H, pm.A)) & np.isin(c[:-1], (pm.H, pm.A))) + 1
    return i


def main(repo):
    rows, offs = [], {"raw": [], "merged": []}
    for mid in MATCHES:
        d = _dfl_raw(mid)
        n1 = int((d["half"] == 1).sum())
        idx = lambda h, f: (f - HALF_START[1]) if h == 1 else n1 + 1 + (f - HALF_START[2])
        ev = pd.read_parquet(os.path.join(repo, "benchmark", "gt", "{}.parquet".format(mid)))
        el = timeline(ev, len(d["raw"]), idx)
        mine = from_dfl(mid)["labels"]
        alive = np.isin(d["raw"], (pm.H, pm.A)) & np.isin(el, (pm.H, pm.A))
        se = switches(el, alive)
        for name, lab in (("raw", d["raw"]), ("merged", mine)):
            ok = alive & np.isin(lab, (pm.H, pm.A))
            so = switches(lab, ok)
            for s in so:
                j = np.argmin(np.abs(se - s)) if len(se) else None
                if j is not None and abs(se[j] - s) <= 3 * FPS:
                    offs[name].append((s - se[j]) / FPS)
            rows.append({"match": mid, "labels": name, "agree": float(np.mean(lab[ok] == el[ok])), "switches": len(so), "elastic_switches": len(se),
                         "matched": int(sum(abs(se[np.argmin(np.abs(se - s))] - s) <= 3 * FPS for s in so)) if len(se) else 0})
        print(rows[-2:], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9g: DFL etiketlerimiz ELASTIC'in alış karelerine göre", "", "`analysis/p9g_elastic_labels.py`.", "",
             "| maç | etiket | kare uyumu (oyunda) | bizim değişim | ELASTIC değişim | ±3 s eşleşen |", "|---|---|---|---|---|---|"]
    lines += ["| {match} | {labels} | {agree:.3f} | {switches} | {elastic_switches} | {matched} |".format(**r) for r in rows]
    for name, o in offs.items():
        o = np.array(o)
        if len(o):
            lines.append("")
            lines.append("{}: bizim − ELASTIC değişim farkı (s) medyan {:+.2f}, ortalama {:+.2f}, p10 {:+.2f}, p90 {:+.2f}, |fark| ≤ 0.2 s %{:.0f}, "
                         "erken (< −0.2 s) %{:.0f}, geç (> +0.2 s) %{:.0f}; n {}.".format(name, np.median(o), o.mean(), np.percentile(o, 10), np.percentile(o, 90),
                                                                                       100 * np.mean(np.abs(o) <= 0.2), 100 * np.mean(o < -0.2), 100 * np.mean(o > 0.2), len(o)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
