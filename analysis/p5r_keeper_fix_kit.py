"""P3 found the keeper fix (ghost keeper slots dropped, keepers recovered from kit colour) "no effect",
by the Smart Tagging side -- the judge P5f/P5m found biased. The kit judge instead. GT-free.

    uv run python analysis/p5r_keeper_fix_kit.py      # -> eda_out/reports/P5r_keeper_fix_kit.md

Two team calls on our match, both with the raw ATD ball and the raw-ATD classifier's dead ball: v1
(the ATD's keeper slots as they are) and v1 with mtah.video_match's keeper fix only. On the frames
where they differ and the ATD has the ball, the kit team of the person nearest to the ball
(analysis/p5m_kit_team.kit_team_at) says which call is right; Smart Tagging alongside.
The recovered keepers come from kit colour themselves, so frames where a recovered keeper position
is the nearest to the ball are circular for the kit judge; they are reported apart from the frames
changed only by dropping a ghost slot.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.video_match import _hom, from_tactical_v2  # noqa: E402
from p4_common import ours  # noqa: E402
from p5m_kit_team import PASS, kit_team_at  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P5r_keeper_fix_kit.md")


def main():
    m = from_tactical()
    hom = _hom()
    v2 = from_tactical_v2(PASS, 0)
    fix = dict(m, px=v2["px"], py=v2["py"])
    dead, _ = ours()
    df = load_frames()
    alive = (m["half"] > 0) & ~np.isin(df["period"].values, NOT_PLAYED) & ~dead
    a, b = control(m, dead=~alive), control(fix, dead=~alive)
    has = ~np.isnan(m["bx"])
    ch = (a != b) & alive & has
    kit = kit_team_at(ch, np.c_[m["bx"], m["by"]], m, hom)
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(a), bool)
    outside[11185:18685] = False
    k = ch & (kit >= 0)
    t = ch & (tag >= 0) & outside
    from mtah.data import GK_SLOTS, PITCH
    L, W = PITCH
    rec = np.zeros(len(a), bool)
    for g in GK_SLOTS.values():
        moved = ~np.isnan(fix["px"][:, g]) & ((np.abs(fix["px"][:, g] - m["px"][:, g]) > 1e-6) | np.isnan(m["px"][:, g]))
        near = np.hypot((fix["px"][:, g] - m["bx"]) * L, (fix["py"][:, g] - m["by"]) * W) <= 2.0
        rec |= moved & near
    lines = ["# P5r: kaleci düzeltmesi, forma hakemiyle (GT'siz)", "",
             "`analysis/p5r_keeper_fix_kit.py`. v1 ile v1 + kaleci düzeltmesi, ATD topuyla; kararların ayrıştığı ve ATD topunun olduğu "
             "{} kare.".format(ch.sum()), "",
             "| hakem | karar belli kare | v1 ile aynı | düzeltmeyle aynı |", "|---|---|---|---|",
             "| forma rengi (topa en yakın kişi) | {} | {:.0%} | {:.0%} |".format(k.sum(), np.mean(kit[k] == a[k]), np.mean(kit[k] == b[k])),
             "| Smart Tagging (public dışı) | {} | {:.0%} | {:.0%} |".format(t.sum(), np.mean(tag[t] == a[t]), np.mean(tag[t] == b[t]))]
    for name, mk in (("kurtarılan kaleci topa ≤ 2 m (forma için döngüsel)", k & rec), ("sadece hayalet iz atıldı", k & ~rec)):
        lines.append("| forma rengi, {} | {} | {:.0%} | {:.0%} |".format(name, mk.sum(), np.mean(kit[mk] == a[mk]) if mk.any() else np.nan,
                                                                        np.mean(kit[mk] == b[mk]) if mk.any() else np.nan))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
