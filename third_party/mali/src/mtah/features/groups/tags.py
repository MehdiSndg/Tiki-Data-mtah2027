"""tags (SOURCE-SPECIFIC): Smart Tagging priors. Only BAR-ATH has them; joint training with
other sources must exclude this group (registry.common_groups does).

Mirror tags (DEFENDING ... codes: the same interval as an attacking code, other team) are never
used, so no tag is counted twice. With features.tags.resolve_na (Gün 2) the 32 set pieces whose
team is N/A get one from tracking:
  * Kick Off (After Goal): the team that conceded the goal tagged just before (exact),
  * Kick Off (Start): the team of the outfield player nearest the centre spot during the tag,
  * Throw-in: the team of the player standing on / outside a touchline during the tag (median
    over the window of the most-outside player, ties broken by isolation from others).
"""

import numpy as np

from ...io.tags import interval_signal
from ..registry import register


def resolve_na_teams(t, ctx):
    t = t.copy()
    L, W = ctx.length, ctx.width
    P, team = ctx.P, ctx.data.team
    goals = t[t["code"] == "GOALS"].sort_values("start")
    for i, r in t[(t["code"] == "SET PIECES") & (t["team"] < 0)].iterrows():
        a, b = max(0, int(r["start"])), min(ctx.n, int(r["end"]))
        if b <= a:
            continue
        if r["type"] == "Kick Off (After Goal)":
            before = goals[goals["start"] <= a]
            if len(before):
                t.at[i, "team"] = 1 - int(before.iloc[-1]["team"])
            continue
        votes = []
        for f in range(a, b, 5):
            xy = P[f]
            ok = np.isfinite(xy[:, 0])
            if ok.sum() < 10:
                continue
            if r["type"].startswith("Kick Off"):
                d = np.hypot(xy[:, 0] - L / 2, xy[:, 1] - W / 2)
                d[~ok] = np.inf
                votes.append(team[int(np.argmin(d))])
            elif r["type"] == "Throw-in":
                inside = np.minimum(xy[:, 1], W - xy[:, 1])                  # < 0 = outside
                inside[~ok] = np.inf
                j = int(np.argmin(inside))
                if inside[j] < 1.0:
                    votes.append(team[j])
        if votes:
            t.at[i, "team"] = int(np.round(np.median(votes)))
    return t


@register("tags", source_specific=True, ball_free=True)
def tags(ctx):
    t = ctx.tags
    if ctx.cfg["features"].get("tags", {}).get("resolve_na", False):
        t = resolve_na_teams(t, ctx)
    n = ctx.n
    team_sign = np.where(t["team"] == 0, 1.0, np.where(t["team"] == 1, -1.0, 0.0))
    is_sp = (t["code"] == "SET PIECES").to_numpy()
    att = t["is_attacking"].to_numpy()
    return {
        "tag_sp_in": (interval_signal(t, n, is_sp) > 0).astype(np.float32),
        "tag_sp_team": interval_signal(t, n, is_sp, team_sign[is_sp]),
        "tag_phase": interval_signal(t, n, att, team_sign[att]),
        "tag_goal_in": (interval_signal(t, n, (t["code"] == "GOALS").to_numpy()) > 0).astype(np.float32),
    }
