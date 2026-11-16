"""Full-match sanity gates (loose warning bands, never optimisation targets; MASTER_PLAN A6)."""

import numpy as np

from ..constants import A, D, FPS, H, HALVES, X

GATES = {
    "dead_share": (0.25, 0.45),
    "n_possessions": (250, 450),
    "mean_possession_s": (9.0, 20.0),
    "home_share": (0.50, 0.66),
    "half_count_ratio": (0.75, 1.33),
}


def _spells(codes):
    out = []
    for a, b in HALVES.values():
        w = codes[a:b]
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        for i in range(len(bounds) - 1):
            k = w[bounds[i]]
            if k in (H, A):
                out.append((a + bounds[i], a + bounds[i + 1], int(k)))
    return out


def stats(codes):
    codes = np.asarray(codes)
    graded = np.concatenate([codes[a:b] for a, b in HALVES.values()])
    spells = _spells(codes)
    lengths = np.array([b - a for a, b, _ in spells]) if spells else np.array([0])
    alive = np.isin(graded, (H, A))
    per_half = [sum(1 for a, _, _ in spells if lo <= a < hi) for lo, hi in HALVES.values()]
    return {
        "dead_share": float(np.mean(graded == D)),
        "n_possessions": len(spells),
        "mean_possession_s": float(lengths.mean() / FPS),
        "home_share": float(np.sum(graded == H) / max(1, alive.sum())),
        "half_count_ratio": per_half[0] / max(1, per_half[1]),
        "graded_x": int(np.sum(graded == X)),
    }


def check(codes):
    s = stats(codes)
    lines = ["full-match sanity gates (warning bands, not targets):"]
    ok = True
    for key, (lo, hi) in GATES.items():
        inside = lo <= s[key] <= hi
        ok &= inside
        lines.append(f"  [{'ok  ' if inside else 'WARN'}] {key:<18} {s[key]:8.3f}   band [{lo}, {hi}]")
    lines.append(f"  [{'ok  ' if s['graded_x'] == 0 else 'FAIL'}] graded X frames    {s['graded_x']}")
    return s, ok and s["graded_x"] == 0, "\n".join(lines)
