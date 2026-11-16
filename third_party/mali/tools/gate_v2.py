r"""Five gates (user decision 12 Nov) between booster_met.py --v2 outputs of the same net seed set.

Changes against the 31 Oct gates (tools/run_day_screen.four_gates):
  * the Metrica referee is the BAR-like copy (dead ball visible, wandering, masked) instead of the raw track:
    the raw track never shows the dead ball, so it rewards "ball hidden = dead";
  * 12 held-out PFF FC World Cup games are a referee of their own;
  * the label-free dead-share band comes from external evidence (DFL 0.42-0.52, PFF median 0.41) instead of
    the reference's own output.
Gates
  1 composite      0.30 DFL v2b + 0.25 Metrica BAR-like + 0.25 PFF held-out + 0.20 DFL v1  >= +0.004
  2 non-inferior   DFL v2b / Metrica BAR-like / PFF >= -0.002, DFL v1 >= -0.004
  3 match signs    >= 5 of the 6 DFL v2b + Metrica BAR-like matches positive
  4 PFF signs      >= 9 of the 12 held-out PFF games positive
  5 label-free     BAR possessions 220-285, dead share 0.41-0.52, home share 0.585-0.645
The raw-track Metrica delta is printed for information only.

  .venv\Scripts\python.exe tools/gate_v2.py --pairs outputs/v2_ref_s1.json:outputs/v2_cand_s1.json [ref_s2:cand_s2]
"""

import argparse
import json
from pathlib import Path

W = {"v2b": 0.30, "metbar": 0.25, "pff": 0.25, "v1": 0.20}
FLOOR = {"v2b": -0.002, "metbar": -0.002, "pff": -0.002, "v1": -0.004}
GAIN, MIN_POS, MIN_PFF = 0.004, 5, 9
BANDS = {"n_possessions": (220, 285), "dead_share": (0.41, 0.52), "home_share": (0.585, 0.645)}


def row(path, w, booster):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return next(x for x in d["rows"] if x["booster"] == booster and abs(x["w"] - w) < 1e-9)


def gates(pairs):
    n = len(pairs)
    delta = {k: sum(c[k] - r[k] for r, c in pairs) / n for k in list(W) + ["metrica"]}
    comp = sum(W[k] * delta[k] for k in W)
    per = {}
    for kind in ("v2b", "metbar", "pff"):
        for m in pairs[0][0]["per_match"][kind]:
            per[f"{kind}:{m}"] = sum(c["per_match"][kind][m] - r["per_match"][kind][m] for r, c in pairs) / n
    pos6 = sum(v > 0 for k, v in per.items() if not k.startswith("pff"))
    pos_pff = sum(v > 0 for k, v in per.items() if k.startswith("pff"))
    g = {"1_composite": comp >= GAIN,
         "2_non_inferiority": all(delta[k] >= FLOOR[k] for k in W),
         "3_match_signs": pos6 >= MIN_POS,
         "4_pff_signs": pos_pff >= MIN_PFF,
         "5_label_free": all(lo <= c["bar"][k] <= hi for _, c in pairs for k, (lo, hi) in BANDS.items())}
    return {"delta": delta, "delta_composite": comp, "positive_6": pos6, "positive_pff": pos_pff,
            "per_match_delta": per, "gates": g, "pass": all(g.values()), "bar": [c["bar"] for _, c in pairs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--ref-booster", default="new")
    ap.add_argument("--cand-booster", default="new")
    ap.add_argument("--out")
    args = ap.parse_args()
    pairs = []
    for p in args.pairs:
        ref, cand = p.rsplit(":", 1) if p.count(":") > 1 else p.split(":", 1)
        pairs.append((row(ref, args.w, args.ref_booster), row(cand, args.w, args.cand_booster)))
    res = {"both": gates(pairs), "each": [gates([p]) for p in pairs]}
    for i, e in enumerate(res["each"]):
        print(f"set{i + 1}", {k: round(v, 4) for k, v in e["delta"].items()}, "comp", round(e["delta_composite"], 4),
              "pos", e["positive_6"], "/6 pff", e["positive_pff"], "/12", "PASS" if e["pass"] else "FAIL",
              {k: round(v, 3) for k, v in e["bar"][0].items()})
    b = res["both"]
    print("both:", {k: round(v, 4) for k, v in b["delta"].items()}, "comp", round(b["delta_composite"], 4), "pos",
          b["positive_6"], "/6 pff", b["positive_pff"], "/12", b["gates"], "PASS" if b["pass"] else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
