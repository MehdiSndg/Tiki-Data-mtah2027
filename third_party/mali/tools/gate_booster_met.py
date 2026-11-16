r"""4 gates between booster_met.py outputs (official blend: Metrica booster, given w, given decoder config).

Each pair = (reference json, candidate json) of the same net seed set; rows with booster "new" at --w are used.

  .venv\Scripts\python.exe tools/gate_booster_met.py --pairs outputs/booster_met_w_s1.json:outputs/booster_met_big_s1.json \
      outputs/booster_met_w_s2.json:outputs/booster_met_big_s2.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_day_screen as screen  # noqa: E402


def scores(path, w):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    r = next(x for x in d["rows"] if x["booster"] == "new" and abs(x["w"] - w) < 1e-9)
    return {"dfl_v1": r["v1"], "dfl_v2b": r["v2b"], "metrica": r["metrica"],
            "dfl_v1_per_match": r["per_match"]["v1"], "dfl_v2b_per_match": r["per_match"]["v2b"],
            "metrica_per_match": r["per_match"]["metrica"], "bar": r["bar"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--out")
    args = ap.parse_args()
    pairs = {}
    for i, p in enumerate(args.pairs):
        ref, cand = p.split(":", 1) if p.count(":") == 1 else p.rsplit(":", 1)
        pairs[f"set{i + 1}"] = {"reference": scores(ref, args.w), "candidate": scores(cand, args.w)}
    res = {"both": screen.four_gates(pairs),
           "each": {k: screen.four_gates({k: v}) for k, v in pairs.items()}}
    for k, v in pairs.items():
        print(k, {m: round(v["candidate"][m] - v["reference"][m], 4) for m in ("dfl_v1", "dfl_v2b", "metrica")},
              "pass", res["each"][k]["pass"], "BAR", v["candidate"]["bar"])
    b = res["both"]
    print("both: d_comp", round(b["delta_composite"], 4), "pos", b["positive_matches"], "/6 gates", b["gates"],
          "PASS" if b["pass"] else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
