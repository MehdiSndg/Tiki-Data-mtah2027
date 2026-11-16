"""Combine the same report written with several degradation seeds into one: mean [min–max].

    MTAH_SEED=1 uv run python analysis/p5f_verification.py      # -> ..._seed1.md (and seed 2)
    uv run python analysis/seed_combine.py eda_out/reports/P5f_verification.md \
        eda_out/reports/P5f_verification_seed1.md eda_out/reports/P5f_verification_seed2.md
        # -> eda_out/reports/P5f_verification_seeds.md

Line by line: where every file has the same line with only its numbers different, each number
becomes "mean [min–max]" over the files (written once if they agree), in the first file's format.
A line whose words differ between seeds (a ranking that changes, say) is kept from the first file
and flagged with ⚠ followed by the other files' versions.
"""

import re
import sys

NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def fmt(v, like):
    d = len(like.split(".")[1]) if "." in like else 0
    s = "{:.{}f}".format(v, d)
    return ("+" + s) if like.startswith("+") and v >= 0 else s


def combine(lines):
    first = lines[0]
    skel = [NUM.sub("#", ln) for ln in lines]
    if any(s != skel[0] for s in skel[1:]):
        return first + "  ⚠ tohumlar arasında farklı: " + " / ".join(ln.strip() for ln in lines[1:])
    nums = [NUM.findall(ln) for ln in lines]
    out, k = [], 0
    for piece in NUM.split(first):
        out.append(piece)
        if k < len(nums[0]):
            vals = [float(n[k]) for n in nums]
            lo, hi, mean = min(vals), max(vals), sum(vals) / len(vals)
            like = nums[0][k]
            if all(n[k] == like for n in nums):            # names, headers, equal values: as written
                out.append(like)
            elif fmt(lo, like) == fmt(hi, like):
                out.append(fmt(mean, like))
            else:
                out.append("{} [{}–{}]".format(fmt(mean, like), fmt(lo, like), fmt(hi, like)))
            k += 1
    return "".join(out)


def main():
    paths = sys.argv[1:]
    texts = [open(p).read().split("\n") for p in paths]
    if len({len(t) for t in texts}) != 1:
        sys.exit("files differ in length: {}".format([len(t) for t in texts]))
    out = [combine(list(ls)) for ls in zip(*texts)]
    out.insert(1, "\n{} tohumun birleşimi (ortalama [en düşük–en yüksek]): {}\n".format(
        len(paths), ", ".join(paths)))
    dst = paths[0][:-3] + "_seeds.md"
    with open(dst, "w") as fh:
        fh.write("\n".join(out))
    print(dst)


if __name__ == "__main__":
    main()
