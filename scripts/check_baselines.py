"""Rebuild the four committed baselines and check they come out byte-identical.

    uv run python scripts/check_baselines.py

The competition requires that a repository regenerates its submission exactly -- "not
approximately, the same file" (RULES.md). The committed baselines are the one artefact in this
repository whose exact bytes we know in advance, so reproducing them is the cheapest proof that
this environment reads the data, aligns the frames and writes CSVs the way the organisers did.

Writes only to a temporary directory, which is removed afterwards.
"""

import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
COMMITTED = os.path.join(ROOT, "evaluation", "results")
NAMES = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable, os.path.join(ROOT, "starter", "baseline.py"), "--out", out],
                       check=True, cwd=ROOT)
        print()
        mismatches = 0
        for name in NAMES:
            mine = sha256(os.path.join(out, name + ".csv"))
            theirs = sha256(os.path.join(COMMITTED, name + ".csv"))
            same = mine == theirs
            mismatches += not same
            print("  {:<4} {:<20} {}".format("ok" if same else "DIFF", name, mine[:16]))
    print("\n{}".format("all baselines byte-identical" if not mismatches
                        else "{} baseline(s) differ".format(mismatches)))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
