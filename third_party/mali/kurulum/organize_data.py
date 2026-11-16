"""Place the downloaded match bundle in the folder layout the official repository expects.

    python kurulum/organize_data.py <download_folder>              # move the files (default)
    python kurulum/organize_data.py <download_folder> --copy       # copy, keep the originals
    python kurulum/organize_data.py <download_folder> --dry-run    # only show what would happen

Target: $MTAH_DATA_DIR if set, otherwise more-than-a-hack-2027/data (the same rule games.py uses):

    data/tactical/BAR-ATH_tactical.mp4, _atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
    data/broadcast/BAR-ATH_broadcast.mp4, _atd.xml, _atd.txt, _homography.csv

Videos may arrive with a numeric suffix (BAR-ATH_tactical-001.mp4); they are renamed to the name
games.py expects. Standard library only, so it runs before the environment exists.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "more-than-a-hack-2027"

# (pattern for the downloaded file name, target sub-folder, target file name)
RULES = [
    (r"BAR-ATH_tactical(?:[-_ ]?\d+)?\.mp4", "tactical", "BAR-ATH_tactical.mp4"),
    (r"BAR-ATH_tactical_atd\.xml", "tactical", "BAR-ATH_tactical_atd.xml"),
    (r"BAR-ATH_tactical_atd\.txt", "tactical", "BAR-ATH_tactical_atd.txt"),
    (r"BAR-ATH_tactical_homography\.csv", "tactical", "BAR-ATH_tactical_homography.csv"),
    (r"BAR-ATH_tactical_smart_tagging\.xml", "tactical", "BAR-ATH_tactical_smart_tagging.xml"),
    (r"BAR-ATH_broadcast(?:[-_ ]?\d+)?\.mp4", "broadcast", "BAR-ATH_broadcast.mp4"),
    (r"BAR-ATH_broadcast_atd\.xml", "broadcast", "BAR-ATH_broadcast_atd.xml"),
    (r"BAR-ATH_broadcast_atd\.txt", "broadcast", "BAR-ATH_broadcast_atd.txt"),
    (r"BAR-ATH_broadcast_homography\.csv", "broadcast", "BAR-ATH_broadcast_homography.csv"),
]


def data_dir():
    return Path(os.environ.get("MTAH_DATA_DIR") or REPO / "data")


def find_sources(source, target_root):
    """{rule index: [paths]} for every file under source that matches a rule.

    Files already inside the target folder are reported separately so a second run is a no-op.
    """
    found = {i: [] for i in range(len(RULES))}
    for path in source.rglob("*"):
        # Skip hidden folders such as .git and .venv: they are large and never hold the bundle.
        if any(part.startswith(".") for part in path.relative_to(source).parts[:-1]):
            continue
        if not path.is_file():
            continue
        for i, (pattern, _, _) in enumerate(RULES):
            if re.fullmatch(pattern, path.name, flags=re.IGNORECASE):
                found[i].append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="folder the bundle was downloaded / unpacked into")
    parser.add_argument("--copy", action="store_true", help="copy instead of move")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target_root = data_dir().resolve()
    if not source.is_dir():
        sys.exit("FAIL  source folder not found: {}".format(source))

    print("source : {}".format(source))
    print("target : {}{}".format(target_root, "   (MTAH_DATA_DIR)" if os.environ.get("MTAH_DATA_DIR") else ""))
    print()

    found = find_sources(source, target_root)
    plan, problems = [], []
    for i, (_, folder, name) in enumerate(RULES):
        destination = target_root / folder / name
        candidates = [p for p in found[i] if p.resolve() != destination]
        if destination.is_file() and not candidates:
            print("  ok      {}/{}  (already in place)".format(folder, name))
            continue
        if not candidates:
            problems.append("missing: no file matching {} under {}".format(name, source))
            continue
        if len(candidates) > 1:
            problems.append("ambiguous: {} candidates for {}: {}".format(
                len(candidates), name, ", ".join(str(c) for c in candidates)))
            continue
        if destination.exists():
            problems.append("exists: {} is already there and differs from {}; remove one first".format(
                destination, candidates[0]))
            continue
        plan.append((candidates[0], destination))

    for src, dst in plan:
        size_gb = src.stat().st_size / 1e9
        print("  {:<7} {}  ->  {}/{}  ({:.2f} GB)".format(
            "copy" if args.copy else "move", src.name, dst.parent.name, dst.name, size_gb))

    if problems:
        print("\nProblems:")
        for problem in problems:
            print("  - " + problem)

    if args.dry_run:
        print("\n--dry-run: nothing changed.")
        return 1 if problems else 0

    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    print("\n{} file(s) {}.".format(len(plan), "copied" if args.copy else "moved"))
    if problems:
        print("Some files could not be placed; see the problems above.")
        return 1
    print("Data layout complete. Next: python kurulum/check_env.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
