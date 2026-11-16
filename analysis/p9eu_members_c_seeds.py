"""P9eu: six more seeds (6-11) of each of v29's six recipe-C team members, so v30 can average twelve seeds per member
instead of six (variance reduction; same recipe, data and code). Saved under mtah.members_c.cache_path with
seeds 6-11. GPU; MEMBERS splits the work (names as analysis/p9ej_members_c_oof.MEMBERS).

    MEMBERS=big,set,set_mt .venv/bin/python analysis/p9eu_members_c_seeds.py

Registered before the run (14 Nov): v30 (twelve seeds) replaces v29 (six) only if on SoccerTrack's full chain (P9ei
pattern) it is not behind v29 at either referee, and the label-free video agreement on our match (P9ep, |e| > 1) is not
lower; public is reported, never used.

P9ex (15 Nov, registered before the run): seeds 12-17 and 18-23 (SEED0=12, 18) for a 24-seed v31; v31 replaces the
candidate only if on SoccerTrack it is not behind the twelve-seed v30 at either referee AND not behind on PFF (P9ew's
judge) at either referee; video agreement is reported (it split 0.0003 against v30, inside its noise).

P9fb (15 Nov, registered before the run): recipe-C members of the two other architectures (gru, tcn; single and
two-head; seeds 0-5) added to the candidate's six as extra members (probability mean over all members); adopted only
if not behind the candidate on SoccerTrack and on PFF at either referee. P9ex2: seeds 24-29 (SEED0=24) under P9ex's rule.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]

MEMBERS = {"big": ("big", False, 1), "tcn128d3": ("tcn128", False, 3), "set": ("set", False, 1),
           "big_mt": ("big", True, 1), "tcn128_mt": ("tcn128", True, 1), "set_mt": ("set", True, 1),
           # P9fb (15 Nov): two more architectures on recipe C, seeds 0-5, for diversity
           "gru": ("gru", False, 1), "gru_mt": ("gru", True, 1), "tcn": ("tcn", False, 1), "tcn_mt": ("tcn", True, 1)}
SEEDS = tuple(range(int(os.environ.get("SEED0", 6)), int(os.environ.get("SEED0", 6)) + 6))   # SEED0=12 / 18: P9ex


def main():
    from mtah.members_c import cache_path, external_gru_c
    for name in os.environ.get("MEMBERS", ",".join(MEMBERS)).split(","):
        arch, mt, draws = MEMBERS[name]
        external_gru_c("cuda", arch=arch, multitask=mt, seeds=SEEDS, draws=draws)
        print("trained", name, os.path.basename(cache_path("team", arch, mt, SEEDS, draws)), flush=True)


if __name__ == "__main__":
    main()
