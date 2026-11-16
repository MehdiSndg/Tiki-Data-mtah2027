"""P9a's out-of-fold dump redone with v16's 6 seeds per net (P9ad): every member, its two-head twin, the boundary GRU and
the dead-ball GRU trained per test match with seeds 3-8 instead of 3-5 (v16 keeps 3 seeds for the boundary and dead-ball
GRUs; 6 there changed nothing, P9ae / P9ag), written to MTAH_OOF_DIR (default
cache/oof_v16) in P9a's format. Gives v16's own full-chain DFL / Metrica score (P9a / P9b scoring) and a new base for
the CPU studies. GPU.

    python analysis/p9al_oof_v16.py <dfl|metrica> [only=<match>,...]      # -> cache/oof_v16/oof_<match>.npz
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof_v16"))
import p9a_oof  # noqa: E402

p9a_oof.SEEDS = (3, 4, 5, 6, 7, 8)

if __name__ == "__main__":
    p9a_oof.main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
