"""Single entry point for the MTAH 2027 pipeline.

    python run.py cache                    # parse the match data into cache/ (fast reloads afterwards)
    python run.py m0                       # M0: features -> weak labels -> 2-stage LightGBM -> HSMM -> submission
    python run.py m0 --emission rules      # same pipeline with the rule-based emission model
    python run.py score <submission.csv>   # official scorer on every labelled validation block
    python run.py idsse J03WPY             # download + convert one IDSSE/DFL open match (only_alive=False)

Run it with the project interpreter: .venv\\Scripts\\python.exe run.py ...
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mtah.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
