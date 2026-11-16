"""Logging for autonomous experiment loops: outputs/autonomous_run.log (human readable, append)
and outputs/experiments.csv (one row per evaluated candidate, union of columns)."""

import datetime as dt
import json

import pandas as pd

from ..paths import OUTPUTS, ensure

LOG = OUTPUTS / "autonomous_run.log"


def log(msg):
    ensure(OUTPUTS)
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    print(line, flush=True)


def record(row):
    """Append one candidate row to outputs/experiments.csv."""
    path = OUTPUTS / "experiments.csv"
    r = {"time": dt.datetime.now().isoformat(timespec="seconds"),
         **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}}
    df = pd.DataFrame([r])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, lineterminator="\n")
