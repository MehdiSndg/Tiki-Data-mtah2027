r"""Clean-clone replay of the sealed official file of the external-data-only recipe (main_ext_met_dt).

The snapshot is pinned to the seal's revision, runs offline with its own empty cache/ and outputs/
inside <job>/code (raw data only is shared, read-only), and the final SHA-256 is compared with
outputs/main/official_seal.json. The official outputs are never written; the seal gets a reproduction
entry only when the replay matches.

  .venv\Scripts\python.exe tools/run_ext_replay.py --job-dir outputs/clean_replay_ext
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_mix_promotion as common

CODE_ROOT = Path(__file__).resolve().parents[1]
PYTHON = CODE_ROOT / ".venv" / "Scripts" / "python.exe"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args()
    job = Path(args.job_dir).resolve()
    job.mkdir(parents=True, exist_ok=True)
    seal_path = CODE_ROOT / "outputs" / "main" / "official_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("config") != "main_ext_met_dt":
        raise SystemExit(f"seal config is {seal.get('config')}, not main_ext_met_dt")
    revision, code = seal["revision"], job / "code"
    common.clone_local(CODE_ROOT, code, revision)
    data = Path(os.environ.get("MTAH_DATA_DIR") or CODE_ROOT / "more-than-a-hack-2027/data").resolve()
    state = {"revision": revision, "started": common.now(), "preflight": common.preflight(CODE_ROOT, data)}
    env = common.offline_env(code, data, code)
    out = "main_ext_met_dt"
    decode = ["tools/decode_tuned.py", "--run", "main_ext_met", "--config", "main_ext_met_dt"]
    steps = [["run.py", "m0", "--config", "main_ext_met"],
             ["run.py", "adapt", "--config", "main_ext_met", "--name", "main_ext_met"]]
    expected = [code / "outputs" / "main_ext_met_lgbm" / "probs.npy", code / "outputs" / "main_ext_met" / "probs.npy"]
    booster = seal.get("booster_in_blend")      # 6 Nov: Metrica-aware blend booster (e.g. ev_ext_mb)
    if booster:
        out = "main_ext_met_dt_mb"
        steps.append(["run.py", "m0", "--config", booster])
        expected.append(code / "outputs" / f"{booster}_lgbm" / "probs.npy")
        decode += ["--booster", booster]
    steps.append(decode + ["--out", out])
    expected.append(code / "outputs" / out / "submission.csv")
    for i, step in enumerate(steps):
        t0 = time.time()
        codes = []
        for attempt in range(2):          # one retry: a step once died instantly with an empty log (5 Nov 00:02)
            with open(job / f"step{i}_try{attempt}.log", "w", encoding="utf-8") as log:
                r = subprocess.run([str(PYTHON), "-u", *step], cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
            codes.append(r.returncode)
            # GPU processes crash on exit (CLAUDE.md, Ek K.7): judge success by the output files, not the exit code
            if expected[i].is_file() and expected[i].stat().st_mtime >= t0:
                break
        state[f"step{i}"] = {"cmd": " ".join(step), "seconds": round(time.time() - t0), "exit_codes": codes,
                             "output_ok": expected[i].is_file() and expected[i].stat().st_mtime >= t0}
        common.write_json(job / "state.json", state)
        print(f"{common.now()} step {i} done ({state[f'step{i}']['seconds']}s)", flush=True)
    sub = code / "outputs" / out / "submission.csv"
    if not sub.is_file():
        raise SystemExit("replay produced no submission; see step logs")
    sha = hashlib.sha256(sub.read_bytes()).hexdigest()
    state.update(sha256=sha, matches_official=sha == seal["sha256"], finished=common.now())
    common.write_json(job / "state.json", state)
    if state["matches_official"]:
        seal["reproduction"] = {"status": "verified_clean_clone", "matches_official": True, "sha256": sha,
                                "job": str(job.relative_to(CODE_ROOT)).replace("\\", "/"), "revision": revision,
                                "finished": state["finished"]}
        common.write_json(seal_path, seal)
    print(f"REPLAY_COMPLETE sha {sha[:16]} matches_official={state['matches_official']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
