"""Kaggle script-kernel runner (generated per job by scripts/kaggle/push.py; JOBS is filled in there).

Inputs (private datasets): mtah-code (git bundle), mtah-ext (DFL, Metrica), mtah-match (BAR-ATH files without the
videos + the submission caches), mtah-video (the two videos), mtah-extra (Drive's cache, models, OOF, SoccerTrack and
SkillCorner caches, detector weights), mtah-local (PC-only caches, PathCRF data). Clones the bundle to /tmp/Hackathon, builds the locked env (uv sync
--frozen, Python 3.11, torch 2.7.1+cu128), links the data, runs each job on its own GPU (CUDA_VISIBLE_DEVICES) and
copies every job's log and outputs to /kaggle/working.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

JOBS = __JOBS__            # [(name, shell command run in the repo root), ...]; job i runs on GPU i % n_gpu
OUTPUTS = __OUTPUTS__      # globs (repo-relative) copied to /kaggle/working/<name>/ after the jobs
W, R = "/kaggle/working", "/tmp/Hackathon"


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def find(name):
    hits = glob.glob("/kaggle/input/**/" + name, recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits, key=len)[0]


t0 = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo no GPU")
sh("pip install -q uv")
bundle = find("mtah.bundle")
sh("git clone -q {} {} && cd {} && git checkout -q p5-env-cu128 && git log --oneline -1".format(bundle, R, R))
sh("cd {} && uv sync --frozen -q".format(R))
data = "/tmp/data"
# every dataset's data/<d>/ and external/<sub>/ files gathered in one tree of symlinks (datasets are read-only)
for d in ("tactical", "broadcast"):
    os.makedirs(os.path.join(data, d), exist_ok=True)
    for f in glob.glob("/kaggle/input/**/BAR-ATH_{}*".format(d), recursive=True):
        dst = os.path.join(data, d, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
os.makedirs(os.path.join(data, "external"), exist_ok=True)
for ext in glob.glob("/kaggle/input/**/external", recursive=True):
    for sub in os.listdir(ext):
        dst = os.path.join(data, "external", sub)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ext, sub), dst)
for d in ("tactical", "broadcast", "external"):
    os.symlink(os.path.join(data, d), os.path.join(R, "data", d))
os.makedirs(os.path.join(R, "cache"), exist_ok=True)
os.makedirs(os.path.join(R, "models"), exist_ok=True)
for c in glob.glob("/kaggle/input/**/cache", recursive=True):
    sh("cp -rn {}/. {}/cache/".format(c, R))
for c in ("st_cache", "sc_cache"):
    for src in glob.glob("/kaggle/input/**/" + c, recursive=True):
        sh("cp -rn {}/. {}/cache/".format(src, R))
for f in glob.glob("/kaggle/input/**/models/*.pt", recursive=True) + glob.glob("/kaggle/input/**/pkg/*.pt", recursive=True):
    if not os.path.exists(os.path.join(R, "models", os.path.basename(f))):
        shutil.copy(f, os.path.join(R, "models"))
for o in glob.glob("/kaggle/input/**/oof", recursive=True):
    sh("mkdir -p {0}/cache/oof && cp -rn {1}/. {0}/cache/oof/".format(R, o))
sh("touch {}/cache/frames_v1.parquet".format(R))      # newest, so load_frames never rebuilds from absent videos
print("setup {:.0f} s".format(time.time() - t0), flush=True)

n_gpu = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1))
env = dict(os.environ, MTAH_DATA_DIR=data, OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
procs = []
for i, (name, cmd) in enumerate(JOBS):
    os.makedirs(os.path.join(W, name), exist_ok=True)
    log = open(os.path.join(W, name, "log.txt"), "w")
    e = dict(env, CUDA_VISIBLE_DEVICES=str(i % n_gpu))
    procs.append((name, subprocess.Popen(cmd, shell=True, cwd=R, env=e, stdout=log, stderr=subprocess.STDOUT)))
for name, p in procs:
    rc = p.wait()
    print(name, "rc", rc, "{:.0f} s".format(time.time() - t0), flush=True)
    for g in OUTPUTS:
        for f in glob.glob(os.path.join(R, g)):
            shutil.copy(f, os.path.join(W, name))
print("done {:.0f} s".format(time.time() - t0), flush=True)
