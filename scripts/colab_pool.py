"""Colab job pool, a background process: keeps up to MAX analysis jobs running at once (the RTX PRO 6000 saturates at
5-6 of ours, 27 Oct), starting the next *.sh in Drive mtah2027/queue2 (name order) whenever fewer are running. Running
jobs are counted from /proc (".venv/bin/python analysis/"), so jobs started by hand count too. Each job runs detached,
its output to queue2/done/<job>.out; the pool's status line is in /content/pool.status.

    setsid nohup python3 /content/drive/MyDrive/mtah2027/pkg/colab_pool.py > /content/pool.log 2>&1 < /dev/null &
"""
import datetime
import glob
import os
import shutil
import subprocess
import time

Q = "/content/drive/MyDrive/mtah2027/queue2"
MAX = int(os.environ.get("POOL_MAX", 6))
os.makedirs(Q + "/done", exist_ok=True)


def running():
    n = 0
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(p, "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        n += ".venv/bin/python analysis/" in cmd and "bash -c" not in cmd
    return n


while True:
    n = running()
    jobs = sorted(glob.glob(Q + "/*.sh"))
    open("/content/pool.status", "w").write("{} running {} queued {}\n".format(datetime.datetime.now().strftime("%H:%M:%S"), n, len(jobs)))
    if jobs and n < MAX:
        name = os.path.basename(jobs[0])
        job = shutil.move(jobs[0], Q + "/done/" + name)
        out = open(Q + "/done/" + name + ".out", "w")
        subprocess.Popen(["bash", job], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(60)                       # let it start its python before counting again
        continue
    time.sleep(30)
