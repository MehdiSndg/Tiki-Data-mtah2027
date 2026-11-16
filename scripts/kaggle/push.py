"""Push one Kaggle script kernel that runs the given jobs (see run_template.py) and print how to fetch its output.

    python scripts/kaggle/push.py <slug> '<name>=<command>' ['<name>=<command>' ...] [--out <glob> ...] [--cpu] [--sources code,ext]

--sources attaches only the named datasets (e.g. code,ext: no BAR-ATH file reaches the kernel; 14 Nov).

The code dataset (mehdinda/mtah-code) must hold the bundle of the commit to run: python scripts/kaggle/push.py --code
first updates it with the current branch.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
USER = "mehdinda"
SOURCES = ["{}/mtah-{}".format(USER, n) for n in ("code", "ext", "match", "video", "extra", "local", "sub")]


def push_code():
    d = tempfile.mkdtemp()
    subprocess.run(["git", "bundle", "create", os.path.join(d, "mtah.bundle"), "p5-env-cu128"], cwd=ROOT, check=True, capture_output=True)
    json.dump({"title": "mtah-code", "id": "{}/mtah-code".format(USER), "licenses": [{"name": "other"}]}, open(os.path.join(d, "dataset-metadata.json"), "w"))
    head = subprocess.run(["git", "log", "--oneline", "-1"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    subprocess.run(["kaggle", "datasets", "version", "-p", d, "-m", head, "-q"], check=True)
    print("code dataset ->", head)


def main(argv):
    if argv[:1] == ["--code"]:
        return push_code()
    slug, rest = argv[0], argv[1:]
    cpu = "--cpu" in rest
    sources = SOURCES
    outs, jobs, it = [], [], iter([a for a in rest if a != "--cpu"])
    for a in it:
        if a == "--out":
            outs.append(next(it))
        elif a == "--sources":
            sources = ["{}/mtah-{}".format(USER, n) for n in next(it).split(",")]
        else:
            n, c = a.split("=", 1)
            jobs.append((n, c))
    d = tempfile.mkdtemp()
    src = open(os.path.join(HERE, "run_template.py")).read().replace("__JOBS__", repr(jobs)).replace("__OUTPUTS__", repr(outs))
    open(os.path.join(d, "run.py"), "w").write(src)
    meta = {"id": "{}/{}".format(USER, slug), "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": not cpu, "enable_internet": True, "dataset_sources": sources,
            "competition_sources": [], "kernel_sources": []}
    if not cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    subprocess.run(["kaggle", "kernels", "push", "-p", d], check=True)
    print("status: kaggle kernels status {0}/{1}\noutput: kaggle kernels output {0}/{1} -p <dir>".format(USER, slug))


if __name__ == "__main__":
    main(sys.argv[1:])
