"""Check that the project environment and the match data are ready. Run inside the project venv.

    python kurulum/check_env.py              # environment + data layout + alignment + scorer smoke test
    python kurulum/check_env.py --video      # + video frame counts from the container (fast, no decode)
    python kurulum/check_env.py --decode     # + full decode of the tactical video with ffmpeg (minutes)
    python kurulum/check_env.py --install-ffmpeg   # put the bundled ffmpeg binary on the venv PATH

Exit code 0 when every check passes, 1 otherwise. Output is ASCII only so it prints on any console.
"""

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "more-than-a-hack-2027"
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, str(REPO / sub))

PYTHON_WANTED = (3, 11)
PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "kloppy", "matplotlib",
            "notebook", "ipykernel", "av", "imageio_ffmpeg", "tqdm", "lightgbm", "catboost", "numba",
            "yaml", "pytest"]
EXPECTED_FRAMES = {"tactical": 199319, "broadcast": 164488}
# The delivered broadcast video has one frame fewer than its tracking and homography (164,487 vs
# 164,488 records), measured on 21 Oct 2026. Broadcast is optional and not aligned to the
# submission, so this is reported as a warning rather than a failure.
KNOWN_VIDEO_FRAMES = {"broadcast": 164487}
# Published by the organisers: the four committed baselines on the public 5-minute window.
EXPECTED_PUBLIC_SPELLF1 = {"baseline_smooth50": 0.3256, "baseline_smooth125": 0.4242}

results = []


def report(ok, name, detail="", warn=False):
    """Print one check. warn=True marks a known, non-blocking deviation: shown, not counted."""
    if warn and not ok:
        print("  [WARN] {:<34} {}".format(name, detail))
        return
    results.append(ok)
    print("  [{}] {:<34} {}".format("OK  " if ok else "FAIL", name, detail))


def section(title):
    print("\n" + title)


# -- environment -------------------------------------------------------------------------------

def check_python():
    version = sys.version_info
    report(version[:2] == PYTHON_WANTED, "python {}.{}".format(*PYTHON_WANTED),
           "{}.{}.{}  ({})".format(version[0], version[1], version[2], sys.executable))


def check_packages():
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report(True, name, getattr(module, "__version__", ""))
        except Exception as error:  # ImportError or a broken binary wheel
            report(False, name, "{}: {}".format(type(error).__name__, error))


def venv_ffmpeg():
    """The ffmpeg that --install-ffmpeg puts next to this interpreter, if it is there."""
    path = Path(sysconfig.get_path("scripts")) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return str(path) if path.is_file() else None


def ffmpeg_binary():
    found = venv_ffmpeg() or shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def check_ffmpeg():
    command = venv_ffmpeg() or shutil.which("ffmpeg")
    if not command:
        report(False, "ffmpeg command", "not found; run: python kurulum/check_env.py --install-ffmpeg")
        return
    output = subprocess.run([command, "-version"], capture_output=True, text=True).stdout
    version = output.splitlines()[0] if output else "?"
    report(bool(output), "ffmpeg command", "{}  ({})".format(version[:60], command))


def install_ffmpeg():
    """Copy imageio-ffmpeg's static binary into the venv's scripts folder as 'ffmpeg'."""
    import imageio_ffmpeg
    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    scripts = Path(sysconfig.get_path("scripts"))
    target = scripts / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    shutil.copy2(source, target)
    if os.name != "nt":
        target.chmod(0o755)
    print("ffmpeg installed: {}  (from {})".format(target, source.name))


# -- data --------------------------------------------------------------------------------------

def check_data_layout():
    import games
    print("  data dir: {}{}".format(games.DATA_DIR,
                                    "   (MTAH_DATA_DIR)" if os.environ.get("MTAH_DATA_DIR") else ""))
    for source in ("tactical", "broadcast"):
        for asset, path in sorted(games.asset_paths(source=source).items()):
            exists = os.path.isfile(path)
            size = "{:.2f} GB".format(os.path.getsize(path) / 1e9) if exists else "missing"
            report(exists, "{}/{}".format(source, os.path.basename(path)), size)


def check_alignment():
    import load_atd
    import load_homography
    import load_smart_tagging

    for source in ("tactical", "broadcast"):
        try:
            atd = load_atd.check_alignment_raw(source=source)
            report(atd["agrees"], "ATD alignment ({})".format(source),
                   "{} records, frame_id {}..{}".format(atd["n_records"], *atd["frame_id_range"]))
        except Exception as error:
            report(False, "ATD alignment ({})".format(source), str(error).splitlines()[0])
        try:
            hom = load_homography.check_alignment(source=source)
            report(hom["agrees"], "homography alignment ({})".format(source),
                   "{} rows".format(hom["n_rows"]))
        except Exception as error:
            report(False, "homography alignment ({})".format(source), str(error).splitlines()[0])
    try:
        tags = load_smart_tagging.check_time_base(load_smart_tagging.load())
        report(tags["agrees"], "smart tagging time base",
               "{} kick-off tags contain the real kick-offs".format(tags["n_kickoff_tags"]))
    except Exception as error:
        report(False, "smart tagging time base", str(error).splitlines()[0])


def check_kloppy():
    """kloppy is pinned by the official repo; make sure it reads our files with this pandas."""
    import games
    from kloppy import metrica, sportscode
    paths = games.asset_paths()
    try:
        dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                             limit=300)
        frame = dataset.to_df()
        report(len(frame) == 300 and "ball_x" in frame.columns, "kloppy EPTS tracking",
               "{} rows x {} cols, teams {}".format(len(frame), frame.shape[1],
                                                     [str(t) for t in dataset.metadata.teams]))
    except Exception as error:
        report(False, "kloppy EPTS tracking", "{}: {}".format(type(error).__name__, error))
    try:
        events = sportscode.load(paths["smart_tagging"]).to_df()
        report(len(events) == 638 and "Team" in events.columns, "kloppy sportscode",
               "{} instances".format(len(events)))
    except Exception as error:
        report(False, "kloppy sportscode", "{}: {}".format(type(error).__name__, error))


def check_scorer():
    import possession_metrics as pm
    gt_path = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt = pm.label_codes(pm.read_label_csv(str(gt_path)))
    for name, expected in EXPECTED_PUBLIC_SPELLF1.items():
        path = REPO / "evaluation" / "results" / (name + ".csv")
        score = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(str(path))))["spell_f1_mean"]
        report(abs(score - expected) < 5e-5, "scorer {}".format(name),
               "spellF1 {:.4f} (published {:.4f})".format(score, expected))


# -- video -------------------------------------------------------------------------------------

def check_video_container():
    """Frame count from the mp4 header and from demuxing every packet. No decoding."""
    import av
    import games
    for source, expected in EXPECTED_FRAMES.items():
        path = games.asset_paths(source=source)["video"]
        if not os.path.isfile(path):
            report(False, "video {}".format(source), "missing")
            continue
        start = time.time()
        with av.open(path) as container:
            stream = container.streams.video[0]
            header = stream.frames
            codec = stream.codec_context
            info = "{} {}x{} @ {} fps".format(codec.name, codec.width, codec.height,
                                              float(stream.average_rate))
            packets = sum(1 for packet in container.demux(stream) if packet.size)
        # A count equal to the documented known deviation is a warning, anything else a failure.
        known = KNOWN_VIDEO_FRAMES.get(source)
        note = "" if known is None else "; known: delivered file has {}".format(known)
        report(header == expected, "video {} header".format(source),
               "{} frames (expected {}{}), {}".format(header, expected, note, info),
               warn=header == known)
        report(packets == expected, "video {} packets".format(source),
               "{} packets ({:.0f} s)".format(packets, time.time() - start),
               warn=packets == known)


def check_video_decode():
    """Decode every frame of the tactical video with ffmpeg and count them. Takes minutes."""
    import games
    path = games.asset_paths(source="tactical")["video"]
    binary = ffmpeg_binary()
    start = time.time()
    run = subprocess.run([binary, "-hide_banner", "-nostats", "-threads", "0", "-i", path,
                          "-map", "0:v:0", "-f", "null", "-progress", "pipe:1", "-"],
                         capture_output=True, text=True)
    frames = re.findall(r"^frame=(\d+)", run.stdout, flags=re.MULTILINE)
    decoded = int(frames[-1]) if frames else -1
    expected = EXPECTED_FRAMES["tactical"]
    report(decoded == expected, "video tactical decoded frames",
           "{} (expected {}), {:.0f} s".format(decoded, expected, time.time() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", action="store_true", help="count video frames (no decode)")
    parser.add_argument("--decode", action="store_true", help="also fully decode the tactical video")
    parser.add_argument("--install-ffmpeg", action="store_true",
                        help="copy the bundled ffmpeg into the venv and exit")
    args = parser.parse_args()

    if args.install_ffmpeg:
        install_ffmpeg()
        return 0

    section("Environment")
    check_python()
    check_packages()
    check_ffmpeg()

    section("Data layout")
    check_data_layout()

    section("Alignment (official starter checks)")
    check_alignment()

    section("kloppy")
    check_kloppy()

    section("Scorer smoke test (public 5-minute ground truth)")
    check_scorer()

    if args.video or args.decode:
        section("Video")
        check_video_container()
    if args.decode:
        check_video_decode()

    failed = results.count(False)
    print("\n{} checks, {} failed.".format(len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
