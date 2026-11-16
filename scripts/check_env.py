"""Check that this machine can run the pipeline: interpreter, packages, GPU, data bundle, video.

    uv run python scripts/check_env.py            # everything
    uv run python scripts/check_env.py --quick    # skip the video and audio probes

Prints one line per check and exits non-zero if any required check fails. Nothing here writes
anywhere; it is safe to run at any time, and worth running after every `uv sync`.
"""

import argparse
import importlib
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, source_frames  # noqa: E402

# Params
PYTHON = (3, 11)
# Imported by name; the version is reported, not checked -- uv.lock is the authority on versions.
PACKAGES = ("numpy", "pandas", "scipy", "lxml", "kloppy", "floodlight", "cv2", "av",
            "torch", "torchvision", "matplotlib")
# Seconds of audio decoded from the tactical video to tell a real track from a silent one.
AUDIO_PROBE_SECONDS = 60

failures = []


def report(ok, name, detail, required=True):
    print("  {:<4} {:<22} {}".format("ok" if ok else ("FAIL" if required else "warn"), name, detail))
    if not ok and required:
        failures.append(name)


def check_python():
    version = sys.version_info[:2]
    report(version == PYTHON, "python", "{} at {}".format(platform.python_version(), sys.executable))


def check_packages():
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report(True, name, getattr(module, "__version__", "?"))
        except Exception as error:  # an ImportError, or a broken binary extension
            report(False, name, "{}: {}".format(type(error).__name__, error))


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        report(False, "cuda", "torch {} sees no GPU".format(torch.__version__), required=False)
        return
    props = torch.cuda.get_device_properties(0)
    report(True, "cuda", "{}, {:.1f} GB, CUDA {}, cuDNN {}".format(
        props.name, props.total_memory / 2 ** 30, torch.version.cuda,
        torch.backends.cudnn.version()))
    # A real kernel launch, because is_available() can be True with a broken runtime.
    x = torch.ones(1024, device="cuda")
    report(float((x * 2).sum()) == 2048.0, "cuda kernel", "elementwise op on the device")


def check_data():
    for source in ("tactical", "broadcast"):
        for asset, path in sorted(asset_paths(source=source).items()):
            exists = os.path.isfile(path)
            size = "{:.1f} MB".format(os.path.getsize(path) / 2 ** 20) if exists else "missing"
            report(exists, "{}/{}".format(source, asset), size, required=(source == "tactical"))


def check_video():
    import av
    import cv2

    path = asset_paths(source="tactical")["video"]
    capture = cv2.VideoCapture(path)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = capture.read()
    capture.release()
    expected = source_frames("tactical", GAME)
    report(ok and frames == expected, "tactical video", "{} frames (expected {}), first frame {}"
           .format(frames, expected, "decoded" if ok else "NOT decoded"))

    # Is the audio track real? A silent or constant track would rule out whistle detection.
    import numpy as np
    with av.open(path) as container:
        if not container.streams.audio:
            report(False, "tactical audio", "no audio stream", required=False)
            return
        stream = container.streams.audio[0]
        peak, decoded = 0.0, 0.0
        for frame in container.decode(stream):
            samples = frame.to_ndarray().astype(np.float64)
            peak = max(peak, float(np.abs(samples).max()))
            decoded += frame.samples / float(frame.sample_rate)
            if decoded >= AUDIO_PROBE_SECONDS:
                break
        report(peak > 0.0, "tactical audio", "{} {} Hz {} ch, peak {:.4f} over first {:.0f} s"
               .format(stream.codec_context.name, stream.rate, stream.channels, peak, decoded),
               required=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the video and audio probes")
    args = parser.parse_args()

    print("interpreter");  check_python()
    print("packages");     check_packages()
    print("gpu");          check_gpu()
    print("data");         check_data()
    if not args.quick:
        print("video");    check_video()

    print("\n{}".format("all required checks passed" if not failures
                        else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
