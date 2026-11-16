"""EDA section F: is there a referee whistle in the tactical video's audio track?

    uv run python analysis/f_audio.py

Computes, every 0.1 s over the whole match, the share of spectral energy in the 2.5-4.5 kHz band
(where a referee's whistle sits) and a tonality score (peak bin over band median), and caches
both to cache/audio_whistle.npz. Then checks it against moments that are known WITHOUT the
ground truth: the published half boundaries (kickoffs and final whistles).
"""

import os
import sys

import av
import numpy as np
from scipy.signal import stft

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

# Params
HOP_S = 0.1
NPERSEG = 4096
BAND = (2500.0, 4500.0)
CACHE = os.path.join(ROOT, "cache", "audio_whistle.npz")
FPS = 25
# Known without the ground truth (DATA.md / FAQ.md): kickoffs and the ends of both halves.
ANCHORS = {"1st half kickoff": 11185, "1st half end": 90651, "2nd half kickoff": 113742, "2nd half end": 190197}


def compute():
    with av.open(asset_paths()["video"]) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        chunks = [frame.to_ndarray().astype(np.float32).mean(0) for frame in container.decode(stream)]
    signal = np.concatenate(chunks)
    hop = int(HOP_S * rate)
    f, t, z = stft(signal, fs=rate, nperseg=NPERSEG, noverlap=NPERSEG - hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    band = (f >= BAND[0]) & (f < BAND[1])
    ratio = power[band].sum(0) / np.maximum(power.sum(0), 1e-20)
    tonal = power[band].max(0) / np.maximum(np.median(power[band], axis=0), 1e-20)
    peak_hz = f[band][np.argmax(power[band], axis=0)]
    np.savez_compressed(CACHE, t=t.astype(np.float32), ratio=ratio.astype(np.float32),
                        tonal=tonal.astype(np.float32), peak_hz=peak_hz.astype(np.float32), rate=rate)


def main():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    t, ratio, tonal = d["t"], d["ratio"], d["tonal"]
    print("{} windows of {:.1f} s over {:.0f} s of audio".format(len(t), HOP_S, t[-1]))
    print("whistle-band ratio over the whole match: p50 {:.3f}  p90 {:.3f}  p99 {:.3f}  p99.9 {:.3f}".format(
        *np.percentile(ratio, [50, 90, 99, 99.9])))
    print("tonality (peak/median in band):          p50 {:.1f}  p90 {:.1f}  p99 {:.1f}  p99.9 {:.1f}".format(
        *np.percentile(tonal, [50, 90, 99, 99.9])))
    pct_rank = lambda x, arr: 100.0 * (arr < x).mean()
    print("\nGT-free check: the four published half boundaries (max within +/-3 s)")
    for name, frame in ANCHORS.items():
        w = np.abs(t - frame / FPS) <= 3.0
        k = np.flatnonzero(w)[np.argmax(ratio[w])]
        print("   {:<17} frame {:6d}: max ratio {:.3f} (percentile {:5.1f}) at {:+.1f} s, tonality {:.0f}, peak {:.0f} Hz".format(
            name, frame, ratio[k], pct_rank(ratio[k], ratio), t[k] - frame / FPS, tonal[k], d["peak_hz"][k]))
    rng = np.random.RandomState(0)
    rand = rng.uniform(450, 7600, 400)
    mx = [ratio[np.abs(t - r) <= 3.0].max() for r in rand]
    print("   random 6-s windows (n=400): max ratio p50 {:.3f}  p90 {:.3f}  p99 {:.3f}".format(*np.percentile(mx, [50, 90, 99])))


if __name__ == "__main__":
    main()
