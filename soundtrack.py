"""Bande-son d'ambiance synthétisée (nappes + arpèges, progression Am-F-C-G)."""

import math
import wave

import numpy as np

SR = 44100

CHORDS = [
    ("Am", [110.00, 220.00, 261.63, 329.63]),
    ("F",  [87.31, 174.61, 220.00, 261.63]),
    ("C",  [130.81, 261.63, 329.63, 392.00]),
    ("G",  [98.00, 196.00, 246.94, 293.66]),
]


def chord_at(t, total):
    idx = min(int(t / total * len(CHORDS) * 1.0), len(CHORDS) - 1) if total > 0 else 0
    return CHORDS[idx]


def env_adsr(n, attack, release, sustain=1.0):
    e = np.full(n, sustain, dtype=np.float64)
    a = min(int(attack * SR), n)
    r = min(int(release * SR), n)
    if a > 0:
        e[:a] *= np.linspace(0, 1, a)
    if r > 0:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def make_soundtrack(total_sec, path="soundtrack.wav"):
    n = int(total_sec * SR)
    t = np.arange(n) / SR
    pad = np.zeros(n)
    bass = np.zeros(n)

    seg = total_sec / len(CHORDS)
    for i, (_, freqs) in enumerate(CHORDS):
        s0, s1 = i * seg, (i + 1) * seg
        i0, i1 = int(s0 * SR), min(int((s1 + 2.5) * SR), n)
        m = i1 - i0
        if m <= 0:
            continue
        e = env_adsr(m, 1.8, 2.2, 0.32)
        seg_pad = np.zeros(m)
        for f in freqs:
            for det, amp in ((0.9985, 0.5), (1.0, 1.0), (1.0015, 0.5)):
                seg_pad += amp * np.sin(2 * np.pi * f * det * t[i0:i1] + i * 0.7)
        pad[i0:i1] += seg_pad * e / (3 * len(freqs))

        fr = freqs[0] / 2
        eb = env_adsr(m, 1.2, 2.5, 0.5)
        bass[i0:i1] += 0.9 * np.sin(2 * np.pi * fr * t[i0:i1]) * eb

    eighth = 60 / 72 / 2
    arp = np.zeros(n)
    k = 0
    tt = 0.0
    while tt < total_sec - 0.5:
        _, freqs = chord_at(tt, total_sec)
        f = freqs[k % len(freqs)] * 2
        i0 = int(tt * SR)
        dur = 0.5
        m = min(int(dur * SR), n - i0)
        if m <= 0:
            break
        decay = np.exp(-t[:m] * 7.0)
        attack = env_adsr(m, 0.006, 0.05, 1.0)
        note = np.sin(2 * np.pi * f * t[i0:i0 + m]) * decay * attack
        arp[i0:i0 + m] += note * (0.9 if k % 2 == 0 else 0.65)
        tt += eighth
        k += 1

    mono = 0.9 * pad + 0.8 * bass + 0.42 * arp

    fade_in = int(2.0 * SR)
    fade_out = int(4.0 * SR)
    gain = np.ones(n)
    gain[:fade_in] = np.linspace(0, 1, fade_in)
    gain[-fade_out:] = np.linspace(1, 0, fade_out)
    mono *= gain

    mono /= np.max(np.abs(mono)) + 1e-9
    mono *= 0.30

    wobble = 0.015 * np.sin(2 * np.pi * 0.11 * t)
    left = mono * (1 - wobble)
    right = mono * (1 + wobble)

    pcm = np.stack([left, right], axis=1)
    pcm = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    import sys
    total = float(sys.argv[1]) if len(sys.argv) > 1 else 84.375
    make_soundtrack(total)
    print("soundtrack.wav ok")
