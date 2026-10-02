"""Per-job pitch shift: applied once, on the cut from the source, tempo kept."""
import shutil
import subprocess

import numpy as np
import pytest

import ffmpeg_utils as fu


def test_no_shift_by_default(monkeypatch):
    monkeypatch.delenv("AUDIO_PITCH_SEMITONES", raising=False)
    assert fu.pitch_filter() is None
    assert "rubberband" not in " ".join(fu.audio_encode_args(pitch=True))


def test_shift_only_where_asked(monkeypatch):
    monkeypatch.setenv("AUDIO_PITCH_SEMITONES", "2")
    assert "rubberband=pitch=1.12246" in " ".join(fu.audio_encode_args(pitch=True))
    # Later re-encodes of an already shifted clip must not stack it.
    assert "rubberband" not in " ".join(fu.audio_encode_args())


def test_clamped_and_garbage_ignored(monkeypatch):
    monkeypatch.setenv("AUDIO_PITCH_SEMITONES", "40")
    assert fu.pitch_filter() == f"rubberband=pitch={2 ** 0.5:.5f}"
    monkeypatch.setenv("AUDIO_PITCH_SEMITONES", "loud")
    assert fu.pitch_filter() is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_cut_raises_pitch_and_keeps_length(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_PITCH_SEMITONES", "2")
    monkeypatch.setenv("AUDIO_NORMALIZE", "0")
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    src, out = str(tmp_path / "src.mp4"), str(tmp_path / "out.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=24:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", src], check=True)
    fu.cut_clip(src, out, 0.5, 3.5, 1)

    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", out, "-f", "f32le",
                          "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(pcm, np.float32)
    assert abs(len(a) / 48000 - 3.0) < 0.1          # same length: tempo untouched
    seg = a[len(a) // 4: len(a) // 4 + 32768]
    peak = np.argmax(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))))
    assert abs(peak * 48000 / len(seg) - 440 * 2 ** (2 / 12)) < 5   # ~493.9 Hz
