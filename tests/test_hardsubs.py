"""Burned-in subtitle detection on synthetic anime-like frames."""

import os
import subprocess
import shutil

import cv2
import numpy as np
import pytest

import hardsubs


def _scene(w=1280, h=720):
    """Flat colours and black line art with white glints: what anime looks like
    to a threshold, and the false positives the filters exist for."""
    f = np.full((h, w, 3), (170, 140, 90), np.uint8)
    cv2.rectangle(f, (0, int(h * 0.7)), (w, h), (60, 120, 70), -1)
    cv2.circle(f, (w // 2, int(h * 0.55)), 120, (200, 210, 240), -1)
    cv2.circle(f, (w // 2, int(h * 0.55)), 120, (0, 0, 0), 4)
    # A white glint wrapped in line art, low in the frame.
    cv2.ellipse(f, (300, int(h * 0.85)), (30, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(f, (300, int(h * 0.85)), (30, 12), 0, 0, 360, (0, 0, 0), 3)
    # A large white area (a shirt) in the band.
    cv2.rectangle(f, (1100, int(h * 0.65)), (w, h), (250, 250, 250), -1)
    return f


def _subtitle(f, text="I never thought we'd meet again", y_frac=0.92):
    # White fill with a black border grown from the glyphs themselves, the way
    # an ASS renderer draws it (putText's own thickness varies by build).
    h, w = f.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, h / 720 * 1.1
    (tw, _), _ = cv2.getTextSize(text, font, scale, 2)
    org = ((w - tw) // 2, int(h * y_frac))
    glyphs = np.zeros((h, w), np.uint8)
    cv2.putText(glyphs, text, org, font, scale, 255, 2, cv2.LINE_8)
    border = cv2.dilate(glyphs, np.ones((5, 5), np.uint8))
    f[border > 0] = 0
    f[glyphs > 0] = 255
    return f


def test_mask_finds_subtitle_line():
    f = _subtitle(_scene())
    rows = np.flatnonzero(hardsubs.text_mask(f).any(axis=1))
    assert rows.size
    assert rows[0] > 0.8 * f.shape[0]


def test_mask_ignores_line_art_glints_and_white_areas():
    assert not hardsubs.text_mask(_scene()).any()


def test_mask_ignores_text_above_the_search_band():
    # A sign in the upper half is part of the scene, not a subtitle.
    f = _subtitle(_scene(), y_frac=0.2)
    assert not hardsubs.text_mask(f).any()


def _write_clip(path, frames, fps=12):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()


def test_detect_band_on_subtitled_and_clean_clips(tmp_path):
    subbed = str(tmp_path / "subbed.mp4")
    clean = str(tmp_path / "clean.mp4")
    _write_clip(subbed, [_subtitle(_scene()) for _ in range(24)])
    _write_clip(clean, [_scene() for _ in range(24)])

    band = hardsubs.detect_band(subbed)
    assert hardsubs.MIN_BAND <= band <= 0.2
    assert hardsubs.detect_band(clean) == 0.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_crop_mode_removes_the_band(tmp_path, monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    path = str(tmp_path / "clip.mp4")
    _write_clip(path, [_subtitle(_scene()) for _ in range(24)])
    assert hardsubs.clean(path, "crop")
    _, _, w, h = hardsubs._probe(path)
    assert w == 1280 and h < 720 * 0.9
    assert hardsubs.detect_band(path) == 0.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_inpaint_mode_erases_text_and_keeps_size(tmp_path, monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    path = str(tmp_path / "clip.mp4")
    _write_clip(path, [_subtitle(_scene()) for _ in range(24)])
    assert hardsubs.clean(path, "inpaint")
    _, _, w, h = hardsubs._probe(path)
    assert (w, h) == (1280, 720)
    assert hardsubs.detect_band(path) == 0.0
    # The detector going quiet is not enough: leftover letter fragments fail
    # it too. The subtitle rows must hold (almost) no white pixels at all.
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    rows = frame[int(720 * 0.85):int(720 * 0.95), 340:1090]
    assert (cv2.cvtColor(rows, cv2.COLOR_BGR2GRAY) > 200).mean() < 0.002


def test_off_mode_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("HARDSUBS", raising=False)
    path = str(tmp_path / "missing.mp4")
    assert hardsubs.clean(path) is False
    assert not os.path.exists(path)


def test_keep_mode_goes_wide_only_where_subtitles_show(tmp_path):
    path = str(tmp_path / "clip.mp4")
    # Scene 0: dialogue (subtitled); scene 1: nobody talking.
    _write_clip(path, [_subtitle(_scene()) for _ in range(24)] +
                [_scene() for _ in range(24)])
    assert hardsubs.keep_strategies(path, [(0, 24), (24, 48)], 12) == ['WIDE', 'TRACK']


def test_keep_mode_never_rewrites_the_clip(tmp_path):
    path = str(tmp_path / "clip.mp4")
    _write_clip(path, [_subtitle(_scene()) for _ in range(12)])
    before = os.path.getmtime(path)
    assert hardsubs.clean(path, "keep") is False
    assert os.path.getmtime(path) == before
