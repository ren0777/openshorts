"""Remove subtitles burned into the source picture before the vertical crop.

A fansub / simulcast rip of an anime carries its subtitles in the pixels. The
9:16 crop keeps the middle third of the width, so those lines come out cut on
both sides ("...ou think you're doi...") and our own captions then land on top
of them. This runs on the cut clip, before the reframe, so every layout sees a
clean picture and the captions we burn later are the only text in the frame.

Three modes, chosen per job (``HARDSUBS``):

- ``crop``: drop the band the subtitles live in. No artifacts and fast (one
  ffmpeg crop), at the price of the bottom ~15-25% of the picture. The band is
  measured per clip; a clip with no subtitles found is left untouched.
- ``inpaint``: keep the full picture and paint the glyphs out frame by frame
  (OpenCV Telea). Fine on the flat colours of anime, slower (~1 min per clip
  on CPU), and a soft smudge is visible where the text was.
- ``keep``: leave the original subtitles alone and frame around them instead.
  Scenes with subtitles on screen render WIDE (the whole 16:9 frame over its
  own blurred copy, so every line stays whole); scenes without fill the 9:16
  frame with the normal tracking crop. Our own captions are not burned.

Detection is geometric, no OCR: subtitle glyphs are bright (white / yellow)
strokes that are **thin**, wrapped in a **dark outline**, and sit in a
**row of several glyphs** at the bottom of the frame. Each filter removes a
class of false positive in anime: thickness drops highlights and skies, the
outline drops plain bright edges, and the row drops the lone white glint
inside black line art that passes the other two.
"""

import os
import subprocess

import cv2
import numpy as np

from ffmpeg_utils import video_encode_args, QUALITY_FAST

MODES = ("off", "crop", "inpaint", "keep")

# Subtitles sit in the bottom part of the frame; nothing above this is looked
# at, so on-screen signs and typesetting higher up are never touched.
SEARCH_FRACTION = 0.40
# Bounds for the cropped band: a single small line still costs at least this
# much, and a misdetection can never eat more than this of the picture.
MIN_BAND, MAX_BAND = 0.08, 0.35
SAMPLE_FRAMES = 24
# Fraction of sampled frames that must show subtitle text before a clip counts
# as subtitled. Dialogue-free stretches are common; noise in one frame is not.
MIN_HIT_FRACTION = 0.15


def mode_from_env():
    mode = os.environ.get("HARDSUBS", "off").strip().lower()
    return mode if mode in MODES else "off"


def text_mask(frame):
    """uint8 mask (0/255) of subtitle-glyph pixels in a BGR frame.

    Rows above the search band are always zero.
    """
    h, w = frame.shape[:2]
    top = int(h * (1 - SEARCH_FRACTION))
    roi = frame[top:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    unit = max(h / 1080.0, 0.25)  # kernels scale with resolution
    k = lambda px: cv2.getStructuringElement(  # noqa: E731
        cv2.MORPH_ELLIPSE, (max(int(px * unit) | 1, 3),) * 2)

    bright = (gray > 185).astype(np.uint8)
    dark = (gray < 60).astype(np.uint8)
    # Strokes of subtitle glyphs are a few px thick; anything that survives a
    # wide erosion is a bright area (sky, highlight, white shirt), not text.
    thick = cv2.dilate(cv2.erode(bright, k(13)), k(17))
    thin = bright & (1 - thick)

    # Glyphs: join the strokes of each letter into one blob, then judge the
    # blob by the ring just outside it. A subtitle letter is wrapped in its
    # border, so most of that ring is dark; a bright edge in the picture
    # touches line art only along part of its outline.
    blobs = cv2.dilate(thin, k(5))
    ring = cv2.dilate(blobs, k(7)) & (1 - blobs)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    ring_labels = cv2.dilate(labels.astype(np.float32), k(7)).astype(np.int32)
    ring_total = np.bincount(ring_labels[ring > 0], minlength=n)
    ring_dark = np.bincount(ring_labels[(ring & dark) > 0], minlength=n)
    glyphs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (0.015 * h <= bh <= 0.09 * h and bw <= 0.2 * w and area >= 12 * unit):
            continue
        if ring_total[i] and ring_dark[i] / ring_total[i] >= 0.6:
            glyphs.append((i, y + bh / 2.0, bh))

    # A subtitle line is several glyphs on one baseline.
    keep = []
    for i, cy, bh in glyphs:
        same_row = sum(1 for _, oy, _ in glyphs if abs(oy - cy) <= 0.5 * bh)
        if same_row >= 4:  # counts itself
            keep.append(i)

    out = np.zeros((h, w), np.uint8)
    if keep:
        # Once a line is confirmed, take every thin bright pixel in its box:
        # punctuation, accents and a lone "I" are too small to pass the glyph
        # filters on their own and would be left floating after the erase.
        boxes = np.zeros_like(thin)
        for i in keep:
            x, y, bw, bh, _ = stats[i]
            pad = bh // 2
            boxes[max(y - pad, 0):y + bh + pad, max(x - 2 * pad, 0):x + bw + 2 * pad] = 1
        out[top:][(boxes & thin) > 0] = 255
    return out


def _probe(path):
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n, fps, w, h


def detect_band(path, samples=SAMPLE_FRAMES):
    """Fraction of the frame height, from the bottom, that holds the subtitles.

    Returns 0.0 when the clip does not look subtitled.
    """
    n, _, _, h = _probe(path)
    if n <= 0 or h <= 0:
        return 0.0
    cap = cv2.VideoCapture(path)
    tops = []
    hits = 0
    try:
        for idx in np.linspace(0, n - 1, num=min(samples, n)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            rows = np.flatnonzero(text_mask(frame).any(axis=1))
            if rows.size:
                hits += 1
                tops.append(rows[0])
    finally:
        cap.release()
    if hits < max(2, MIN_HIT_FRACTION * samples):
        return 0.0
    # The highest line seen (two-line subtitles) plus room for its outline.
    top = float(np.percentile(tops, 10))
    band = (h - top) / h + 0.02
    return float(min(max(band, MIN_BAND), MAX_BAND))


def _replace(src, tmp):
    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, src)
        return True
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def crop_band(path, band):
    """Re-encode ``path`` in place without its bottom ``band`` of height."""
    tmp = path + ".nosubs.mp4"
    # Even height for yuv420p; the crop is anchored at the top.
    vf = f"crop=iw:trunc(ih*{1 - band:.4f}/2)*2:0:0"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", vf,
           *video_encode_args(QUALITY_FAST), "-c:a", "copy", tmp]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True, errors="replace")
    if r.returncode != 0:
        print(f"   ⚠️ [hardsubs] crop failed: {(r.stderr or '').strip()[-300:]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    return _replace(path, tmp)


def inpaint(path):
    """Paint the subtitle glyphs out of every frame, in place."""
    _, fps, w, h = _probe(path)
    if not fps or not w or not h:
        return False
    tmp = path + ".nosubs.mp4"
    top = int(h * (1 - SEARCH_FRACTION))
    unit = max(h / 1080.0, 0.25)
    # Grow the glyph mask over its outline and drop shadow.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(9 * unit) | 1, 3),) * 2)

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
         "-i", "-", "-i", path, "-map", "0:v", "-map", "1:a?",
         *video_encode_args(QUALITY_FAST), "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-shortest", tmp],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(path)
    prev = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = cv2.dilate(text_mask(frame), grow)
            # A glyph missed in one frame would flash back for 1/24 s; OR-ing
            # the previous mask holds the erase across that frame.
            both = mask if prev is None else cv2.bitwise_or(mask, prev)
            prev = mask
            if both[top:].any():
                frame[top:] = cv2.inpaint(frame[top:], both[top:], 5, cv2.INPAINT_TELEA)
            enc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            enc.stdin.close()
        except OSError:
            pass
        err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
        enc.wait()
    if enc.returncode != 0:
        print(f"   ⚠️ [hardsubs] inpaint encode failed: {err.strip()[-300:]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    return _replace(path, tmp)


def clean(path, mode=None):
    """Apply the job's hardsub mode to a cut clip, in place. Never raises: a
    failure leaves the clip as it was, subtitles and all."""
    mode = mode or mode_from_env()
    if mode not in ("crop", "inpaint"):  # "keep" frames around them instead
        return False
    try:
        band = detect_band(path)
        if band <= 0:
            print("   [hardsubs] no burned-in subtitles found; clip left as is")
            return False
        if mode == "crop":
            print(f"   ✂️ [hardsubs] cropping the bottom {band:.0%} (subtitle band)")
            return crop_band(path, band)
        print("   🧽 [hardsubs] erasing burned-in subtitles")
        return inpaint(path)
    except Exception as e:  # never cost the clip over this
        print(f"   ⚠️ [hardsubs] skipped: {type(e).__name__}: {e}")
        return False


def scene_has_subs(path, start_f, end_f, fps):
    """True when subtitles show up anywhere in frames [start_f, end_f).

    One sample every half second: a line stays up at least ~1 s, so a scene
    that has one is caught, and the whole scene then keeps the full width
    rather than switching layout under the line.
    """
    count = max(end_f - start_f, 0)
    if count == 0:
        return False
    step = max(int((fps or 24) / 2), 1)
    idxs = list(range(start_f, end_f, step))[:40]
    # A single hit is enough on a short scene; a long one needs two so one
    # stray false positive does not cost it the full-screen crop.
    need = 1 if len(idxs) <= 4 else 2
    cap = cv2.VideoCapture(path)
    hits = 0
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and text_mask(frame).any():
                hits += 1
                if hits >= need:
                    return True
    finally:
        cap.release()
    return False


def keep_strategies(path, scene_boundaries, fps):
    """Per-scene layouts for ``keep``: WIDE where subtitles are on screen,
    TRACK (fill the frame) where they are not."""
    out = []
    for start_f, end_f in scene_boundaries:
        try:
            subbed = scene_has_subs(path, start_f, end_f, fps)
        except Exception as e:
            print(f"   ⚠️ [hardsubs] scene check failed ({e}); keeping it wide")
            subbed = True  # cutting a line in half is the failure we avoid
        out.append('WIDE' if subbed else 'TRACK')
    return out
