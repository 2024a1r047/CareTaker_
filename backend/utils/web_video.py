"""
Make analysed videos playable in the browser.

The pipeline writes an XVID .avi (reliable on Windows) and used to need the `ffmpeg`
command to convert it to H.264 MP4. When ffmpeg is not installed the raw XVID file was
just renamed to .mp4, which browsers cannot play (the player stayed at 0:00).

to_web_mp4():
  1. uses ffmpeg if it is on PATH;
  2. otherwise re-encodes with OpenCV's built-in H.264 encoder ('avc1');
  3. then moves the MP4 index ("moov" atom) to the front ("faststart") so the browser can
     start playing before the whole file is downloaded.
"""
import os
import shutil
import struct
import subprocess
import logging

logger = logging.getLogger(__name__)

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"mvex"}


def _atoms(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            break
        yield i, size, typ, hdr
        i += size


def _patch_offsets(moov: bytearray, start: int, end: int, delta: int) -> None:
    for pos, size, typ, hdr in list(_atoms(bytes(moov), start, end)):
        body = pos + hdr
        if typ in _CONTAINERS:
            _patch_offsets(moov, body, pos + size, delta)
        elif typ == b"stco":
            count = struct.unpack(">I", moov[body + 4:body + 8])[0]
            for k in range(count):
                o = body + 8 + 4 * k
                moov[o:o + 4] = struct.pack(">I", struct.unpack(">I", moov[o:o + 4])[0] + delta)
        elif typ == b"co64":
            count = struct.unpack(">I", moov[body + 4:body + 8])[0]
            for k in range(count):
                o = body + 8 + 8 * k
                moov[o:o + 8] = struct.pack(">Q", struct.unpack(">Q", moov[o:o + 8])[0] + delta)


def faststart(path: str) -> bool:
    """Move the moov atom before mdat (in place). Returns True if the file was rewritten."""
    with open(path, "rb") as f:
        buf = f.read()
    top = list(_atoms(buf, 0, len(buf)))
    moov = next((a for a in top if a[2] == b"moov"), None)
    mdat = next((a for a in top if a[2] == b"mdat"), None)
    if not moov or not mdat or moov[0] < mdat[0]:
        return False
    moov_bytes = bytearray(buf[moov[0]:moov[0] + moov[1]])
    _patch_offsets(moov_bytes, moov[3], len(moov_bytes), len(moov_bytes))
    before = b"".join(buf[a[0]:a[0] + a[1]] for a in top if a[0] < mdat[0] and a[2] != b"moov")
    rest = b"".join(buf[a[0]:a[0] + a[1]] for a in top if a[0] >= mdat[0] and a[2] != b"moov")
    tmp = path + ".faststart"
    with open(tmp, "wb") as f:
        f.write(before + bytes(moov_bytes) + rest)
    os.replace(tmp, path)
    return True


def _opencv_h264(src: str, dst: str) -> bool:
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # H.264 needs even dimensions
    w2, h2 = w - (w % 2), h - (h % 2)
    tmp = dst + ".h264.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"avc1"), fps, (w2, h2))
    if not out.isOpened():
        cap.release()
        return False
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != w2 or frame.shape[0] != h2:
            frame = frame[:h2, :w2]
        out.write(frame)
        n += 1
    cap.release()
    out.release()
    if n == 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        return False
    os.replace(tmp, dst)
    try:
        faststart(dst)
    except Exception as e:
        logger.warning(f"faststart skipped for {dst}: {e}")
    return True


def to_web_mp4(src: str, dst: str) -> bool:
    """Convert `src` (any OpenCV-readable video) into a browser-playable H.264 MP4 at `dst`."""
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            subprocess.run([ff, "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                            "-movflags", "+faststart", "-crf", "23", "-y", dst],
                           check=True, capture_output=True, text=True)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return True
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed, using OpenCV H.264 instead: {e}")
    return _opencv_h264(src, dst)


def is_browser_playable(path: str) -> bool:
    """True for H.264 / VP8 / VP9 / AV1 files."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        code = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1").lower()
        cap.release()
        return code in ("avc1", "h264", "vp80", "vp90", "av01")
    except Exception:
        return False
