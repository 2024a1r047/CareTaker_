"""
STEP 1: Extract YOLO pose keypoints from all videos and save as .npy files.

Run this ONCE before training. Takes ~20-60 min depending on dataset size.

Saves to backend/VideoMonitoring/data/:
  sequences.npy  shape: (N, 30, 34)  float32
  labels.npy     shape: (N,)         int64   0=normal, 1=fall
  groups.npy     shape: (N,)         object  event-group IDs (for leakage-free split)
"""

import os
import re
import glob
import sys
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\hp\Downloads\Dataset"
FALL_DIR     = os.path.join(DATASET_ROOT, "Fall")
NORMAL_DIR   = os.path.join(DATASET_ROOT, "Normal activities")
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# YOLO model: look in parent (backend/) then current dir
_here        = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL   = (
    os.path.join(_here, "..", "yolo11n-pose.pt")
    if os.path.isfile(os.path.join(_here, "..", "yolo11n-pose.pt"))
    else os.path.join(_here, "yolo11n-pose.pt")
)

SEQ_LEN    = 30   # frames per sequence window
FRAME_SKIP = 2    # process every Nth frame (speeds up extraction)
STRIDE     = 15   # sliding window stride  (overlap = SEQ_LEN - STRIDE)
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# Import normalisation helper from sibling module
sys.path.insert(0, _here)
from fall_lstm_model import normalize_keypoints


def get_event_id(filepath: str, label: int) -> str:
    """
    Derive a scene/event group ID from the filename so that all camera angles
    of the same event are kept together in the same train or val split.

    Patterns handled:
      c10cam1.avi  – c10cam8.avi              → fall_c10
      c20cam1.avi  – c20cam8.avi              → fall_c20
      fall-03-cam0.mp4 / fall-03-cam1.mp4     → fall_fall-03
      fall-16-cam0 (1).mp4                    → fall_fall-16
      adl-01-cam0.mp4                         → normal_adl-01
      adl-03-cam0 (1).mp4                     → normal_adl-03
    """
    prefix = "fall" if label == 1 else "normal"
    name   = os.path.splitext(os.path.basename(filepath))[0].strip()

    # cNNcamM  (e.g. c18cam3)
    m = re.match(r'^(c\d+)cam\d+', name, re.IGNORECASE)
    if m:
        return f"{prefix}_{m.group(1).lower()}"

    # fall-NN-camM  (e.g. fall-03-cam0, fall-16-cam0 (1))
    m = re.match(r'^(fall-\d+)-cam\d+', name, re.IGNORECASE)
    if m:
        return f"fall_{m.group(1).lower()}"

    # adl-NN-camM  (e.g. adl-01-cam0, adl-03-cam0 (1))
    m = re.match(r'^(adl-\d+)-cam\d+', name, re.IGNORECASE)
    if m:
        return f"normal_{m.group(1).lower()}"

    # Fallback: each file is its own group
    return f"{prefix}_{name.lower()}"


def extract_video(video_path: str,
                  yolo,
                  seq_len: int,
                  frame_skip: int,
                  stride: int):
    """Extract sliding-window sequences from one video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return []

    frame_vecs = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        results  = yolo(frame, verbose=False)
        kps_data = results[0].keypoints

        if kps_data is None or len(kps_data.xy) == 0:
            vec = np.zeros(34, dtype=np.float32)
        else:
            xy   = kps_data.xy[0].cpu().numpy()   # (17, 2)
            conf = kps_data.conf[0].cpu().numpy()  # (17,)
            vec  = normalize_keypoints(xy, conf)

        frame_vecs.append(vec)

    cap.release()

    if len(frame_vecs) == 0:
        return []

    # Sliding window
    sequences = []
    for start in range(0, max(1, len(frame_vecs) - seq_len + 1), stride):
        chunk = frame_vecs[start: start + seq_len]
        # Zero-pad if video shorter than seq_len
        while len(chunk) < seq_len:
            chunk.insert(0, np.zeros(34, dtype=np.float32))
        sequences.append(np.stack(chunk, axis=0))   # (seq_len, 34)

    return sequences


def main():
    if not os.path.isfile(YOLO_MODEL):
        print(f"[ERROR] YOLO model not found: {YOLO_MODEL}")
        sys.exit(1)

    print(f"[Config] YOLO model : {YOLO_MODEL}")
    print(f"[Config] Dataset    : {DATASET_ROOT}")
    print(f"[Config] SEQ_LEN={SEQ_LEN}  FRAME_SKIP={FRAME_SKIP}  STRIDE={STRIDE}")
    print(f"[Config] Output dir : {OUT_DIR}\n")

    yolo = YOLO(YOLO_MODEL)

    all_seqs   = []
    all_labels = []
    all_groups = []

    for label, folder, tag in [
        (1, FALL_DIR,   "Fall"),
        (0, NORMAL_DIR, "Normal"),
    ]:
        videos = sorted(
            glob.glob(os.path.join(folder, "*.avi")) +
            glob.glob(os.path.join(folder, "*.mp4"))
        )
        print(f"[{tag}] {len(videos)} videos found in: {folder}")

        for vpath in videos:
            event_id = get_event_id(vpath, label)
            seqs     = extract_video(vpath, yolo, SEQ_LEN, FRAME_SKIP, STRIDE)

            if not seqs:
                print(f"  [SKIP] {os.path.basename(vpath):40s}  (no frames extracted)")
                continue

            all_seqs.extend(seqs)
            all_labels.extend([label] * len(seqs))
            all_groups.extend([event_id] * len(seqs))

            print(f"  {os.path.basename(vpath):45s}  "
                  f"event={event_id:25s}  seqs={len(seqs)}")

    if not all_seqs:
        print("\n[ERROR] No sequences extracted. Check dataset paths.")
        sys.exit(1)

    X = np.stack(all_seqs, axis=0).astype(np.float32)   # (N, 30, 34)
    y = np.array(all_labels, dtype=np.int64)             # (N,)
    g = np.array(all_groups, dtype=object)               # (N,)

    np.save(os.path.join(OUT_DIR, "sequences.npy"), X)
    np.save(os.path.join(OUT_DIR, "labels.npy"),    y)
    np.save(os.path.join(OUT_DIR, "groups.npy"),    g)

    print(f"\n✓ Saved to: {OUT_DIR}")
    print(f"  sequences.npy : {X.shape}")
    print(f"  labels.npy    : fall={int(y.sum())}  normal={int((y==0).sum())}")
    print(f"  groups.npy    : {len(set(g))} unique event groups")


if __name__ == "__main__":
    main()
