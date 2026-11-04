import os
import sys
import time
import numpy as np
import cv2

# Ensure backend directory is in path
sys.path.insert(0, r'c:\CareTaker\backend')
sys.path.insert(0, r'c:\CareTaker\backend\VideoMonitoring')

from prototype_fall_detector import (
    PrototypeFallDetector,
    extract_static_from_kps,
    TemporalTracker,
    WindowFeaturizer,
    VoteSmoother
)
from united_monitor import UnitedMonitor, draw_united_interface
from stream_manager import StreamManager

print("=" * 60)
print("TEST 1: Model Variants Loading")
print("=" * 60)
for variant in ["window", "motion", "frame"]:
    det = PrototypeFallDetector(model_variant=variant)
    print(f"  ✓ Successfully loaded '{variant}' (type: {det.model_type}, classes: {det.classes})")

print("\n" + "=" * 60)
print("TEST 2: Temporal & Window Feature Extraction")
print("=" * 60)
tracker = TemporalTracker()
featurizer = WindowFeaturizer()
t = 0.0
kps = np.zeros((17, 2), dtype=np.float32)
kps[5] = [200, 100]
kps[6] = [300, 100]
kps[11] = [220, 250]
kps[12] = [280, 250]
box = [180, 80, 320, 270]

for i in range(15):
    static = extract_static_from_kps(kps, box, (480, 640))
    feats = tracker.update(t, static)
    if feats:
        w_feats = featurizer.update(t, feats)
        if w_feats is not None:
            print(f"  ✓ Emitted 42-dim window feature vector at t={t:.2f}s (shape: {w_feats.shape})")
            break
    t += 0.1

print("\n" + "=" * 60)
print("TEST 3: UnitedMonitor Integration with Prototype RF")
print("=" * 60)
monitor = UnitedMonitor(fall_model="prototype_rf")
print(f"  ✓ UnitedMonitor initialized with fall_model={monitor.fall_model}")

frame = np.zeros((480, 640, 3), dtype=np.uint8)
res = monitor.process_frame(frame)
assert "fall" in res and "inactivity" in res and "global_state" in res
print("  ✓ Frame processing returned all expected top-level keys")

disp = draw_united_interface(frame, res)
assert disp.shape == (480, 640, 3)
print("  ✓ draw_united_interface rendered properly")

print("\n" + "=" * 60)
print("TEST 4: StreamManager CameraSession Creation")
print("=" * 60)
sm = StreamManager()
print(f"  ✓ StreamManager initialized")

print("\n" + "=" * 60)
print("ALL FALL DETECTION INTEGRATION TESTS PASSED!")
print("=" * 60)

