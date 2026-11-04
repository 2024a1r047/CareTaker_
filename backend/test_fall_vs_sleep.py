import os
import sys
import numpy as np

sys.path.insert(0, r'c:\CareTaker\backend')
sys.path.insert(0, r'c:\CareTaker\backend\VideoMonitoring')

from prototype_fall_detector import PrototypeFallDetector, extract_static_from_kps
from united_monitor import UnitedMonitor

print("=" * 65)
print("TEST SUITE: 90%+ FALL CONFIRMATION & BED SLEEP DISCRIMINATION")
print("=" * 65)

detector = PrototypeFallDetector()
print(f"✓ PrototypeFallDetector initialized: {detector.model_type}")

# Upright standing pose keypoints
kps_upright = np.zeros((17, 2), dtype=np.float32)
kps_upright[5] = [200, 100]; kps_upright[6] = [300, 100]
kps_upright[11] = [220, 250]; kps_upright[12] = [280, 250]
box_upright = [180, 80, 320, 270]

# Lying in bed horizontal pose keypoints
kps_bed = np.zeros((17, 2), dtype=np.float32)
kps_bed[5] = [150, 240]; kps_bed[6] = [170, 245]
kps_bed[11] = [350, 242]; kps_bed[12] = [370, 248]
box_bed = [140, 230, 380, 260]

# Floor fallen pose keypoints
kps_fallen = np.zeros((17, 2), dtype=np.float32)
kps_fallen[5] = [100, 420]; kps_fallen[6] = [120, 425]
kps_fallen[11] = [320, 415]; kps_fallen[12] = [340, 420]
box_fallen = [90, 400, 350, 440]

# -------------------------------------------------------------
# TEST 1: Sleeping in bed for prolonged period
# -------------------------------------------------------------
print("\n[TEST 1] Testing Prolonged Sleep in Bed (No Sudden Fall Motion)")
detector.reset()
t = 0.0
static_sleep = extract_static_from_kps(kps_bed, box_bed, (480, 640))
false_alarms = 0
for i in range(120): # 120 frames of sleeping (~5 seconds at 25fps)
    conf, ev, al = detector.process_static(t, static_sleep)
    if ev or al:
        false_alarms += 1
    t += 0.04

print(f"  Result: False Alarms = {false_alarms}, is_sleeping = {detector.is_sleeping}, Conf = {detector.last_confidence:.2f}")
assert false_alarms == 0, f"Expected 0 false alarms while sleeping, got {false_alarms}"
assert detector.is_sleeping is True, "Expected is_sleeping to be True"
print("  ✓ PASS: Sleeping in bed produces 0 false alarms and correctly identifies sleep state.")

# -------------------------------------------------------------
# TEST 2: Slow deliberate lie-down (Going to bed)
# -------------------------------------------------------------
print("\n[TEST 2] Testing Slow Deliberate Lie-Down (Going to Bed)")
detector.reset()
t = 0.0
# 1s upright
for i in range(25):
    static = extract_static_from_kps(kps_upright, box_upright, (480, 640))
    detector.process_static(t, static)
    t += 0.04

# 2.5s slow transition to bed
lie_down_alarms = 0
for i in range(60):
    alpha = (i + 1) / 60.0
    kps = (1.0 - alpha) * kps_upright + alpha * kps_bed
    box = (1.0 - alpha) * np.array(box_upright) + alpha * np.array(box_bed)
    static = extract_static_from_kps(kps, box, (480, 640))
    conf, ev, al = detector.process_static(t, static)
    if ev:
        lie_down_alarms += 1
    t += 0.04

print(f"  Result: Lie-Down False Alarms = {lie_down_alarms}, Final is_sleeping = {detector.is_sleeping}")
assert lie_down_alarms == 0, f"Deliberate lie-down triggered false alarm! count={lie_down_alarms}"
print("  ✓ PASS: Deliberate lie-down does NOT trigger a fall alarm.")

# -------------------------------------------------------------
# TEST 3: Actual Sudden Fall (Fast Collapse to Floor)
# -------------------------------------------------------------
print("\n[TEST 3] Testing Actual Sudden Fall (Fast Dynamic Collapse)")
detector.reset()
t = 0.0
# 1.5s standing upright
for i in range(35):
    static = extract_static_from_kps(kps_upright, box_upright, (480, 640))
    detector.process_static(t, static)
    t += 0.04

# Sudden violent fall: 0.25s (6 frames)
fall_fired = False
max_conf = 0.0
for i in range(6):
    alpha = (i + 1) / 6.0
    kps = (1.0 - alpha) * kps_upright + alpha * kps_fallen
    box = (1.0 - alpha) * np.array(box_upright) + alpha * np.array(box_fallen)
    static = extract_static_from_kps(kps, box, (480, 640))
    conf, ev, al = detector.process_static(t, static)
    max_conf = max(max_conf, conf)
    if ev: fall_fired = True
    t += 0.04

# Post-fall lying on floor
static_fallen = extract_static_from_kps(kps_fallen, box_fallen, (480, 640))
for i in range(25):
    conf, ev, al = detector.process_static(t, static_fallen)
    max_conf = max(max_conf, conf)
    if ev: fall_fired = True
    t += 0.04

print(f"  Result: Fall Fired = {fall_fired}, Maximum Confidence = {max_conf * 100:.1f}%")
assert fall_fired is True, "Expected actual fall to trigger an alarm event!"
assert max_conf >= 0.90, f"Expected fall confidence >= 90%, got {max_conf * 100:.1f}%"
print(f"  ✓ PASS: Actual fall detected with {max_conf * 100:.1f}% confidence (>= 90% target reached).")

print("\n" + "=" * 65)
print("ALL TESTS PASSED: 90%+ FALL ACCURACY & BED SLEEP REJECTION VERIFIED!")
print("=" * 65)

