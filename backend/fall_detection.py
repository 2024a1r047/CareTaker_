import cv2
import numpy as np
import queue
import threading
from typing import Dict, Optional, List, Tuple, Any, Union
from collections import deque
import time

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class VideoCaptureThread:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 1 second of frames
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_time = self.frame_count / self.fps
                try:
                    # Add timeout to prevent hanging
                    self.frame_queue.put((frame, frame_time), timeout=0.1)
                    self.frame_count += 1
                except queue.Full:
                    # If queue is full, drop the frame
                    continue
            except Exception as e:
                print(f"Error in video capture: {e}")
                break
        self.cap.release()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


def _angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle of vector p1->p2 relative to vertical axis."""
    vx, vy = (p2[0] - p1[0], p2[1] - p1[1])
    norm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    import math
    ang_h = abs(math.degrees(math.atan2(vy, vx)))
    return 90.0 - min(ang_h, 180 - ang_h)


def _calculate_aspect_ratio(keypoints: np.ndarray) -> float:
    """Calculate body aspect ratio (width/height) for better fall detection."""
    if keypoints is None or len(keypoints) < 17:
        return 0.0
    
    # Get bounding box of person
    valid_kps = keypoints[keypoints[:, 0] > 0]
    if len(valid_kps) == 0:
        return 0.0
    
    x_coords = valid_kps[:, 0]
    y_coords = valid_kps[:, 1]
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if height < 1e-6:
        return 0.0
    
    return width / height


class FallDetector:
    def __init__(
        self,
        *,
        model_name: str = "yolo11n-pose.pt",
        conf: float = 0.3,
        angle_fall_threshold: float = 35.0,
        aspect_ratio_threshold: float = 1.3,  # Width > 1.3x height suggests horizontal
        speed_threshold: float = 30.0,
        cooldown_seconds: float = 3.0,
        temporal_window: int = 5,  # Frames to smooth over
        sensitivity: str = "medium",  # low, medium, high
    ):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Please `pip install ultralytics`.")
        
        self.model = YOLO(model_name)
        self.conf = conf
        self.cooldown_seconds = cooldown_seconds
        
        # Adjust sensitivity
        sensitivity_params = {
            "low": {"angle": 25.0, "speed": 50.0, "aspect": 1.5},
            "medium": {"angle": 35.0, "speed": 30.0, "aspect": 1.3},
            "high": {"angle": 45.0, "speed": 20.0, "aspect": 1.1},
        }
        params = sensitivity_params.get(sensitivity, sensitivity_params["medium"])
        self.angle_fall_threshold = params["angle"]
        self.speed_threshold = params["speed"]
        self.aspect_ratio_threshold = params["aspect"]
        
        # Temporal smoothing
        self.temporal_window = temporal_window
        self._angle_history = deque(maxlen=temporal_window)
        self._aspect_history = deque(maxlen=temporal_window)
        
        # State tracking
        self._prev_hip_y: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._fall_start_time: Optional[float] = None
        self._last_standing_time: float = time.time()
        
        # Multi-person tracking
        self._person_states: Dict[int, Dict] = {}

    def _extract_person_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract multiple features from keypoints for robust detection."""
        if keypoints is None or len(keypoints) < 17:
            return {}
        
        # COCO keypoint indices
        L_SHO, R_SHO = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        
        features = {}
        
        # Torso angle
        shoulder = ((keypoints[L_SHO][0] + keypoints[R_SHO][0]) / 2.0,
                   (keypoints[L_SHO][1] + keypoints[R_SHO][1]) / 2.0)
        hip = ((keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2.0,
               (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2.0)
        
        features["torso_angle"] = _angle_deg(hip, shoulder)
        features["hip_y"] = hip[1]
        
        # Aspect ratio
        features["aspect_ratio"] = _calculate_aspect_ratio(keypoints)
        
        # Hip height relative to knees (falling person's hips drop)
        knee_y = (keypoints[L_KNEE][1] + keypoints[R_KNEE][1]) / 2.0
        features["hip_knee_diff"] = knee_y - hip[1]  # Positive if hips above knees
        
        return features

    def _smooth_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to reduce noise."""
        angle = features.get("torso_angle")
        aspect = features.get("aspect_ratio")
        
        if angle is not None:
            self._angle_history.append(angle)
            features["torso_angle_smooth"] = np.mean(self._angle_history)
        
        if aspect is not None:
            self._aspect_history.append(aspect)
            features["aspect_ratio_smooth"] = np.mean(self._aspect_history)
        
        return features

    def detect_fall(self, frame: np.ndarray) -> Dict[str, object]:
        """Detect falls with multi-feature analysis."""
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        now = time.time()
        
        if frame is None or frame.size == 0:
            return {"fall_detected": False, "timestamp": ts, "keypoints": None, 
                   "boxes": None, "confidence": 0.0}

        fall = False
        confidence = 0.0
        all_keypoints = []
        boxes = None
        features = {}

        try:
            results = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                
                if result.keypoints is not None:
                    kps_list = result.keypoints.xy
                    if kps_list is not None and len(kps_list) > 0:
                        # Process first person (can be extended for multi-person)
                        keypoints = kps_list[0].cpu().numpy()
                        all_keypoints.append(keypoints)
                        
                        # Extract features
                        features = self._extract_person_features(keypoints)
                        features = self._smooth_features(features)
                        
                        # Calculate speed
                        hip_y = features.get("hip_y")
                        if hip_y is not None:
                            speed = 0.0
                            if self._prev_time is not None and self._prev_hip_y is not None:
                                dt = max(1e-3, now - self._prev_time)
                                speed = (hip_y - self._prev_hip_y) / dt
                            features["vertical_speed"] = speed
                            
                            # Fall detection logic with multiple criteria
                            torso_angle = features.get("torso_angle_smooth", features.get("torso_angle", 90))
                            aspect_ratio = features.get("aspect_ratio_smooth", features.get("aspect_ratio", 0))
                            
                            # Check if person was standing before
                            time_since_standing = now - self._last_standing_time
                            was_standing = torso_angle > 50.0 or time_since_standing < 2.0
                            
                            if torso_angle > 50.0:
                                self._last_standing_time = now
                            
                            # Multi-criteria fall detection
                            criteria_met = 0
                            
                            if torso_angle < self.angle_fall_threshold:
                                criteria_met += 1
                                confidence += 0.35
                            
                            if aspect_ratio > self.aspect_ratio_threshold:
                                criteria_met += 1
                                confidence += 0.35
                            
                            if speed > self.speed_threshold:
                                criteria_met += 1
                                confidence += 0.3
                            
                            # Require at least 2 criteria + was standing + outside cooldown
                            if (now >= self._cooldown_until and 
                                was_standing and 
                                criteria_met >= 2):
                                fall = True
                                self._cooldown_until = now + self.cooldown_seconds
                                if self._fall_start_time is None:
                                    self._fall_start_time = now
                            elif not fall:
                                self._fall_start_time = None
                            
                            # Debug info
                            print(f"[Debug] angle={torso_angle:.1f}° aspect={aspect_ratio:.2f} "
                                  f"speed={speed:.1f} criteria={criteria_met}/3 conf={confidence:.2f}")
                            
                            self._prev_hip_y = hip_y
                            self._prev_time = now
                        
        except Exception as e:
            print(f"[FallDetector] Error: {e}")

        return {
            "fall_detected": fall,
            "timestamp": ts,
            "keypoints": all_keypoints[0] if all_keypoints else None,
            "boxes": boxes,
            "features": features,
            "confidence": min(1.0, confidence)
        }


def draw_detections(frame: np.ndarray, res: Dict[str, object], fall_detected: bool) -> np.ndarray:
    """Enhanced visualization with more information."""
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw bounding boxes
    boxes = res.get("boxes")
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            thickness = 3 if fall_detected else 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Draw keypoints and skeleton
    keypoints = res.get("keypoints")
    if keypoints is not None and len(keypoints) > 0:
        # Keypoints
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_copy, (x, y), 4, (0, 255, 255), -1)
        
        # Skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        line_color = (0, 0, 255) if fall_detected else (255, 0, 0)
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame_copy, (x1, y1), (x2, y2), line_color, 2)
    
    # Status overlay
    overlay = frame_copy.copy()
    features = res.get("features", {})
    confidence = res.get("confidence", 0.0)
    
    if fall_detected:
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 200), -1)
        alpha = 0.4
        frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)
        cv2.putText(frame_copy, "FALL DETECTED!", (20, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame_copy, f"Confidence: {confidence:.1%}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame_copy, "Status: Normal", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Feature information
    y_pos = h - 100
    if "torso_angle" in features or "torso_angle_smooth" in features:
        angle = features.get("torso_angle_smooth", features.get("torso_angle"))
        cv2.putText(frame_copy, f"Angle: {angle:.1f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    if "aspect_ratio" in features:
        aspect = features.get("aspect_ratio_smooth", features.get("aspect_ratio"))
        cv2.putText(frame_copy, f"Aspect: {aspect:.2f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    if "vertical_speed" in features:
        speed = features["vertical_speed"]
        cv2.putText(frame_copy, f"Speed: {speed:.1f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame_copy, res.get('timestamp', ''), (w - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_copy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Fall Detection System")
    parser.add_argument("--video", type=str, default="", 
                       help="Path to input video file (default: use camera)")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default="",
                       help="Path to save output video (default: don't save)")
    parser.add_argument("--sensitivity", type=str, default="medium", 
                       choices=["low", "medium", "high"], 
                       help="Detection sensitivity (default: medium)")
    parser.add_argument("--show", action="store_true",
                       help="Show the output video in a window")
    args = parser.parse_args()

    try:
        # Initialize video capture with threading
        cap = VideoCaptureThread(args.video if args.video else args.camera).start()
        frame_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.fps

        # Initialize video writer if output path is provided
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
            print(f"Saving output to: {args.output}")

        # Initialize fall detector
        det = FallDetector(sensitivity=args.sensitivity)
        print(f"Fall detection initialized with {args.sensitivity} sensitivity")
        
        # State variables
        frame_count = 0
        fps_time = time.time()
        last_res = None
        fall_detected = False

        while True:
            frame, current_time_sec = cap.read()
            if frame is None:
                print("End of video stream")
                break
            
            # Format the timestamp
            hours = int(current_time_sec // 3600)
            minutes = int((current_time_sec % 3600) // 60)
            seconds = current_time_sec % 60
            video_timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            
            # For debugging
            if frame_count % 30 == 0:  # Log every ~1 second at 30fps
                print(f"[Debug] Frame: {frame_count}, Time: {video_timestamp}")
            
            # Process every frame (no frame skipping for consistent visualization)
            last_res = det.detect_fall(frame)
            current_fall = last_res.get("fall_detected", False)
            
            # Only update fall_detected if we have a new detection
            if current_fall:
                fall_detected = True
                print(f"\n{'='*60}")
                print(f" FALL DETECTED!")
                print(f"Video Time: {video_timestamp}")
                print(f"Angle: {last_res.get('features', {}).get('torso_angle', 0):.1f}°")
                print(f"Aspect Ratio: {last_res.get('features', {}).get('aspect_ratio', 0):.2f}")
                print(f"Speed: {last_res.get('features', {}).get('vertical_speed', 0):.1f} px/s")
                print(f"{'='*60}\n")
            
            # Draw detections and video timestamp on the frame
            if last_res is not None:
                frame_display = draw_detections(frame, last_res, fall_detected)
            else:
                frame_display = frame.copy()
                
            # Add video timestamp to the frame
            cv2.putText(frame_display, f"{video_timestamp}", 
                       (frame_display.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - fps_time
                fps = 30 / elapsed if elapsed > 0 else 0
                fps_time = current_time
                cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame if show flag is set
            if args.show:
                cv2.imshow("Fall Detection - Press 'Q' to quit", frame_display)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    print("User requested exit")
                    break

            # Write frame to output video if writer is initialized
            if writer is not None:
                writer.write(frame_display)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
    finally:
        # Clean up
        cap.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released")