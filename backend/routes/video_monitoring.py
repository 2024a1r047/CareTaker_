from fastapi import APIRouter, Depends, HTTPException, status, Header, UploadFile, File, BackgroundTasks, Request, Form, Query
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from pydantic import BaseModel
import cv2
import numpy as np
import json
import os
import sys
import uuid
import copy
import shutil
import time
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.append(str(Path(__file__).parent.parent))

from config import get_db, SessionLocal
from repository.users import JWTRepo, UsersRepo
from tables.users import CareTaker, CareRecipient
from tables.video_analysis import VideoAnalysis
from utils.email import send_fall_alert_email
# Import FastMail for direct email sending with attachments if needed
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv

# Import stream manager and united monitor
sys.path.append(str(Path(__file__).parent.parent / "VideoMonitoring"))
from stream_manager import stream_manager
from united_monitor import UnitedMonitor, draw_united_interface
from camera_registry import (
    list_cameras, get_camera, add_camera, remove_camera, build_stream_url
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=['Video Monitoring'], prefix='/video-monitoring')

# ============================================================================
# PERSISTENCE HANDLER
# ============================================================================
STATUS_FILE = Path("processing_status.json")

def load_process_status():
    """Load processing status from JSON file"""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load status file: {e}")
    return {}

def save_process_status(status_dict):
    """Save processing status to JSON file"""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save status file: {e}")

# Load status on startup
process_status = load_process_status()

# Global inactivity threshold (can be updated per user)
inactivity_thresholds = {}  # username -> threshold_seconds

# Models
class StartLiveRequest(BaseModel):
    """
    Start a live monitoring session.
    Supply ONE of:
      - camera_index : int   0 / 1 / 2 …  (local webcam)
      - camera_url   : str  RTSP/HTTP URL for a CCTV / IP camera
      - camera_id    : str  ID of a saved camera from the registry
    """
    camera_index: int              = 0
    camera_url:   Optional[str]    = None
    camera_id:    Optional[str]    = None
    sensitivity:  str              = "medium"
    fall_model:   str              = "prototype_rf"

class AddCameraRequest(BaseModel):
    name:     str
    url:      str          # RTSP / HTTP URL, or integer string for webcam
    cam_type: str = "ip"  # "webcam" | "ip" | "rtsp"
    location: str = ""
    username: str = ""
    password: str = ""

class InactivityThresholdRequest(BaseModel):
    threshold_seconds: int

class UpdateSessionThresholdRequest(BaseModel):
    threshold_seconds: int


def _get_username_from_auth(auth_header: Optional[str]):
    """Extract username from JWT token"""
    if not auth_header:
        return None
    
    try:
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
            
        token = parts[1]
        decoded = JWTRepo.decode_token(token)
        
        if not decoded or not isinstance(decoded, dict):
            return None
            
        return decoded.get('sub')
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return None


# ============================================================================
# EMAIL HELPER WITH ATTACHMENT SUPPORT
# ============================================================================
load_dotenv()
conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD"),
    MAIL_FROM = os.getenv("MAIL_FROM"),
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER = os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS = True,
    MAIL_SSL_TLS = False,
    USE_CREDENTIALS = True
)

async def send_alert_email_with_image(recipient_email: str, subject: str, body: str, image_bytes: Optional[bytes] = None, image_name: str = "alert.jpg"):
    """Send email with optional image attachment"""
    try:
        # Build message kwargs
        message_kwargs = {
            "subject": subject,
            "recipients": [recipient_email],
            "body": body,
            "subtype": "html"
        }
        
        # Only add attachments if we have image bytes
        if image_bytes:
            message_kwargs["attachments"] = [(image_name, image_bytes, "image/jpeg")]
        
        message = MessageSchema(**message_kwargs)
        fm = FastMail(conf)
        # await fm.send_message(message)
        logger.info(f"Alert email sent to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


# ============================================================================
# UPLOAD AND VIDEO FILE ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/upload-video")
async def upload_video_for_analysis(
    file: UploadFile = File(...),
    recipient_id: Optional[int] = Form(None),
    fall_model: str = Form("prototype_rf"),
    background_tasks: BackgroundTasks = None,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Upload a video file for fall detection and inactivity analysis.
    Returns a process_id to track the analysis status.
    """
    # Authenticate user
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Attach the analysis to this caretaker's patient (their only one when none was given)
    recipient_id = _resolve_recipient(db, user, recipient_id)

    try:
        # Validate file
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save uploaded file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare output directory
        output_dir = Path("processed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"analyzed_{unique_filename}"
        # Force .mp4 extension for output to ensure browser compatibility
        output_path = output_dir / f"{Path(output_filename).stem}.mp4"
        
        # Start video processing in background
        process_id = str(uuid.uuid4())
        
        # Initialize process status
        process_status[process_id] = {
            "status": "processing",
            "progress": 0,
            "user_email": user.email,
            "username": username,
            "caretaker_id": user.id,
            "recipient_id": recipient_id,
            "input_file": str(file_path),
            "output_file": str(output_path),
            "output_filename": output_filename,
            "start_time": time.time(),
            "error": None,
            "falls_detected": [],
            "has_falls": False
        }
        save_process_status(process_status)
        
        # Run processing in background
        # Usage of background_tasks.add_task with a regular def runs in thread pool
        if background_tasks:
            background_tasks.add_task(
                process_video_direct,
                process_id=process_id,
                input_path=str(file_path),
                output_path=str(output_path),
                user_email=user.email,
                caretaker_id=user.id,
                recipient_id=recipient_id,
                fall_model=fall_model
            )
        
        return {
            "status": "success",
            "process_id": process_id,
            "message": "Video uploaded successfully. Processing started."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def process_video_direct(process_id: str, input_path: str, output_path: str, user_email: str, caretaker_id: int, recipient_id: Optional[int] = None, fall_model: str = "prototype_rf"):
    """
    Direct in-process video analysis using UnitedMonitor.
    Runs in a thread pool managed by FastAPI BackgroundTasks.

    Speed optimizations:
      - FRAME_SKIP: only run YOLO on every Nth frame (re-use last result for skipped frames)
      - MAX_WIDTH: resize wide frames before inference (YOLO runs faster on smaller images)
    """
    # FRAME_SKIP is now adaptive (set below, after we know video duration)
    MAX_WIDTH  = 640   # Resize to this width before YOLO (keeps aspect ratio)

    try:
        logger.info(f"Starting direct analysis for {process_id}")
        
        # NOTE: UnitedMonitor is initialised AFTER fps is validated below,
        # so we can pass the correct video_fps for accurate dt calculation.
        # Open Video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open input video")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── FPS sanity check ───────────────────────────────────────────────────
        # Some uploaded videos report fps=0 or absurdly high values from OpenCV.
        # VideoWriter with fps=0 writes nothing / a corrupt file — hence the 4-sec clip.
        if fps <= 0 or fps > 120:
            logger.warning(f"Suspicious FPS value {fps} detected — defaulting to 30")
            fps = 30.0

        # Pre-compute resize scale so we scale frames before inference
        if width > MAX_WIDTH:
            scale = MAX_WIDTH / width
            infer_w = MAX_WIDTH
            infer_h = int(height * scale)
        else:
            scale = 1.0
            infer_w, infer_h = width, height
        
        # Initialize Writer — XVID is the most reliable AVI codec on Windows.
        # We transcode to H.264 via FFmpeg afterwards for browser compatibility.
        temp_avi_path = str(output_path).replace(".mp4", "_raw.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error(f"Failed to initialize VideoWriter with XVID for {temp_avi_path}")
            # Fallback: try mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise Exception("Could not initialize VideoWriter with any available codec")
        
        logger.info(f"Using intermediate AVI file: {temp_avi_path}")
        logger.info(f"Video props: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

        # ── Initialize Monitor ────────────────────────────────────────────
        # Pass video_fps so that the dt used in time-based gates (standing duration,
        # HUD clearing) reflects VIDEO time, not slow processing wall-clock time.
        monitor = UnitedMonitor(
            sensitivity="medium",
            inactivity_threshold=30,
            is_live=False,
            process_every_n_frames=1,
            video_fps=fps,
            fall_model=fall_model
        )

        # ── Adaptive FRAME_SKIP + WARMUP based on video duration ──────────────
        # Problem: with static FRAME_SKIP=4 and WARMUP=30:
        #   - A 1-sec video (30 frames) skips the LSTM buffer fill entirely
        #   - A 4-sec video (120 frames) uses all frames just filling the buffer
        # Fix: for short clips, process EVERY frame and use minimal warmup.
        duration_sec = total_frames / fps if fps > 0 else 999
        if duration_sec < 10:
            # Short clip: process every frame; WARMUP=18 gives 3 frames after the 15-frame
            # LSTM buffer fills before we allow recording (prevents buffer-fill false positives)
            FRAME_SKIP    = 1
            WARMUP_FRAMES = 18
        elif duration_sec < 30:
            # Medium clip: process every other frame
            FRAME_SKIP    = 2
            WARMUP_FRAMES = 10
        else:
            # Long video: skip more aggressively
            FRAME_SKIP    = 4
            WARMUP_FRAMES = 20

        logger.info(f"Duration: {duration_sec:.1f}s  →  FRAME_SKIP={FRAME_SKIP}, WARMUP={WARMUP_FRAMES}")

        # Whole-video fall analysis (handles partial bodies, camera cuts and falls out of view)
        try:
            from offline_fall_analyzer import OfflineFallAnalyzer, merge_falls
            timeline = OfflineFallAnalyzer()
        except Exception as _tl_err:
            logger.warning(f"Timeline fall analyzer unavailable: {_tl_err}")
            timeline, merge_falls = None, None

        frame_count = 0
        falls_detected = []
        total_inactivity_frames = 0
        last_results = None
        last_fall_video_time = -999.0        # for deduplication
        MIN_FALL_GAP_SEC = 3.0               # don't record two falls within 3 video-seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Video clock for the sudden-fall check (true speed even with frame skipping)
            monitor._video_time = frame_count / fps if fps > 0 else None

            # ── Frame-skip: only run YOLO every FRAME_SKIP frames ──────────────
            is_inference_frame = (frame_count % FRAME_SKIP == 0)
            if is_inference_frame:
                # Resize for faster inference
                if scale != 1.0:
                    small = cv2.resize(frame, (infer_w, infer_h))
                else:
                    small = frame
                results = monitor.process_frame(small)
                if timeline is not None:
                    try:
                        timeline.add(frame_count / fps, getattr(monitor, "last_raw_people", []), infer_w, infer_h)
                    except Exception as _tl_err:
                        logger.debug(f"timeline add failed: {_tl_err}")

                # ── Scale bounding boxes back to ORIGINAL frame size ──────────
                # YOLO ran on the small frame, so all coordinates are in small-frame
                # space. We must multiply by (1/scale) before drawing on the full frame.
                if scale != 1.0:
                    inv = 1.0 / scale
                    for person in results["fall"].get("all_persons", []):
                        if person.get("box") is not None:
                            person["box"] = person["box"] * inv
                        if person.get("xy") is not None:
                            person["xy"] = person["xy"] * inv
                    if results["fall"].get("boxes") is not None:
                        results["fall"]["boxes"] = results["fall"]["boxes"] * inv
                    if results["fall"].get("keypoints") is not None:
                        results["fall"]["keypoints"] = results["fall"]["keypoints"] * inv

                last_results = results
            else:
                # Re-use the last result — avoids re-running YOLO on every frame
                # IMPORTANT: clear fall_event_fired on reused frames so the same
                # event is not recorded multiple times (once per skipped frame)
                if last_results is not None:
                    results = copy.deepcopy(last_results)
                    results["fall"]["fall_event_fired"] = False
                    results["fall"]["got_up_event_fired"] = False   # same one-shot rule for "got up"
                else:
                    results = monitor.process_frame(frame)
                    last_results = results

            # Process Frame
            if results is None:
                results = monitor.process_frame(frame)

            # Draw Interface
            display_frame = draw_united_interface(frame, results)

            # Write Frame
            writer.write(display_frame)

            # Track Inactivity
            if results["inactivity"].get("alert"):
                total_inactivity_frames += 1

            # ── Record fall events ─────────────────────────────────────────
            # Guard 1: only on inference frames after warmup
            # Guard 2: MIN_FALL_GAP_SEC deduplication — prevents the same physical
            #   fall from being logged multiple times if _is_currently_falling clears
            #   unexpectedly (e.g. top-down camera where lying flat looks upright).
            if is_inference_frame and frame_count >= WARMUP_FRAMES and results["fall"]["fall_event_fired"]:
                video_time_sec = frame_count / fps if fps > 0 else 0.0
                if video_time_sec - last_fall_video_time >= MIN_FALL_GAP_SEC:
                    conf = results["fall"].get("event_confidence") or results["fall"].get("confidence") or None
                    falls_detected.append({
                        "timestamp": round(video_time_sec, 2),
                        "message": "Fall Detected",
                        "frame": frame_count,
                        "confidence": conf,
                        "detection_method": results["fall"].get("detection_method"),
                        "got_up": False,
                        "time_to_get_up_sec": None,
                    })
                    last_fall_video_time = video_time_sec
                    logger.info(f"Fall Event Recorded at {video_time_sec:.2f}s (frame {frame_count}, conf {conf}%)")
                else:
                    logger.debug(f"Duplicate fall suppressed at {video_time_sec:.2f}s (last was {last_fall_video_time:.2f}s)")

            # ── Person got up after the latest fall (reporting only) ──
            if is_inference_frame and results["fall"].get("got_up_event_fired") and falls_detected and not falls_detected[-1]["got_up"]:
                video_time_sec = frame_count / fps if fps > 0 else 0.0
                f = falls_detected[-1]
                f["got_up"] = True
                # on the ground = from the fall until they stood up (minus the 2 s of standing used to confirm it)
                f["time_to_get_up_sec"] = round(max(0.0, video_time_sec - f["timestamp"] - 2.0), 1)
                f["got_up_at_video_sec"] = round(video_time_sec, 2)
                logger.info(f"Person got up at {video_time_sec:.2f}s, {f['time_to_get_up_sec']}s after the fall")
            
            frame_count += 1
            
            # Update progress every 30 frames
            if frame_count % 30 == 0:
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                if process_id in process_status:
                    process_status[process_id]["progress"] = progress
                    # Optimization: Don't save to disk on every frame, maybe every 10%
                    if progress % 10 == 0:
                        save_process_status(process_status)
        
        # Cleanup
        cap.release()
        writer.release()

        # --- H.264 Transcoding for Web Playback ---
        # OpenCV output with mp4v/avc1 in Windows often results in non-playable files.
        # We use FFmpeg to re-encode into a standard web-friendly H.264 format with YUV420P.
        if os.path.exists(temp_avi_path) and os.path.getsize(temp_avi_path) > 0 and not shutil.which("ffmpeg"):
            # No ffmpeg installed: re-encode to H.264 MP4 with OpenCV so the browser can play it
            # (previously the XVID file was just renamed to .mp4 and the player stayed at 0:00)
            try:
                from utils.web_video import to_web_mp4
                if to_web_mp4(temp_avi_path, str(output_path)):
                    os.remove(temp_avi_path)
                    logger.info(f"Converted to browser-playable H.264 MP4 with OpenCV: {output_path}")
                elif not os.path.exists(output_path):
                    os.rename(temp_avi_path, output_path)
            except Exception as e:
                logger.error(f"OpenCV H.264 conversion failed: {e}")
                if not os.path.exists(output_path):
                    os.rename(temp_avi_path, output_path)
        elif os.path.exists(temp_avi_path) and os.path.getsize(temp_avi_path) > 0:
            try:
                logger.info(f"Transcoding {temp_avi_path} to {output_path} for web compatibility...")
                # -pix_fmt yuv420p and -movflags faststart are critical for browser playback
                cmd = f'ffmpeg -i "{temp_avi_path}" -vcodec libx264 -pix_fmt yuv420p -preset ultrafast -movflags +faststart -crf 23 -y "{output_path}"'
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    os.remove(temp_avi_path)
                    logger.info("Transcoding successful. Intermediate file removed.")
                else:
                    raise Exception("Output file was not created or is empty")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg command failed with return code {e.returncode}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
                # If transcoding failed, try to just move the AVI to MP4 so there's AT LEAST a file (though it may not play)
                if not os.path.exists(output_path):
                    os.rename(temp_avi_path, output_path)
            except Exception as e:
                logger.error(f"FFmpeg transcoding error: {e}")
                if not os.path.exists(output_path):
                    os.rename(temp_avi_path, output_path)
        else:
            logger.error(f"Intermediate file {temp_avi_path} is missing or empty. Cannot transcode.")
        
        # ── Fall summary (got up? how long on the ground?) ──
        from services.fall_event_store import build_fall_summary, build_video_summary, save_fall_event
        video_duration = frame_count / fps if fps > 0 else 0.0

        # Add falls found by the whole-video timeline analysis (and fill in "got up" times)
        if timeline is not None and merge_falls is not None:
            try:
                tl_falls = timeline.finalize(video_duration)
                if tl_falls:
                    logger.info(f"Timeline analysis found {len(tl_falls)} fall(s): {tl_falls}")
                falls_detected = merge_falls(falls_detected, tl_falls)
            except Exception as _tl_err:
                logger.error(f"Timeline fall analysis failed: {_tl_err}")
        for f in falls_detected:
            if not f["got_up"]:
                f["time_on_ground_sec"] = round(max(0.0, video_duration - f["timestamp"]), 1)
            f["source"] = "upload"
            f["video_time_sec"] = f["timestamp"]
            f["summary"] = build_fall_summary(f)
            f["message"] = f["summary"]          # shown in the Fall Details table
        fall_summary = build_video_summary(falls_detected, video_duration)

        # Final Status Update
        if process_id in process_status:
            process_status[process_id].update({
                "status": "completed",
                "progress": 100,
                "falls_detected": falls_detected,
                "fall_summary": fall_summary,
                "has_falls": len(falls_detected) > 0,
                "output_filename": os.path.basename(output_path), 
                "completed_at": datetime.now().isoformat()
            })
            save_process_status(process_status)
            
            # Send Email if falls detected
            if len(falls_detected) > 0:
                asyncio.run(send_alert_email_with_image(
                    recipient_email=user_email,
                    subject="⚠️ Fall Detected in Uploaded Video",
                    body=f"A fall was detected during the analysis of your uploaded video.\n\n{fall_summary}\n\n"
                         + "\n".join(f"- {f['summary']}" for f in falls_detected)
                         + "\n\nPlease log in to your dashboard to review the details and playback."
                    # No image attachment for now
                ))

            # --- SAVE TO DATABASE ---
            try:
                db_session = SessionLocal()
                
                # Calculate metrics from actual monitoring results
                fall_count = len(falls_detected)
                inactivity_secs = int(total_inactivity_frames / fps) if fps > 0 else 0
                
                # Activity Score logic: 10 base, minus deductions for falls and excessive inactivity
                # (Assuming 1 hour video, if 50% inactive, score 5.0)
                duration_secs = frame_count / fps if fps > 0 else 1
                inactivity_ratio = inactivity_secs / duration_secs if duration_secs > 0 else 0
                
                activity_score = max(0.0, 10.0 * (1.0 - inactivity_ratio) - (fall_count * 2.0))
                mobility_score = max(0.0, 10.0 * (1.0 - (inactivity_ratio * 0.5)) - (fall_count * 1.5))
                
                analysis_entry = VideoAnalysis(
                    recipient_id=recipient_id,
                    caretaker_id=caretaker_id,
                    video_filename=os.path.basename(output_path),
                    has_fall=len(falls_detected) > 0,
                    fall_count=fall_count,
                    inactivity_duration_seconds=inactivity_secs,
                    activity_score=activity_score,
                    mobility_score=mobility_score,
                    timestamp=datetime.utcnow()
                )
                db_session.add(analysis_entry)
                db_session.commit()
                logger.info(f"Saved analysis to DB for Recipient {recipient_id}")

                # Keep every fall (with its summary) permanently in fall_events
                analysed_at = datetime.utcnow()
                for f in falls_detected:
                    f["event_id"] = save_fall_event({
                        "caretaker_id": caretaker_id,
                        "recipient_id": recipient_id,
                        "video_analysis_id": analysis_entry.id,
                        "source": "upload",
                        "video_filename": os.path.basename(output_path),
                        "occurred_at": analysed_at,
                        "video_time_sec": f["timestamp"],
                        "confidence": f.get("confidence"),
                        "detection_method": f.get("detection_method"),
                        "got_up": f["got_up"],
                        "time_to_get_up_sec": f.get("time_to_get_up_sec"),
                        "time_on_ground_sec": f.get("time_on_ground_sec"),
                        "summary": f["summary"],
                    })
                if falls_detected:
                    logger.info(f"Saved {len(falls_detected)} fall event(s) with summaries to fall_events")
                    if process_id in process_status:
                        process_status[process_id]["falls_detected"] = falls_detected
                        save_process_status(process_status)
            except Exception as e:
                logger.error(f"Failed to save analysis to DB: {e}")
            finally:
                db_session.close()
        
        # Remove input file
        if os.path.exists(input_path):
            os.remove(input_path)
            
        logger.info(f"Analysis completed for {process_id}")

    except Exception as e:
        logger.error(f"Analysis error for {process_id}: {e}", exc_info=True)
        if process_id in process_status:
            process_status[process_id].update({
                "status": "error",
                "error": str(e)
            })
            save_process_status(process_status)
        # Try cleanup
        if os.path.exists(input_path):
             os.remove(input_path)
        # Close window if it exists
        try:
            cv2.destroyAllWindows()
        except:
            pass


@router.get("/status/{process_id}")
async def get_processing_status(
    process_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get the status of a video processing job.
    """
    global process_status  # Declare at the start before any usage
    
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    if process_id not in process_status:
        # Reload from disk to be sure
        process_status = load_process_status()
        
    if process_id not in process_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    state = process_status[process_id].copy()
    
    # Add elapsed time
    if "start_time" in state:
        elapsed = time.time() - state["start_time"]
        state["elapsed_seconds"] = round(elapsed, 2)
    
    # Frontend will construct the absolute URL from output_filename
    # No need to add processed_url here
        
    return state


class LogEventRequest(BaseModel):
    type: str = "fall"                       # fall | got_up
    timestamp: Optional[str] = None          # ISO date string or ms since epoch (browser)
    confidence: Optional[float] = None       # 0-1 (browser detector) or 0-100
    source: Optional[str] = "tfjs_live"
    recipient_id: Optional[int] = None
    event_id: Optional[int] = None           # for got_up: which fall to update
    time_to_get_up_sec: Optional[float] = None
    time_on_ground_sec: Optional[float] = None   # for long_lie / stopped_down


LONG_LIE_SEC = 120   # getting up within 2 minutes is fine; longer is a "long lie"


def _resolve_recipient(db: Session, user, rid: Optional[int]) -> Optional[int]:
    """This caretaker's patient; if none was given and the caretaker has exactly one, use that one."""
    if rid and db.query(CareRecipient).filter(CareRecipient.id == rid, CareRecipient.caretaker_id == user.id).first():
        return rid
    only = db.query(CareRecipient).filter(CareRecipient.caretaker_id == user.id).limit(2).all()
    return only[0].id if len(only) == 1 else None


def _raise_long_lie_alert(db: Session, user, event_id: int, secs: Optional[float]) -> None:
    """Critical MedicalAlert + e-mail to the caretaker when someone is still on the floor after 2 minutes."""
    from tables.video_analysis import FallEvent
    from tables.medical_conditions import MedicalAlert, AlertType, AlertSeverity
    ev = db.query(FallEvent).filter(FallEvent.id == event_id, FallEvent.caretaker_id == user.id).first()
    if not ev:
        return
    rid = ev.recipient_id or _resolve_recipient(db, user, None)
    mins = f"{int((secs or LONG_LIE_SEC) // 60)} min {int((secs or LONG_LIE_SEC) % 60)} s"
    name = ""
    if rid:
        r = db.query(CareRecipient).filter(CareRecipient.id == rid).first()
        name = r.full_name if r else ""
        try:
            db.add(MedicalAlert(
                care_recipient_id=rid, alert_type=AlertType.critical, severity=AlertSeverity.critical,
                message=f"LONG LIE: {name or 'The person'} fell and has NOT got up after {mins} (home camera). Check on them now.",
            ))
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"[FallEvents] long-lie alert not saved: {e}")
    try:
        import asyncio
        from utils.email import send_fall_alert_email
        if user.email:
            asyncio.get_event_loop().create_task(send_fall_alert_email(user.email, {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "fall_count": 1,
                "fall_details": [{"time": (ev.occurred_at or datetime.utcnow()).strftime("%H:%M:%S UTC"),
                                  "confidence": ev.confidence or 0, "angle": "-"}],
                "location": f"Home camera — {name or 'care recipient'} still on the floor after {mins}",
            }))
    except Exception as e:
        logger.error(f"[FallEvents] long-lie e-mail failed: {e}")


@router.post("/log-event")
async def log_monitoring_event(
    payload: LogEventRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Save a fall detected by the in-browser (TF.js) live monitor in fall_events,
    or mark that the person got up after it.
    """
    from services.fall_event_store import save_fall_event, update_fall_event
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Only this caretaker's own patient (falls into the only patient when none was given)
    rid = _resolve_recipient(db, user, payload.recipient_id)

    if payload.type == "got_up":
        update_fall_event(payload.event_id, got_up=True, time_to_get_up_sec=payload.time_to_get_up_sec,
                          got_up_at=datetime.utcnow())
        return {"status": "success", "event_id": payload.event_id,
                "within_limit": (payload.time_to_get_up_sec or 0) <= LONG_LIE_SEC}

    if payload.type == "long_lie":
        # Still on the floor after 2 minutes: record it and alert the caretaker
        update_fall_event(payload.event_id, time_on_ground_sec=payload.time_on_ground_sec)
        if payload.event_id:
            _raise_long_lie_alert(db, user, payload.event_id, payload.time_on_ground_sec)
        return {"status": "success", "event_id": payload.event_id, "alerted": True}

    if payload.type == "stopped_down":
        # Monitoring stopped while the person was still down
        update_fall_event(payload.event_id, time_on_ground_sec=payload.time_on_ground_sec)
        return {"status": "success", "event_id": payload.event_id}

    conf = payload.confidence
    if conf is not None and conf <= 1.0:
        conf = round(conf * 100, 1)          # browser sends 0-1
    occurred = datetime.utcnow()
    if payload.timestamp:
        try:
            ts = str(payload.timestamp).strip()
            if ts.replace('.', '', 1).isdigit():
                occurred = datetime.utcfromtimestamp(float(ts) / 1000.0)
            else:
                from datetime import timezone as _tz
                d = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                occurred = d.astimezone(_tz.utc).replace(tzinfo=None) if d.tzinfo else d
        except Exception:
            pass
    event_id = save_fall_event({
        "caretaker_id": user.id, "recipient_id": rid, "source": payload.source or "tfjs_live",
        "occurred_at": occurred, "confidence": conf, "detection_method": "tfjs",
    })
    return {"status": "success", "event_id": event_id}


@router.get("/fall-events")
async def get_fall_events(
    recipient_id: Optional[int] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Saved falls with their summaries (newest first) for this caretaker."""
    from tables.video_analysis import FallEvent
    from services.fall_event_store import fall_event_to_dict
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    q = db.query(FallEvent).filter(FallEvent.caretaker_id == user.id)
    if recipient_id:
        q = q.filter(FallEvent.recipient_id == recipient_id)
    rows = q.order_by(FallEvent.occurred_at.desc(), FallEvent.id.desc()).limit(max(1, min(limit, 500))).all()
    return {"total": len(rows), "events": [fall_event_to_dict(r) for r in rows]}


@router.get("/history")
async def get_monitoring_history(
    authorization: Optional[str] = Header(None)
):
    """
    Get history of processed videos.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    try:
        output_dir = Path("processed_videos")
        if not output_dir.exists():
            return []
            
        files = []
        for file in output_dir.glob("analyzed_*.mp4"):
            stats = file.stat()
            files.append({
                "filename": file.name,
                "timestamp": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "size": stats.st_size
                # Frontend will construct absolute URL from filename
            })
            
        # Sort by timestamp descending
        files.sort(key=lambda x: x["timestamp"], reverse=True)
        return files
        
    except Exception as e:
        logger.error(f"Error listing history: {e}")
        return []


@router.get("/stats")
async def get_monitoring_stats(
    care_recipient_id: Optional[int] = Query(None, description="Filter by care recipient ID"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Get true statistics for video monitoring from the database.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
        
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    try:
        query = db.query(VideoAnalysis).filter(VideoAnalysis.caretaker_id == user.id)
        if care_recipient_id:
            query = query.filter(VideoAnalysis.recipient_id == care_recipient_id)
            
        analyses = query.all()
        
        total_uploads = len(analyses)
        total_alerts = sum((a.fall_count or 0) for a in analyses)

        # Falls caught by the live camera / in-browser monitor are stored in fall_events
        live_falls = 0
        try:
            from tables.video_analysis import FallEvent
            from sqlalchemy import or_
            fq = db.query(FallEvent).filter(FallEvent.caretaker_id == user.id, FallEvent.source != "upload")
            if care_recipient_id:
                fq = fq.filter(or_(FallEvent.recipient_id == care_recipient_id, FallEvent.recipient_id.is_(None)))
            live_falls = fq.count()
        except Exception as _e:
            logger.error(f"Error counting live falls: {_e}")

        return {
            "total_uploads": total_uploads,
            "total_alerts": total_alerts + live_falls,
            "live_falls": live_falls,
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {
            "total_uploads": 0,
            "total_alerts": 0
        }


# ============================================================================
# NEW STREAMING ENDPOINTS FOR WEB-BASED VIDEO MONITORING
# ============================================================================

@router.post("/start")
async def start_live_monitoring(
    request: StartLiveRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Start a new live monitoring session with MJPEG streaming.
    Returns session_id and stream_url for embedding in dashboard.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Check for existing active session for this user
    for session_id in stream_manager.get_active_sessions():
        session = stream_manager.get_session(session_id)
        if session and session.username == username:
             return {
                "status": "already_active",
                "session_id": session_id,
                "stream_url": f"/api/video-monitoring/stream/{session_id}",
                "message": "You already have an active session"
            }
    
    # Determine the camera source: Registry > URL > Local
    camera_source = None
    camera_name   = ""

    if request.camera_id:
        # Use a saved camera from the registry
        cam_url = build_stream_url(request.camera_id)
        if cam_url is None:
            raise HTTPException(status_code=404, detail=f"Camera '{request.camera_id}' not found in registry")
        cam_meta = get_camera(request.camera_id)
        camera_source = cam_url
        camera_name   = cam_meta.get("name", request.camera_id) if cam_meta else request.camera_id
        logger.info(f"[Live] Using Registered camera: {camera_name}")

    elif request.camera_url:
        # Direct URL (RTSP / HTTP) provided by user
        url = request.camera_url.strip()
        if not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(status_code=400, detail="camera_url must start with rtsp://, http://, or https://")
        camera_source = url
        camera_name   = url
        logger.info(f"[Live] Using IP camera URL: {camera_source}")

    else:
        # Fall back to local webcam index
        camera_source = request.camera_index
        camera_name   = f"Webcam #{request.camera_index}"
        logger.info(f"[Live] Using USB camera index: {camera_source}")

    # Create new session
    session_id = str(uuid.uuid4())
    inactivity_threshold = inactivity_thresholds.get(username, 30)
    
    success = stream_manager.create_session(
        session_id=session_id,
        camera_source=camera_source,
        sensitivity=request.sensitivity,
        inactivity_threshold=inactivity_threshold,
        camera_name=camera_name,
        fall_model=request.fall_model,
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to start camera session. "
                "Check that the camera is reachable and the URL/index is correct."
            )
        )
    
    # Assign ownership
    session = stream_manager.get_session(session_id)
    if session:
        session.username  = username
        session.user_email = user.email
    
    return {
        "status": "success",
        "session_id": session_id,
        "stream_url": f"/api/video-monitoring/stream/{session_id}",
        "camera_name": camera_name,
        "message": "Live monitoring started successfully"
    }


# IMPORTANT: More specific routes must come BEFORE generic routes
# /session/{session_id}/alerts must be before /stream/{session_id}
# Otherwise FastAPI will match "session" as a session_id value in /stream/{session_id}

@router.get("/session/{session_id}/alerts")
async def get_session_alerts(
    session_id: str,
    since: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """
    Get alerts for an active session.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    alerts = session.get_alerts(since=since)
    
    return {
        "session_id": session_id,
        "alerts": alerts,
        "total": len(alerts)
    }


@router.get("/stream/{session_id}")
async def stream_video(session_id: str, request: Request):
    """
    MJPEG stream endpoint for live video feed.
    Handles client disconnects to stop the stream generator.
    """
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logger.info(f"Stream requested for session {session_id}, running={session.running}")
    
    async def generate():
        """Generator function for MJPEG stream with disconnect check"""
        frame_counter = 0
        try:
            while session.running:
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from stream {session_id}")
                    break

                frame_bytes = session.get_frame()
                if frame_bytes:
                    frame_counter += 1
                    if frame_counter % 30 == 0:  # Log every 30 frames
                        logger.debug(f"Stream {session_id}: Sent {frame_counter} frames, size={len(frame_bytes)} bytes")
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # FPS limiting - target ~25 FPS
                    await asyncio.sleep(0.04)
                else:
                    # No frame available, wait a bit
                    if frame_counter == 0:
                        logger.warning(f"Stream {session_id}: No frames available yet")
                    await asyncio.sleep(0.05)
                    
        except asyncio.CancelledError:
            logger.info(f"Stream {session_id} cancelled")
        except Exception as e:
            logger.error(f"Stream {session_id} error: {e}", exc_info=True)
        finally:
            logger.info(f"Stream {session_id} ended after {frame_counter} frames")
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/stop/{session_id}")
async def stop_live_monitoring(
    session_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    Stop an active monitoring session and release camera.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.username != username:
        raise HTTPException(status_code=403, detail="Not authorized to stop this session")
    
    stream_manager.stop_session(session_id)
    return {"status": "success", "message": "Monitoring session stopped"}


@router.post("/update-threshold/{session_id}")
async def update_session_threshold(
    session_id: str,
    request: UpdateSessionThresholdRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Update inactivity threshold for a running session.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
         raise HTTPException(status_code=404, detail="Session not found")

    if session.username != username:
         raise HTTPException(status_code=403, detail="Not authorized")
         
    if request.threshold_seconds < 5:
        raise HTTPException(status_code=400, detail="Threshold too low")
        
    session.update_threshold(request.threshold_seconds)
    
    return {"status": "success", "message": f"Threshold updated to {request.threshold_seconds}s"}


@router.post("/set-inactivity-threshold")
async def set_inactivity_threshold(
    request: InactivityThresholdRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Set default inactivity threshold for user.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    if request.threshold_seconds < 5 or request.threshold_seconds > 300:
        raise HTTPException(status_code=400, detail="Threshold must be between 5 and 300 seconds")
    
    inactivity_thresholds[username] = request.threshold_seconds
    
    return {
        "status": "success",
        "threshold_seconds": request.threshold_seconds,
        "message": f"Inactivity threshold set to {request.threshold_seconds} seconds"
    }

@router.get("/get-inactivity-threshold")
async def get_inactivity_threshold(authorization: Optional[str] = Header(None)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    threshold = inactivity_thresholds.get(username, 30)
    return {"threshold_seconds": threshold}


# ── IP Camera Test Endpoint ─────────────────────────────────────────────────

class TestIPCameraRequest(BaseModel):
    camera_url: str

@router.post("/test-ip-camera")
async def test_ip_camera(
    request: TestIPCameraRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Test whether an IP camera URL is reachable by briefly opening it with OpenCV.
    Returns {reachable: true/false} within ~5 seconds.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    url = request.camera_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="camera_url is required")

    def _check():
        cap = cv2.VideoCapture(url)
        # Give it up to 5 seconds to open
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        opened = cap.isOpened()
        if opened:
            ret, _ = cap.read()
            opened = ret
        cap.release()
        return opened

    try:
        # Run the blocking OpenCV call in a thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        reachable = await asyncio.wait_for(
            loop.run_in_executor(None, _check),
            timeout=8.0
        )
        return {"reachable": reachable, "url": url}
    except asyncio.TimeoutError:
        return {"reachable": False, "detail": "Connection timed out after 8 seconds"}
    except Exception as e:
        logger.error(f"[test-ip-camera] Error testing {url}: {e}")
        return {"reachable": False, "detail": str(e)}


@router.get("/recipient/{recipient_id}/stats")
async def get_recipient_video_stats(
    recipient_id: int,
    days: Optional[int] = Query(None, description="Number of days to analyze"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Get aggregated video analysis stats for a specific recipient.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    # 1. Get all analysis records for this recipient
    query = db.query(VideoAnalysis).filter(VideoAnalysis.recipient_id == recipient_id)
    if days is not None:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(VideoAnalysis.timestamp >= cutoff_date)
    analyses = query.all()
    
    if not analyses:
        return {
            "total_videos": 0,
            "total_falls": 0,
            "avg_activity_score": 0,
            "avg_mobility_score": 0,
            "history": []
        }
    
    # 2. Aggregate Data
    total_videos = len(analyses)
    total_falls = sum(r.fall_count for r in analyses)
    avg_activity = sum(r.activity_score for r in analyses) / total_videos if total_videos > 0 else 0
    avg_mobility = sum(r.mobility_score for r in analyses) / total_videos if total_videos > 0 else 0
    
    # 3. Format History for Graphs (last 10 sessions)
    history = []
    # Sort chronologically
    sorted_analyses = sorted(analyses, key=lambda x: x.timestamp)
    for r in sorted_analyses:
        history.append({
            "date": r.timestamp.isoformat(),
            "falls": r.fall_count,
            "activity": r.activity_score,
            "mobility": r.mobility_score
        })
        
    return {
        "total_videos": total_videos,
        "total_falls": total_falls,
        "avg_activity_score": round(avg_activity, 1),
        "avg_mobility_score": round(avg_mobility, 1),
        "history": history
    }


@router.get("/recipient/{recipient_id}/fall-analysis")
async def get_recipient_fall_analysis(
    recipient_id: int,
    days: int = Query(30, ge=1, le=365, description="Look-back window in days"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Elder-care fall analysis for one care recipient, built from fall_events (live camera,
    in-browser monitor and uploaded videos) plus uploaded-video sessions.

    - Detections less than 90 s apart are merged into one fall incident (older detector
      versions logged the same fall several times).
    - Got up within 2 minutes = fine; longer, or never seen getting up = long lie.
    """
    from tables.video_analysis import FallEvent
    from sqlalchemy import or_, and_
    from datetime import timedelta
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id,
                                           CareRecipient.caretaker_id == user.id).first()
    if not recip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found")

    now = datetime.utcnow()
    since = now - timedelta(days=days)
    # Falls tagged with this patient, plus untagged ones from this caretaker's cameras when
    # the caretaker looks after only this patient
    n_recips = db.query(CareRecipient).filter(CareRecipient.caretaker_id == user.id).count()
    scope = FallEvent.recipient_id == recipient_id
    if n_recips == 1:
        scope = or_(scope, and_(FallEvent.recipient_id.is_(None), FallEvent.caretaker_id == user.id))
    rows = (db.query(FallEvent).filter(scope, FallEvent.occurred_at >= since - timedelta(days=days))
            .order_by(FallEvent.occurred_at.asc(), FallEvent.id.asc()).all())

    # ── Merge repeated detections of the same fall into incidents ──
    incidents = []
    for r in rows:
        if not r.occurred_at:
            continue
        last = incidents[-1] if incidents else None
        same_video = last and r.source == "upload" and last["video"] == (r.video_filename or "")
        if last and (r.occurred_at - last["last_at"]).total_seconds() <= 90 and (r.source != "upload" or same_video):
            last["last_at"] = r.occurred_at
            last["detections"] += 1
            last["confidence"] = max(last["confidence"] or 0, r.confidence or 0)
            if r.got_up and (last["time_to_get_up_sec"] is None or not last["got_up"]):
                last["got_up"] = True
                last["time_to_get_up_sec"] = r.time_to_get_up_sec
            if r.time_on_ground_sec:
                last["time_on_ground_sec"] = max(last["time_on_ground_sec"] or 0, r.time_on_ground_sec)
            continue
        incidents.append({
            "id": r.id, "occurred_at": r.occurred_at, "last_at": r.occurred_at, "detections": 1,
            "source": r.source, "video": r.video_filename or "", "camera": r.camera_name,
            "confidence": r.confidence, "got_up": bool(r.got_up),
            "time_to_get_up_sec": r.time_to_get_up_sec, "time_on_ground_sec": r.time_on_ground_sec,
            "untagged": r.recipient_id is None,
        })

    def outcome(i):
        if i["got_up"] and i["time_to_get_up_sec"] is not None:
            return "ok" if i["time_to_get_up_sec"] <= LONG_LIE_SEC else "long_lie"
        if i["got_up"]:
            return "ok"
        if (i["time_on_ground_sec"] or 0) > LONG_LIE_SEC:
            return "long_lie"
        return "unknown"   # monitoring stopped / no record of getting up

    for i in incidents:
        i["outcome"] = outcome(i)

    current = [i for i in incidents if i["occurred_at"] >= since]
    previous = [i for i in incidents if i["occurred_at"] < since]
    up_times = [i["time_to_get_up_sec"] for i in current if i["got_up"] and i["time_to_get_up_sec"] is not None]
    ok = sum(1 for i in current if i["outcome"] == "ok")
    long_lies = sum(1 for i in current if i["outcome"] == "long_lie")
    unknown = sum(1 for i in current if i["outcome"] == "unknown")

    # Time-of-day pattern (patient's local time, IST by default)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(os.getenv("PATIENT_TIMEZONE", "Asia/Kolkata"))
    except Exception:
        from datetime import timezone as _tz
        tz = _tz(timedelta(hours=5, minutes=30))
    from datetime import timezone as _utc_tz

    def local(dt):
        return dt.replace(tzinfo=_utc_tz.utc).astimezone(tz)

    buckets = {"Morning (5-12)": 0, "Afternoon (12-17)": 0, "Evening (17-21)": 0, "Night (21-5)": 0}
    for i in current:
        h = local(i["occurred_at"]).hour
        key = ("Morning (5-12)" if 5 <= h < 12 else "Afternoon (12-17)" if 12 <= h < 17
               else "Evening (17-21)" if 17 <= h < 21 else "Night (21-5)")
        buckets[key] += 1

    # Falls per day for the chart
    per_day = {}
    for d in range(days):
        day = (local(now) - timedelta(days=days - 1 - d)).date().isoformat()
        per_day[day] = 0
    for i in current:
        k = local(i["occurred_at"]).date().isoformat()
        if k in per_day:
            per_day[k] += 1

    # Uploaded-video sessions (activity / mobility scores)
    va = db.query(VideoAnalysis).filter(VideoAnalysis.recipient_id == recipient_id,
                                        VideoAnalysis.timestamp >= since).all()
    avg_mob = round(sum(v.mobility_score or 0 for v in va) / len(va), 1) if va else None

    # ── Plain-language analysis ──
    n = len(current)
    insights = []
    if n == 0:
        insights.append(f"No falls recorded in the last {days} days.")
    else:
        insights.append(f"{n} fall{'s' if n != 1 else ''} in the last {days} days"
                        + (f" (previous {days} days: {len(previous)})." if previous or n else "."))
        if up_times:
            avg_up = sum(up_times) / len(up_times)
            insights.append(f"Got up within 2 minutes after {ok} of {n} falls; average time to get up "
                            f"{int(avg_up // 60)} min {int(avg_up % 60)} s, slowest {int(max(up_times) // 60)} min {int(max(up_times) % 60)} s.")
        if long_lies:
            insights.append(f"{long_lies} long lie{'s' if long_lies != 1 else ''} (on the floor for more than 2 minutes) — "
                            f"risk of injury, dehydration and pressure sores; consider a wearable alarm and a check-in routine.")
        if unknown:
            insights.append(f"{unknown} fall{'s were' if unknown != 1 else ' was'} not followed by a recorded "
                            f"stand-up (monitoring stopped or the person left the camera view) — confirm with the person.")
        peak = max(buckets, key=buckets.get)
        if buckets[peak] >= 2 and buckets[peak] >= n / 2:
            insights.append(f"Most falls happen in the {peak.split(' (')[0].lower()} — check lighting, footwear and "
                            f"medicine timing (BP / sugar medicines) around that time.")
        if previous and n > len(previous):
            insights.append("Falls are increasing compared with the previous period — discuss with the doctor.")
        repeated = sum(1 for i in current if i["detections"] > 1)
        if repeated:
            insights.append(f"{repeated} incident{'s' if repeated != 1 else ''} were detected several times by the camera "
                            f"and have been merged into single falls.")
    risk = ("high" if long_lies or n >= 3 else "moderate" if n >= 1 else "low")

    def fmt(i):
        return {
            "id": i["id"],
            "occurred_at": i["occurred_at"].isoformat() + "Z",
            "source": i["source"], "camera": i["camera"], "video": i["video"] or None,
            "confidence": round(i["confidence"] or 0),
            "detections": i["detections"],
            "got_up": i["got_up"],
            "time_to_get_up_sec": i["time_to_get_up_sec"],
            "time_on_ground_sec": i["time_on_ground_sec"],
            "outcome": i["outcome"],
            "untagged": i["untagged"],
        }

    return {
        "days": days,
        "risk_level": risk,
        "total_falls": n,
        "previous_period_falls": len(previous),
        "raw_detections": sum(i["detections"] for i in current),
        "got_up_within_limit": ok,
        "long_lies": long_lies,
        "unknown_outcome": unknown,
        "avg_time_to_get_up_sec": round(sum(up_times) / len(up_times), 1) if up_times else None,
        "max_time_to_get_up_sec": max(up_times) if up_times else None,
        "last_fall_at": (current[-1]["occurred_at"].isoformat() + "Z") if current else None,
        "limit_sec": LONG_LIE_SEC,
        "time_of_day": buckets,
        "per_day": [{"date": k, "falls": v} for k, v in per_day.items()],
        "video_sessions": len(va),
        "avg_mobility_score": avg_mob,
        "insights": insights,
        "incidents": [fmt(i) for i in reversed(current)],
    }


# ============================================================================
# MEDICINES TO FIND IN THE ROOM  (photo -> recognised by the room camera -> location)
# ============================================================================

class MedicineRefRequest(BaseModel):
    name: str
    recipient_id: Optional[int] = None
    image: Optional[str] = None               # small JPEG data URL
    source: Optional[str] = "upload"          # upload | camera
    mode: Optional[str] = "mobilenet"
    embeddings: List[List[float]]
    negatives: Optional[List[List[float]]] = None


class MedicineSeenRequest(BaseModel):
    location: Optional[str] = None
    location_hi: Optional[str] = None
    box: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    probable: Optional[bool] = False          # strong guess, not yet confirmed


def _caretaker_or_401(db: Session, authorization: Optional[str]):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


def _medicine_ref_dict(m, with_vectors: bool = False) -> Dict:
    d = {
        "id": m.id, "name": m.name, "image": m.image, "source": m.source, "mode": m.mode,
        "recipient_id": m.recipient_id,
        "last_seen_at": (m.last_seen_at.isoformat() + "Z") if m.last_seen_at else None,
        "last_location": m.last_location, "last_location_hi": m.last_location_hi,
        "last_box": json.loads(m.last_box) if m.last_box else None,
        "last_confidence": m.last_confidence,
    }
    if with_vectors:
        d["embeddings"] = json.loads(m.embeddings or "[]")
        d["negatives"] = json.loads(m.negatives or "[]")
    return d


@router.get("/medicine-refs")
async def list_medicine_refs(recipient_id: Optional[int] = None, vectors: bool = True,
                             authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Medicines the room camera looks for (with fingerprints for the browser recogniser)."""
    from tables.medicine_ref import MedicineRef
    user = _caretaker_or_401(db, authorization)
    rid = _resolve_recipient(db, user, recipient_id)
    q = db.query(MedicineRef).filter(MedicineRef.caretaker_id == user.id)
    if rid:
        from sqlalchemy import or_
        q = q.filter(or_(MedicineRef.recipient_id == rid, MedicineRef.recipient_id.is_(None)))
    return {"medicines": [_medicine_ref_dict(m, vectors) for m in q.order_by(MedicineRef.id.asc()).all()]}


@router.post("/medicine-refs")
async def create_medicine_ref(payload: MedicineRefRequest, authorization: Optional[str] = Header(None),
                              db: Session = Depends(get_db)):
    from tables.medicine_ref import MedicineRef
    user = _caretaker_or_401(db, authorization)
    if not payload.embeddings:
        raise HTTPException(status_code=400, detail="No fingerprints in the request")
    if payload.image and len(payload.image) > 600_000:
        raise HTTPException(status_code=413, detail="Photo too large — please use a smaller picture")
    round4 = lambda vs: [[round(float(v), 4) for v in vec] for vec in (vs or [])]
    m = MedicineRef(caretaker_id=user.id, recipient_id=_resolve_recipient(db, user, payload.recipient_id),
                    name=(payload.name or "Medicine").strip()[:120], image=payload.image,
                    source=(payload.source or "upload")[:20], mode=(payload.mode or "mobilenet")[:20],
                    embeddings=json.dumps(round4(payload.embeddings)),
                    negatives=json.dumps(round4(payload.negatives)), created_at=datetime.utcnow())
    db.add(m)
    db.commit()
    return {"status": "success", "medicine": _medicine_ref_dict(m)}


@router.delete("/medicine-refs/{med_id}")
async def delete_medicine_ref(med_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medicine_ref import MedicineRef
    user = _caretaker_or_401(db, authorization)
    m = db.query(MedicineRef).filter(MedicineRef.id == med_id, MedicineRef.caretaker_id == user.id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Medicine not found")
    db.delete(m)
    db.commit()
    return {"status": "success"}


@router.post("/medicine-refs/{med_id}/seen")
async def medicine_seen(med_id: int, payload: MedicineSeenRequest, authorization: Optional[str] = Header(None),
                        db: Session = Depends(get_db)):
    """The room camera saw this medicine: remember where."""
    from tables.medicine_ref import MedicineRef
    user = _caretaker_or_401(db, authorization)
    m = db.query(MedicineRef).filter(MedicineRef.id == med_id, MedicineRef.caretaker_id == user.id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Medicine not found")
    now = datetime.utcnow()
    if payload.probable:
        # a confirmed sighting from the last 10 minutes is better than a guess: keep it
        confirmed_recently = (m.last_seen_at and (now - m.last_seen_at).total_seconds() < 600
                              and m.last_location and not m.last_location.startswith("probably "))
        if confirmed_recently:
            return {"status": "kept", "reason": "recent confirmed sighting"}
        m.last_seen_at = now
        m.last_location = ("probably " + (payload.location or "in the room"))[:300]
        m.last_location_hi = ("shayad " + (payload.location_hi or "kamre mein"))[:300]
        m.last_box = json.dumps(payload.box) if payload.box else m.last_box
        m.last_confidence = payload.confidence
        _record_sighting(db, user, m, payload, probable=True, now=now)
        db.commit()
        return {"status": "success", "probable": True}
    m.last_seen_at = now
    m.last_location = (payload.location or "")[:300] or m.last_location
    m.last_location_hi = (payload.location_hi or "")[:300] or m.last_location_hi
    m.last_box = json.dumps(payload.box) if payload.box else m.last_box
    m.last_confidence = payload.confidence
    _record_sighting(db, user, m, payload, probable=False, now=now)
    db.commit()
    return {"status": "success"}


def _record_sighting(db: Session, user, m, payload, probable: bool, now: datetime) -> None:
    """Keep a history row of where the camera found the medicine (one row per visit to a place)."""
    from tables.medicine_ref import MedicineSighting
    loc = (payload.location or "in the room")[:300]
    loc_hi = (payload.location_hi or "kamre mein")[:300]
    last = (db.query(MedicineSighting).filter(MedicineSighting.medicine_ref_id == m.id)
            .order_by(MedicineSighting.last_seen_at.desc()).first())
    same_place = last and (last.location or "") == loc and (now - last.last_seen_at).total_seconds() < 600
    if same_place:
        last.last_seen_at = now
        last.sightings = (last.sightings or 1) + 1
        last.confidence = max(last.confidence or 0, payload.confidence or 0)
        if not probable:
            last.probable = 0          # a confirmed sighting upgrades the visit
        return
    db.add(MedicineSighting(
        caretaker_id=user.id, recipient_id=m.recipient_id or _resolve_recipient(db, user, None),
        medicine_ref_id=m.id, medicine_name=m.name, location=loc, location_hi=loc_hi,
        confidence=payload.confidence, probable=1 if probable else 0, sightings=1,
        first_seen_at=now, last_seen_at=now,
    ))


@router.get("/medicine-sightings")
async def medicine_sightings(recipient_id: Optional[int] = None, days: int = Query(30, ge=1, le=365),
                             authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """History of where the room camera found each medicine (newest first) + a short summary."""
    from tables.medicine_ref import MedicineSighting
    from datetime import timedelta
    from sqlalchemy import or_
    user = _caretaker_or_401(db, authorization)
    rid = _resolve_recipient(db, user, recipient_id)
    q = db.query(MedicineSighting).filter(MedicineSighting.caretaker_id == user.id,
                                          MedicineSighting.last_seen_at >= datetime.utcnow() - timedelta(days=days))
    if rid:
        q = q.filter(or_(MedicineSighting.recipient_id == rid, MedicineSighting.recipient_id.is_(None)))
    rows = q.order_by(MedicineSighting.last_seen_at.desc()).limit(300).all()
    per_med: Dict[str, Dict] = {}
    for r in rows:
        s = per_med.setdefault(r.medicine_name, {"name": r.medicine_name, "visits": 0, "last_seen_at": None, "last_location": None})
        s["visits"] += 1
        if not s["last_seen_at"]:
            s["last_seen_at"] = r.last_seen_at.isoformat() + "Z"
            s["last_location"] = r.location
    return {
        "days": days,
        "total": len(rows),
        "medicines": list(per_med.values()),
        "sightings": [{
            "id": r.id, "medicine": r.medicine_name, "location": r.location, "location_hi": r.location_hi,
            "confidence": r.confidence, "probable": bool(r.probable), "sightings": r.sightings,
            "first_seen_at": r.first_seen_at.isoformat() + "Z", "last_seen_at": r.last_seen_at.isoformat() + "Z",
            "minutes": round((r.last_seen_at - r.first_seen_at).total_seconds() / 60, 1),
        } for r in rows],
    }


@router.get("/default-recipient")
async def default_recipient(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """The caretaker's patient to show when the page was opened without ?recipient_id."""
    user = _caretaker_or_401(db, authorization)
    rid = _resolve_recipient(db, user, None)
    if not rid:
        first = db.query(CareRecipient).filter(CareRecipient.caretaker_id == user.id).first()
        rid = first.id if first else None
    r = db.query(CareRecipient).filter(CareRecipient.id == rid).first() if rid else None
    return {"recipient_id": rid, "name": r.full_name if r else None}


@router.get("/medicine-locations")
async def medicine_locations(recipient_id: Optional[int] = None, authorization: Optional[str] = Header(None),
                             db: Session = Depends(get_db)):
    """Where each medicine was last seen, plus a ready-to-speak Hinglish answer for the assistant."""
    from services.voice_health_brief import medicine_locations_list, medicine_location_text
    user = _caretaker_or_401(db, authorization)
    rid = _resolve_recipient(db, user, recipient_id)
    return {"medicines": medicine_locations_list(db, rid, caretaker_id=user.id),
            "answer": medicine_location_text(db, rid, "Ji", caretaker_id=user.id)}


# ============================================================================
# TABLE WATCH  (water glass + medicines on the table, watched by the camera)
# ============================================================================

class TableWatchEventRequest(BaseModel):
    item: str                                  # water | medicine
    event: str                                 # baseline | taken | missing | returned | untouched_alert
    detail: Optional[str] = None
    idle_minutes: Optional[float] = None
    confidence: Optional[float] = None
    recipient_id: Optional[int] = None


@router.post("/table-watch/event")
async def log_table_watch_event(
    payload: TableWatchEventRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Save a Table Watch event; 'untouched_alert' also raises an alert on the patient's dashboard."""
    from tables.table_watch import TableWatchEvent
    from tables.medical_conditions import MedicalAlert, AlertType, AlertSeverity
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    item = (payload.item or "").lower()
    event = (payload.event or "").lower()
    if item not in ("water", "medicine") or event not in ("baseline", "taken", "missing", "returned", "untouched_alert"):
        raise HTTPException(status_code=400, detail="Unknown item or event")
    rid = _resolve_recipient(db, user, payload.recipient_id)

    row = TableWatchEvent(caretaker_id=user.id, recipient_id=rid, item=item, event=event,
                          detail=(payload.detail or "")[:500], idle_minutes=payload.idle_minutes,
                          confidence=payload.confidence, occurred_at=datetime.utcnow())
    db.add(row)
    if event == "untouched_alert" and rid:
        msg = (f"MEDICINES NOT TAKEN: the medicines are still on the table. {payload.detail or ''}".strip()
               if item == "medicine" else
               f"WATER NOT TOUCHED: the water glass has not been touched for "
               f"{int(payload.idle_minutes or 0)} minutes. {payload.detail or ''}".strip())
        db.add(MedicalAlert(care_recipient_id=rid, alert_type=AlertType.critical if item == "medicine" else AlertType.monitoring_gap,
                            severity=AlertSeverity.high if item == "medicine" else AlertSeverity.medium, message=msg[:1000]))
    db.commit()
    return {"status": "success", "id": row.id, "recipient_id": rid}


@router.get("/table-watch/events")
async def get_table_watch_events(
    recipient_id: Optional[int] = None,
    days: int = Query(7, ge=1, le=90),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Table Watch history + per-day summary (water drinks, medicines taken, alerts)."""
    from tables.table_watch import TableWatchEvent
    from datetime import timedelta
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    since = datetime.utcnow() - timedelta(days=days)
    q = db.query(TableWatchEvent).filter(TableWatchEvent.caretaker_id == user.id, TableWatchEvent.occurred_at >= since)
    rid = _resolve_recipient(db, user, recipient_id)
    if rid:
        from sqlalchemy import or_
        q = q.filter(or_(TableWatchEvent.recipient_id == rid, TableWatchEvent.recipient_id.is_(None)))
    rows = q.order_by(TableWatchEvent.occurred_at.desc()).limit(500).all()

    per_day: Dict[str, Dict[str, int]] = {}
    for r in rows:
        d = r.occurred_at.date().isoformat()
        s = per_day.setdefault(d, {"water_taken": 0, "medicine_taken": 0, "alerts": 0})
        if r.event == "taken":
            s["water_taken" if r.item == "water" else "medicine_taken"] += 1
        if r.event == "untouched_alert":
            s["alerts"] += 1
    return {
        "days": days,
        "summary": {
            "water_taken": sum(1 for r in rows if r.item == "water" and r.event == "taken"),
            "medicine_taken": sum(1 for r in rows if r.item == "medicine" and r.event == "taken"),
            "water_alerts": sum(1 for r in rows if r.item == "water" and r.event == "untouched_alert"),
            "medicine_alerts": sum(1 for r in rows if r.item == "medicine" and r.event == "untouched_alert"),
        },
        "per_day": [{"date": k, **v} for k, v in sorted(per_day.items())],
        "events": [{"id": r.id, "item": r.item, "event": r.event, "detail": r.detail,
                    "idle_minutes": r.idle_minutes, "confidence": r.confidence,
                    "occurred_at": r.occurred_at.isoformat() + "Z"} for r in rows],
    }


# ============================================================================
# CAMERA REGISTRY ENDPOINTS  (CCTV / IP Camera management)
# ============================================================================

@router.get("/cameras")
async def get_cameras(authorization: Optional[str] = Header(None)):
    """
    List all saved cameras (webcam indexes, RTSP / HTTP CCTV feeds).
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    cameras = list_cameras()
    return {"cameras": cameras, "total": len(cameras)}


@router.post("/cameras")
async def create_camera(
    request: AddCameraRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Add a new camera to the registry.

    Supports:
      - Local webcam : url="0" or url="1"  ,  cam_type="webcam"
      - IP camera    : url="rtsp://..."    ,  cam_type="rtsp"
      - HTTP MJPEG   : url="http://..."    ,  cam_type="ip"

    Credentials (username/password) are stored encrypted-at-rest only in
    the local JSON file; passwords are never returned in GET responses.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    cam_id = str(uuid.uuid4())[:8]   # short readable ID

    saved = add_camera(
        cam_id   = cam_id,
        name     = request.name.strip(),
        url      = request.url.strip(),
        cam_type = request.cam_type,
        location = request.location.strip(),
        username = request.username.strip(),
        password = request.password,
    )

    logger.info(f"User {username} added camera: {saved['name']} ({saved['url']})")
    return {"status": "success", "camera": saved}


@router.delete("/cameras/{cam_id}")
async def delete_camera(
    cam_id: str,
    authorization: Optional[str] = Header(None),
):
    """Remove a camera from the registry."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    deleted = remove_camera(cam_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")

    logger.info(f"User {username} removed camera: {cam_id}")
    return {"status": "success", "message": f"Camera {cam_id} removed"}


@router.post("/cameras/{cam_id}/test")
async def test_camera_connection(
    cam_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Quick connectivity test — tries to open the camera and grab one frame.
    Returns within ~5 seconds.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    stream_source = build_stream_url(cam_id)
    if stream_source is None:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")

    def _test():
        if isinstance(stream_source, str) and (
            stream_source.startswith("rtsp") or stream_source.startswith("http")
        ):
            cap = cv2.VideoCapture(stream_source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(stream_source)

        if not cap.isOpened():
            return False, "Could not open camera stream"

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "Camera opened but no frame received"

        h, w = frame.shape[:2]
        return True, f"OK — got {w}×{h} frame"

    try:
        ok, message = await asyncio.get_event_loop().run_in_executor(None, _test)
        return {
            "status": "online" if ok else "unreachable",
            "message": message,
            "cam_id": cam_id,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "cam_id": cam_id}


@router.post("/cameras/test-url")
async def test_camera_url(
    request: AddCameraRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Test an arbitrary URL/index before saving it.
    Same as /cameras/{id}/test but accepts a raw URL in the request body.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    url = request.url.strip()

    # Embed credentials if provided (for RTSP)
    if request.username and request.password and url.startswith("rtsp://"):
        url = url.replace("rtsp://", f"rtsp://{request.username}:{request.password}@", 1)

    # Determine source
    if url.isdigit():
        source = int(url)
    else:
        source = url

    def _test():
        if isinstance(source, str) and (source.startswith("rtsp") or source.startswith("http")):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return False, "Could not connect to camera"

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "Connected but no frame received. Check stream path/credentials."

        h, w = frame.shape[:2]
        return True, f"Connection successful — got {w}×{h} frame"

    try:
        ok, message = await asyncio.get_event_loop().run_in_executor(None, _test)
        return {"status": "online" if ok else "unreachable", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}
