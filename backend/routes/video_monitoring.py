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
        await fm.send_message(message)
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
                recipient_id=recipient_id
            )
        
        return {
            "status": "success",
            "process_id": process_id,
            "message": "Video uploaded successfully. Processing started."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def process_video_direct(process_id: str, input_path: str, output_path: str, user_email: str, caretaker_id: int, recipient_id: Optional[int] = None):
    """
    Direct in-process video analysis using UnitedMonitor.
    Runs in a thread pool managed by FastAPI BackgroundTasks.
    """
    try:
        logger.info(f"Starting direct analysis for {process_id}")
        
        # Initialize Monitor
        monitor = UnitedMonitor(
            sensitivity="medium",
            inactivity_threshold=30,
            is_live=False,
            process_every_n_frames=1
        )
        
        # Open Video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open input video")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize Writer - use a standard avi container for the intermediate analysis file
        # to avoid Windows DLL issues with H.264 in OpenCV directly. 
        # We will transcode to H.264 using FFmpeg afterwards.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_avi_path = str(output_path).replace(".mp4", "_raw.avi")
        writer = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error(f"Failed to initialize VideoWriter with mp4v for {temp_avi_path}")
            # Try one more fallback
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise Exception("Could not initialize VideoWriter with any available codec")
        
        logger.info(f"Using intermediate AVI file: {temp_avi_path}")
        
        frame_count = 0
        falls_detected = []
        total_inactivity_frames = 0
        last_alert_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process Frame
            results = monitor.process_frame(frame)
            
            # Draw Interface
            display_frame = draw_united_interface(frame, results)
            
            # Write Frame
            writer.write(display_frame)
            
            # Track Inactivity
            if results["inactivity"].get("alert"):
                total_inactivity_frames += 1
            if results["fall"]["fall_event_fired"]:
                timestamp = results["global_state"].get("timestamp", datetime.now().isoformat())
                falls_detected.append({
                    "timestamp": timestamp,
                    "message": "Fall Detected",
                    "frame": frame_count
                })
                logger.info(f"Fall Event Recorded: Frame {frame_count}")
            
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
        if os.path.exists(temp_avi_path) and os.path.getsize(temp_avi_path) > 0:
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
        
        # Final Status Update
        if process_id in process_status:
            process_status[process_id].update({
                "status": "completed",
                "progress": 100,
                "falls_detected": falls_detected,
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
                    body=f"A fall was detected during the analysis of your uploaded video. Please log in to your dashboard to review the details and playback."
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
        
        return {
            "total_uploads": total_uploads,
            "total_alerts": total_alerts
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
    days: int = Query(7, description="Number of days to analyze"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Get aggregated video analysis stats for a specific recipient.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    # 1. Get all analysis records for this recipient within the time range
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    analyses = db.query(VideoAnalysis).filter(
        VideoAnalysis.recipient_id == recipient_id,
        VideoAnalysis.timestamp >= cutoff_date
    ).all()
    
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
