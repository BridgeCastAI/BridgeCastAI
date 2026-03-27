"""
BridgeCast AI — Uni-Sign API Server
Runs on Azure GPU VM. Accepts video input from the frontend and returns
sign language recognition results via Uni-Sign inference.
"""

import argparse
import os
import sys
import tempfile
import time
import json
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="BridgeCast AI - Sign Recognition API")

# CORS: Allow frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
model = None
model_loaded = False


def load_model():
    """Load Uni-Sign model (called once at startup)"""
    global model, model_loaded

    unisign_dir = os.path.expanduser("~/Uni-Sign")
    if not os.path.exists(unisign_dir):
        print("❌ Uni-Sign not found. Run 02_setup_unisign.sh first.")
        return

    sys.path.insert(0, unisign_dir)

    try:
        print("Loading Uni-Sign model...")
        start = time.time()

        weight_path = os.path.join(unisign_dir, "weights", "wlasl_pose_only_islr.pth")
        if not os.path.exists(weight_path):
            print(f"❌ Weight file not found: {weight_path}")
            print("Run 02_setup_unisign.sh to download weights.")
            return

        # Store config for subprocess-based inference
        # Uni-Sign uses deepspeed + argparse internally, so we call it as a subprocess
        model = {
            "unisign_dir": unisign_dir,
            "weight_path": weight_path,
        }

        elapsed = time.time() - start
        print(f"✅ Model config ready in {elapsed:.1f}s")
        model_loaded = True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Falling back to health-check only mode.")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    import torch
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Sign language recognition endpoint.
    Accepts a video file (mp4, webm, etc.) and returns the recognized text
    via Uni-Sign inference pipeline.
    """
    if not model_loaded:
        raise HTTPException(503, "Model not loaded yet. Check /health endpoint.")

    # Save uploaded video to temp file
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()

        # Run Uni-Sign inference via subprocess
        # Uni-Sign uses deepspeed internally, so subprocess is the safest way
        import subprocess
        cmd = [
            sys.executable, "-m", "demo.online_inference",
            "--online_video", tmp_path,
            "--finetune", model["weight_path"],
        ]
        proc = subprocess.run(
            cmd,
            cwd=model["unisign_dir"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Parse "Prediction result is: <word>" from output
        result = "unknown"
        for line in proc.stdout.splitlines():
            if "Prediction result is:" in line:
                result = line.split("Prediction result is:")[-1].strip()
                break

        if proc.returncode != 0 and result == "unknown":
            raise RuntimeError(f"Inference failed: {proc.stderr[-500:]}")

        elapsed = time.time() - start

        return JSONResponse({
            "text": result,
            "latency_ms": round(elapsed * 1000),
        })

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


@app.post("/predict/frames")
async def predict_frames(video: UploadFile = File(...)):
    """
    Real-time recognition endpoint for short clips (1-3 seconds).
    Designed for browser MediaRecorder chunks sent from the frontend.
    """
    return await predict(video)


# ===================================================================
# FACIAL EXPRESSION RECOGNITION (RTMPose keypoint-based)
# ===================================================================
# RTMPose extracts 133 keypoints including 18 face landmarks.
# We use geometric ratios between face keypoints to classify expressions
# without any additional model — just math on the keypoints Uni-Sign
# already produces.
#
# Face keypoint indices (RTMPose wholebody):
#   Left eye:  23-26   Right eye: 27-30
#   Nose:      31      Mouth:     32-35
#   Left brow: 36-38   Right brow: 39-41
# ===================================================================

def classify_expression_from_keypoints(face_keypoints):
    """
    Classify facial expression from RTMPose face keypoints.

    Uses geometric ratios:
    - Mouth openness (mouth height / mouth width)
    - Brow raise (brow-eye distance)
    - Eye openness (eye height)

    Returns: expression name + confidence
    """
    if face_keypoints is None or len(face_keypoints) < 18:
        return {"expression": "neutral", "confidence": 0.0}

    try:
        kp = np.array(face_keypoints)

        # Mouth landmarks (indices relative to face keypoints)
        mouth_top = kp[10]       # upper lip
        mouth_bottom = kp[12]    # lower lip
        mouth_left = kp[9]       # left corner
        mouth_right = kp[11]     # right corner

        mouth_height = abs(mouth_bottom[1] - mouth_top[1])
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        mouth_ratio = mouth_height / max(mouth_width, 1e-6)

        # Mouth corner angle (smile = corners up)
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        left_corner_up = mouth_center_y - mouth_left[1]
        right_corner_up = mouth_center_y - mouth_right[1]
        smile_score = (left_corner_up + right_corner_up) / 2

        # Brow landmarks
        left_brow = kp[0:3]      # left brow points
        right_brow = kp[3:6]     # right brow points
        left_eye_top = kp[6]     # left eye top
        right_eye_top = kp[8]    # right eye top

        # Brow raise = distance between brow and eye top
        left_brow_raise = abs(left_brow[1][1] - left_eye_top[1])
        right_brow_raise = abs(right_brow[1][1] - right_eye_top[1])
        brow_raise_avg = (left_brow_raise + right_brow_raise) / 2

        # Normalize brow raise by face height
        face_height = abs(kp[0][1] - mouth_bottom[1])
        norm_brow_raise = brow_raise_avg / max(face_height, 1e-6)

        # Classification with confidence
        if mouth_ratio > 0.4:
            # Mouth wide open
            if norm_brow_raise > 0.25:
                return {"expression": "surprised", "confidence": min(mouth_ratio + norm_brow_raise, 1.0)}
            else:
                return {"expression": "speaking", "confidence": min(mouth_ratio, 1.0)}
        elif smile_score > 0.02:
            conf = min(abs(smile_score) * 10, 1.0)
            if smile_score > 0.05:
                return {"expression": "big_smile", "confidence": conf}
            return {"expression": "smile", "confidence": conf}
        elif smile_score < -0.02:
            return {"expression": "frown", "confidence": min(abs(smile_score) * 10, 1.0)}
        elif norm_brow_raise > 0.28:
            return {"expression": "brow_raise", "confidence": min(norm_brow_raise * 2, 1.0)}
        else:
            return {"expression": "neutral", "confidence": 0.8}

    except Exception as e:
        print(f"Expression classification error: {e}")
        return {"expression": "neutral", "confidence": 0.0}


@app.post("/expression")
async def detect_expression(video: UploadFile = File(...)):
    """
    Extract facial expression from video using RTMPose keypoints.
    Returns expression classification without needing a separate model.
    """
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()

        # Use RTMPose to extract keypoints (same as Uni-Sign pipeline)
        try:
            from rtmlib import Wholebody
            wholebody = Wholebody(to_openpose=False)

            import cv2
            cap = cv2.VideoCapture(tmp_path)
            face_keypoints_all = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                keypoints = wholebody(frame)
                if keypoints is not None and len(keypoints) > 0:
                    # Extract face keypoints (indices 23-41 in wholebody 133)
                    person_kp = keypoints[0] if len(keypoints.shape) > 2 else keypoints
                    if len(person_kp) >= 42:
                        face_kp = person_kp[23:41]
                        face_keypoints_all.append(face_kp.tolist())

            cap.release()

            # Classify expression from average keypoints across frames
            if face_keypoints_all:
                avg_kp = np.mean(face_keypoints_all, axis=0)
                result = classify_expression_from_keypoints(avg_kp)
            else:
                result = {"expression": "neutral", "confidence": 0.0}

        except ImportError:
            # RTMPose not available — return neutral
            result = {"expression": "neutral", "confidence": 0.0, "note": "rtmlib not installed"}

        elapsed = time.time() - start
        result["latency_ms"] = round(elapsed * 1000)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(500, f"Expression detection failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
