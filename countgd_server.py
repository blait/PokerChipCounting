"""CountGD Chip Counting Server — Independent FastAPI server.

Runs separately from SAM3. Provides:
  - POST /api/count — count objects in a single image
  - POST /api/count-stacks — count chips in SAM3-detected stack regions
  - WebSocket /ws/count-video — process video frames with CountGD

Run on GPU server:
    cd /tmp && python3 countgd_server.py \
        --countgd-repo /home/ubuntu/CountGD \
        --checkpoint /home/ubuntu/CountGD/checkpoints/checkpoint_fsc147_best.pth \
        --port 8001
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import uvicorn

from countgd_engine import CountGDEngine

ORIGIN_SECRET = os.environ.get("ORIGIN_SECRET", "")
engine: CountGDEngine = None
args = None


class OriginVerifyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not ORIGIN_SECRET:
            return await call_next(request)
        if request.url.path == "/health":
            return await call_next(request)
        header_val = request.headers.get("x-origin-verify", "")
        if header_val != ORIGIN_SECRET:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return await call_next(request)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countgd-repo", default="/home/ubuntu/CountGD",
                    help="Path to CountGD repository clone")
    ap.add_argument("--checkpoint",
                    default="/home/ubuntu/CountGD/checkpoints/checkpoint_fsc147_best.pth",
                    help="Path to CountGD model checkpoint")
    ap.add_argument("--bert-path", default=None,
                    help="Path to bert-base-uncased (default: <repo>/checkpoints/bert-base-uncased)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--origin-secret", default="",
                    help="Secret header value for origin verification")
    return ap.parse_args()


@asynccontextmanager
async def lifespan(app):
    global engine, args, ORIGIN_SECRET
    args = parse_args()
    if args.origin_secret:
        ORIGIN_SECRET = args.origin_secret
    os.makedirs("/tmp/countgd_static", exist_ok=True)
    print("[countgd-server] loading model...", flush=True)
    engine = CountGDEngine(
        repo_path=args.countgd_repo,
        checkpoint=args.checkpoint,
        bert_path=args.bert_path,
    )
    print("[countgd-server] ready", flush=True)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(OriginVerifyMiddleware)
app.mount("/static", StaticFiles(directory="/tmp/countgd_static"), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse("/tmp/countgd_static/countgd.html")


@app.post("/api/count")
async def count_image(
    file: UploadFile = File(None),
    image_b64: str = Form(None),
    prompt: str = Form("poker chip"),
    box_threshold: float = Form(0.23),
):
    """Count objects in a single uploaded image or base64-encoded image."""
    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_b64:
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    if image_bgr is None:
        return JSONResponse({"error": "Failed to decode image"}, status_code=400)

    result = engine.count(image_bgr, prompt=prompt, box_threshold=box_threshold)

    overlay = _draw_overlay(image_bgr, result)
    _, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["overlay_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return result


@app.post("/api/count-stacks")
async def count_stacks(body: dict):
    """Count chips within SAM3-detected stack bounding boxes.

    Body:
        {
            "image_b64": str,  // base64 JPEG/PNG of full frame
            "stacks": [[x0,y0,x1,y1], ...],  // bounding boxes
            "prompt": "poker chip",
            "box_threshold": 0.23
        }
    """
    image_b64 = body.get("image_b64")
    if not image_b64:
        return JSONResponse({"error": "image_b64 required"}, status_code=400)

    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return JSONResponse({"error": "Failed to decode image"}, status_code=400)

    stacks = body.get("stacks", [])
    prompt = body.get("prompt", "poker chip")
    box_threshold = body.get("box_threshold", 0.23)

    result = engine.count_stacks(
        frame_bgr, stacks, prompt=prompt, box_threshold=box_threshold
    )

    overlay = _draw_stacks_overlay(frame_bgr, result)
    _, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["overlay_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return result


@app.websocket("/ws/count-video")
async def ws_count_video(websocket: WebSocket):
    """WebSocket for real-time video frame counting.

    Client sends:
        {"type": "frame", "image_b64": str, "prompt": str, "box_threshold": float}
        {"type": "frame_with_stacks", "image_b64": str, "stacks": [...], ...}
        {"type": "stop"}

    Server responds:
        {"type": "result", "count": int, "boxes": [...], "overlay_b64": str, ...}
        {"type": "stacks_result", "total_chips": int, "stacks": [...], "overlay_b64": str}
    """
    if ORIGIN_SECRET:
        header_val = websocket.headers.get("x-origin-verify", "")
        if header_val != ORIGIN_SECRET:
            await websocket.close(code=4003, reason="forbidden")
            return
    await websocket.accept()

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "frame")

            if msg_type == "stop":
                break

            image_b64 = msg.get("image_b64")
            if not image_b64:
                await websocket.send_json({"type": "error", "message": "No image"})
                continue

            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send_json({"type": "error", "message": "Decode failed"})
                continue

            prompt = msg.get("prompt", "poker chip")
            box_threshold = msg.get("box_threshold", 0.23)

            if msg_type == "frame_with_stacks":
                stacks = msg.get("stacks", [])
                result = engine.count_stacks(
                    frame_bgr, stacks, prompt=prompt, box_threshold=box_threshold
                )
                overlay = _draw_stacks_overlay(frame_bgr, result)
                _, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
                result["overlay_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")
                result["type"] = "stacks_result"
                await websocket.send_json(result)
            else:
                result = engine.count(
                    frame_bgr, prompt=prompt, box_threshold=box_threshold
                )
                overlay = _draw_overlay(frame_bgr, result)
                _, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
                result["overlay_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")
                result["type"] = "result"
                await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


def _draw_overlay(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Draw CountGD detection boxes on image."""
    draw = image_bgr.copy()
    h, w = draw.shape[:2]

    for i, box in enumerate(result.get("boxes", [])):
        cx, cy, bw, bh = box
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)

        score = result["scores"][i] if i < len(result.get("scores", [])) else 0
        color = (0, 255, 200)
        cv2.rectangle(draw, (x0, y0), (x1, y1), color, 2)
        cv2.putText(draw, f"{score:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(draw, f"Count: {result.get('count', 0)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    return draw


def _draw_stacks_overlay(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Draw per-stack chip counts on full frame."""
    draw = frame_bgr.copy()
    total = result.get("total_chips", 0)

    for i, stack in enumerate(result.get("stacks", [])):
        bbox = stack.get("bbox", [0, 0, 0, 0])
        x0, y0, x1, y1 = [int(v) for v in bbox]
        count = stack.get("count", 0)

        color = (0, 200, 255)
        cv2.rectangle(draw, (x0, y0), (x1, y1), color, 2)
        label = f"Stack#{i+1}: {count} chips"
        cv2.putText(draw, label, (x0, max(12, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(draw, f"Total Chips: {total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    return draw


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
