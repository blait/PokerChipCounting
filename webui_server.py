"""SAM3 Chip Counting Web UI — FastAPI server.

Run from /tmp/ on GPU server:
    cd /tmp && PYTORCH_ALLOC_CONF=expandable_segments:True python3 webui_server.py --video /home/ubuntu/dealer_540p.mp4
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import time
import threading
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from sam3_engine import SAM3Engine, TUNABLE_ATTRS

ORIGIN_SECRET = os.environ.get("ORIGIN_SECRET", "")

engine: SAM3Engine = None
state = {"status": "loading", "frame": 0, "total": 0}
run_history = []
runs_dir = "/tmp/chipcounting_runs"
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
    ap.add_argument("--video", default="/home/ubuntu/dealer_540p.mp4")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--video-dir", default="/home/ubuntu",
                    help="Directory to scan for video files")
    ap.add_argument("--origin-secret", default="",
                    help="Secret header value for CloudFront origin verification")
    return ap.parse_args()


@asynccontextmanager
async def lifespan(app):
    global engine, state, args, ORIGIN_SECRET
    args = parse_args()
    if args.origin_secret:
        ORIGIN_SECRET = args.origin_secret
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs("/tmp/static", exist_ok=True)
    engine = SAM3Engine(video_path=args.video)
    state["status"] = "idle"
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(OriginVerifyMiddleware)
app.mount("/static", StaticFiles(directory="/tmp/static"), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse("/tmp/static/index.html")


@app.get("/api/thresholds")
async def get_thresholds():
    return engine.get_thresholds()


@app.post("/api/thresholds")
async def set_thresholds(body: dict):
    valid = {}
    for k, v in body.items():
        if k in TUNABLE_ATTRS:
            valid[k] = v
    if valid:
        engine.set_thresholds(**valid)
    return {"updated": valid}


@app.get("/api/videos")
async def list_videos():
    patterns = ["*.mp4", "*.avi", "*.mov"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(args.video_dir, pat)))
    videos = [{"path": f, "name": os.path.basename(f)} for f in sorted(files)]
    return {"videos": videos, "current": engine.video_path}


@app.post("/api/video")
async def set_video(body: dict):
    path = body.get("path", "")
    if not os.path.exists(path):
        return JSONResponse({"error": f"File not found: {path}"}, status_code=400)
    engine.set_video(path)
    return {"status": "ok", "video": path, "frames": len(engine.frames),
            "resolution": f"{engine.W}x{engine.H}"}


@app.get("/api/status")
async def get_status():
    return state


@app.get("/api/history")
async def get_history():
    return {"runs": run_history}


@app.get("/api/download/{run_id}")
async def download_overlay(run_id: str):
    path = os.path.join(runs_dir, run_id, "overlay.mp4")
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4",
                            filename=f"overlay_{run_id}.mp4")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    if ORIGIN_SECRET:
        header_val = websocket.headers.get("x-origin-verify", "")
        if header_val != ORIGIN_SECRET:
            await websocket.close(code=4003, reason="forbidden")
            return
    await websocket.accept()

    if state["status"] == "running":
        await websocket.send_json({"type": "error", "message": "A run is already in progress"})
        await websocket.close()
        return

    try:
        msg = await websocket.receive_json()
        prompt = msg.get("prompt", "stack of poker chips")
        thresholds = msg.get("thresholds", {})

        if thresholds:
            valid = {k: v for k, v in thresholds.items() if k in TUNABLE_ATTRS}
            if valid:
                engine.set_thresholds(**valid)

        run_id = f"run_{int(time.time())}"
        out_dir = os.path.join(runs_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)

        state["status"] = "running"
        state["frame"] = 0
        state["total"] = len(engine.frames)

        cancel_requested = False

        def check_cancel():
            return cancel_requested

        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_in_thread():
            try:
                for item in engine.run_streaming(
                    prompt=prompt, out_dir=out_dir, cancel_flag=check_cancel
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except torch.cuda.OutOfMemoryError as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"frame_index": -1, "error": str(e)})
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"frame_index": -1, "error": str(e)})

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                # Check for client messages (cancel)
                try:
                    client_msg = await asyncio.wait_for(
                        websocket.receive_json(), timeout=0.05)
                    if client_msg.get("type") == "cancel":
                        cancel_requested = True
                except asyncio.TimeoutError:
                    pass
                except WebSocketDisconnect:
                    cancel_requested = True
                    break

                # Get frames from queue
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
                    continue

                if item["frame_index"] == -1:
                    if "error" in item:
                        await websocket.send_json({
                            "type": "error", "message": item["error"]})
                    else:
                        result = item["result"]
                        result["run_id"] = run_id
                        run_history.append(result)
                        await websocket.send_json({
                            "type": "done",
                            "result": result,
                            "download_url": f"/api/download/{run_id}",
                        })
                    break
                else:
                    state["frame"] = item["frame_index"]
                    jpeg_b64 = base64.b64encode(item["jpeg_bytes"]).decode("ascii")
                    await websocket.send_json({
                        "type": "frame",
                        "frame_index": item["frame_index"],
                        "total_frames": item["total_frames"],
                        "count": item["count"],
                        "unique_so_far": item["unique_so_far"],
                        "elapsed_sec": item["elapsed_sec"],
                        "jpeg_b64": jpeg_b64,
                    })

            except WebSocketDisconnect:
                cancel_requested = True
                break

        thread.join(timeout=30)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        state["status"] = "idle"
        state["frame"] = 0


if __name__ == "__main__":
    import torch
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
