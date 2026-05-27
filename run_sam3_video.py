"""Run SAM 3 video predictor on pokervideo.mp4 with 'poker chip stack' text prompt.

Outputs per frame:
- visualization PNGs (every N frames)
- per-frame masks + bboxes as a .npz
- a short MP4 overlay
"""
import os
import sys
import json
import argparse
import glob

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def propagate(predictor, session_id):
    out = {}
    for resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        out[resp["frame_index"]] = resp["outputs"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default="poker chip stack")
    ap.add_argument("--vis-stride", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gpus = list(range(torch.cuda.device_count()))
    print(f"[info] GPUs: {gpus}", flush=True)

    print("[info] building predictor (downloads weights from HF on first run)...", flush=True)
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    print(f"[info] loading video frames for vis: {args.video}", flush=True)
    cap = cv2.VideoCapture(args.video)
    frames_rgb = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    H, W = frames_rgb[0].shape[:2]
    print(f"[info] loaded {len(frames_rgb)} frames, {W}x{H}", flush=True)

    print("[info] start_session...", flush=True)
    resp = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.video)
    )
    sid = resp["session_id"]

    print(f"[info] add_prompt text='{args.prompt}' frame=0", flush=True)
    resp = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=sid,
            frame_index=0,
            text=args.prompt,
        )
    )
    out0 = resp["outputs"]
    print(f"[info] frame-0 detections: {len(out0.get('masks', []))} objects", flush=True)

    print("[info] propagating through video...", flush=True)
    outputs_per_frame = propagate(predictor, sid)
    print(f"[info] propagated {len(outputs_per_frame)} frames", flush=True)

    per_frame_counts = {
        fi: len(o.get("masks", [])) for fi, o in outputs_per_frame.items()
    }
    print("[info] per-frame object counts (first 10):", flush=True)
    for fi in sorted(per_frame_counts)[:10]:
        print(f"  frame {fi}: {per_frame_counts[fi]} objects", flush=True)

    with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
        json.dump(
            {
                "video": args.video,
                "prompt": args.prompt,
                "num_frames": len(frames_rgb),
                "WxH": [W, H],
                "per_frame_counts": per_frame_counts,
            },
            f,
            indent=2,
        )

    vis = prepare_masks_for_visualization(outputs_per_frame)
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[info] writing vis every {args.vis_stride} frames to {vis_dir}", flush=True)
    for fi in range(0, len(frames_rgb), args.vis_stride):
        plt.close("all")
        visualize_formatted_frame_output(
            fi,
            frames_rgb,
            outputs_list=[vis],
            titles=[f"SAM3 — {args.prompt}"],
            figsize=(8, 5),
        )
        plt.savefig(os.path.join(vis_dir, f"frame_{fi:04d}.png"), dpi=120, bbox_inches="tight")

    print("[info] writing overlay mp4...", flush=True)
    overlay_path = os.path.join(args.out_dir, "overlay.mp4")
    vw = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    rng = np.random.default_rng(0)
    color_by_id = {}
    for fi in range(len(frames_rgb)):
        frame = frames_rgb[fi].copy()
        out = outputs_per_frame.get(fi, {})
        masks = out.get("masks", [])
        obj_ids = out.get("obj_ids", list(range(len(masks))))
        for m, oid in zip(masks, obj_ids):
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            m = np.asarray(m).astype(bool)
            if m.ndim == 3:
                m = m[0]
            if m.shape != frame.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if oid not in color_by_id:
                color_by_id[oid] = rng.integers(60, 255, size=3).tolist()
            c = color_by_id[oid]
            overlay = frame.copy()
            overlay[m] = c
            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[done] overlay: {overlay_path}", flush=True)


if __name__ == "__main__":
    main()
