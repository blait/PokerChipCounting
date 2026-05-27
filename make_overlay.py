"""Build overlay.mp4 from SAM3 video outputs using the correct keys.

Re-runs the video predictor once and streams mask overlays per frame.
"""
import argparse
import os
import json

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--prompt", default="poker chip")
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    gpus = list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    H, W = frames[0].shape[:2]
    print(f"[info] {len(frames)} frames {W}x{H} @ {fps:.2f}fps", flush=True)

    resp = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.video)
    )
    sid = resp["session_id"]
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=sid,
            frame_index=0,
            text=args.prompt,
        )
    )
    print(f"[info] propagating '{args.prompt}' ...", flush=True)
    outs = propagate(predictor, sid)
    print(f"[info] got {len(outs)} frames", flush=True)

    rng = np.random.default_rng(0)
    color_by_id = {}
    per_frame_counts = {}
    seen_ids = set()
    first_seen_frame = {}

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi, frame in enumerate(frames):
        out = outs.get(fi, {})
        obj_ids = out.get("out_obj_ids")
        masks = out.get("out_binary_masks")
        boxes = out.get("out_boxes_xywh")
        probs = out.get("out_probs")

        n = 0
        if obj_ids is not None:
            ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
            for idx, oid in enumerate(ids):
                m = masks[idx]
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                m = np.asarray(m).astype(bool)
                if m.ndim == 3:
                    m = m[0]
                if not m.any():
                    continue
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                if oid not in color_by_id:
                    color_by_id[oid] = tuple(int(v) for v in rng.integers(80, 255, size=3))
                    first_seen_frame[oid] = fi
                seen_ids.add(oid)
                color = color_by_id[oid]
                overlay = frame.copy()
                overlay[m] = color
                frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

                ys, xs = np.where(m)
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                label = f"id{oid}"
                if probs is not None and idx < len(probs):
                    p = float(probs[idx].item() if hasattr(probs[idx], "item") else probs[idx])
                    label += f" {p:.2f}"
                cv2.putText(frame, label, (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                n += 1

        cv2.putText(frame, f"frame {fi}  now={n}  total unique={len(seen_ids)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        per_frame_counts[fi] = n
        vw.write(frame)

    vw.release()
    with open(args.counts, "w") as f:
        json.dump({"per_frame_counts": per_frame_counts,
                   "num_frames": len(frames),
                   "num_unique_objs": len(color_by_id),
                   "first_seen_frame": {str(k): v for k, v in first_seen_frame.items()},
                   "prompt": args.prompt}, f, indent=2)
    nonzero = [c for c in per_frame_counts.values() if c > 0]
    print(f"[done] {args.out}  unique_objs={len(color_by_id)} "
          f"frames_with_det={len(nonzero)}/{len(frames)}  "
          f"avg_per_frame={np.mean(list(per_frame_counts.values())):.2f}",
          flush=True)


if __name__ == "__main__":
    main()
