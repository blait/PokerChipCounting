"""SAM3 Video Engine — load once, run many configs without restarting.

Usage:
    engine = SAM3Engine(video_path="/home/ubuntu/dealer.mp4")
    engine.set_thresholds(score_threshold_detection=0.2, new_det_thresh=0.5)
    result = engine.run(prompt="stack of poker chips", out_dir="/tmp/run1")
    engine.set_thresholds(score_threshold_detection=0.3, new_det_thresh=0.7)
    result = engine.run(prompt="stack of poker chips", out_dir="/tmp/run2")
"""
import os
import json
import time

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

TUNABLE_ATTRS = [
    "score_threshold_detection",
    "new_det_thresh",
    "assoc_iou_thresh",
    "det_nms_thresh",
    "max_trk_keep_alive",
    "init_trk_keep_alive",
    "hotstart_delay",
    "suppress_overlapping_based_on_recent_occlusion_threshold",
    "recondition_every_nth_frame",
]


class SAM3Engine:
    def __init__(self, video_path, alpha=0.45):
        self.video_path = video_path
        self.alpha = alpha

        gpus = list(range(torch.cuda.device_count()))
        print(f"[engine] building predictor on {len(gpus)} GPU(s)...", flush=True)
        t0 = time.time()
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        print(f"[engine] predictor ready ({time.time()-t0:.1f}s)", flush=True)

        self._load_frames()

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        self.frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(f)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        self.H, self.W = self.frames[0].shape[:2]
        print(f"[engine] {len(self.frames)} frames {self.W}x{self.H} @ {self.fps:.1f}fps", flush=True)

    def get_thresholds(self):
        model = self._get_model()
        return {attr: getattr(model, attr, None) for attr in TUNABLE_ATTRS}

    def set_thresholds(self, **kwargs):
        model = self._get_model()
        for k, v in kwargs.items():
            if k not in TUNABLE_ATTRS:
                raise ValueError(f"Unknown threshold: {k}. Valid: {TUNABLE_ATTRS}")
            setattr(model, k, v)
        print(f"[engine] thresholds updated: {kwargs}", flush=True)

    def _get_model(self):
        p = self.predictor
        if hasattr(p, "video_predictors"):
            return p.video_predictors[0]
        if hasattr(p, "model"):
            return p.model
        return p

    def run(self, prompt="stack of poker chips", out_dir=None, tag=None):
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        t0 = time.time()
        resp = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=self.video_path)
        )
        sid = resp["session_id"]

        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=sid,
                frame_index=0,
                text=prompt,
            )
        )

        outs = {}
        for resp in self.predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=sid)
        ):
            outs[resp["frame_index"]] = resp["outputs"]

        self.predictor.handle_request(
            request=dict(type="close_session", session_id=sid)
        )
        torch.cuda.empty_cache()

        elapsed = time.time() - t0

        result = self._build_overlay(outs, out_dir, tag, prompt)
        result["elapsed_sec"] = round(elapsed, 1)
        result["thresholds"] = self.get_thresholds()
        result["prompt"] = prompt

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "counts.json"), "w") as f:
                json.dump(result, f, indent=2)

        self._print_summary(result, tag)
        return result

    def _build_overlay(self, outs, out_dir, tag, prompt):
        rng = np.random.default_rng(0)
        color_by_id = {}
        per_frame_counts = {}
        seen_ids = set()
        first_seen_frame = {}

        vw = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "overlay.mp4")
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.W, self.H))

        for fi, frame in enumerate(self.frames):
            out = outs.get(fi, {})
            obj_ids = out.get("out_obj_ids")
            masks = out.get("out_binary_masks")
            probs = out.get("out_probs")

            draw = frame.copy() if vw else None
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
                    if m.shape != (self.H, self.W):
                        m = cv2.resize(m.astype(np.uint8), (self.W, self.H),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)

                    if oid not in color_by_id:
                        color_by_id[oid] = tuple(int(v) for v in rng.integers(80, 255, size=3))
                        first_seen_frame[oid] = fi
                    seen_ids.add(oid)
                    n += 1

                    if draw is not None:
                        color = color_by_id[oid]
                        overlay = draw.copy()
                        overlay[m] = color
                        draw = cv2.addWeighted(overlay, self.alpha, draw, 1 - self.alpha, 0)
                        ys, xs = np.where(m)
                        x0, y0 = int(xs.min()), int(ys.min())
                        x1, y1 = int(xs.max()), int(ys.max())
                        cv2.rectangle(draw, (x0, y0), (x1, y1), color, 2)
                        label = f"id{oid}"
                        if probs is not None and idx < len(probs):
                            p = float(probs[idx].item() if hasattr(probs[idx], "item") else probs[idx])
                            label += f" {p:.2f}"
                        cv2.putText(draw, label, (x0, max(12, y0 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if draw is not None:
                cv2.putText(draw, f"frame {fi}  now={n}  total unique={len(seen_ids)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vw.write(draw)
            per_frame_counts[fi] = n

        if vw:
            vw.release()

        vals = list(per_frame_counts.values())
        return {
            "per_frame_counts": per_frame_counts,
            "num_frames": len(self.frames),
            "num_unique_objs": len(color_by_id),
            "first_seen_frame": {str(k): v for k, v in first_seen_frame.items()},
            "avg_per_frame": round(sum(vals) / max(len(vals), 1), 2),
            "max_per_frame": max(vals) if vals else 0,
            "nonzero_frames": sum(1 for v in vals if v > 0),
        }

    def set_video(self, video_path):
        self.video_path = video_path
        self._load_frames()

    def run_streaming(self, prompt="stack of poker chips", out_dir=None, cancel_flag=None):
        """Generator yielding per-frame overlay data during propagation.

        Yields dicts with keys: frame_index, total_frames, count, unique_so_far,
        jpeg_bytes, elapsed_sec.  Final yield has frame_index=-1 and full result.
        """
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        t0 = time.time()
        resp = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=self.video_path)
        )
        sid = resp["session_id"]

        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=sid,
                frame_index=0,
                text=prompt,
            )
        )

        color_by_id = {}
        seen_ids = set()
        first_seen_frame = {}
        per_frame_counts = {}
        rng = np.random.default_rng(0)

        vw = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            vw = cv2.VideoWriter(
                os.path.join(out_dir, "overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.W, self.H))

        cancelled = False
        for resp in self.predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=sid)
        ):
            if cancel_flag and cancel_flag():
                cancelled = True
                break

            fi = resp["frame_index"]
            outputs = resp["outputs"]
            frame = self.frames[fi]
            draw = frame.copy()

            obj_ids = outputs.get("out_obj_ids")
            masks = outputs.get("out_binary_masks")
            probs = outputs.get("out_probs")
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
                    if m.shape != (self.H, self.W):
                        m = cv2.resize(m.astype(np.uint8), (self.W, self.H),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)

                    if oid not in color_by_id:
                        color_by_id[oid] = tuple(int(v) for v in rng.integers(80, 255, size=3))
                        first_seen_frame[oid] = fi
                    seen_ids.add(oid)
                    n += 1

                    color = color_by_id[oid]
                    overlay = draw.copy()
                    overlay[m] = color
                    draw = cv2.addWeighted(overlay, self.alpha, draw, 1 - self.alpha, 0)
                    ys, xs = np.where(m)
                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    cv2.rectangle(draw, (x0, y0), (x1, y1), color, 2)
                    label = f"id{oid}"
                    if probs is not None and idx < len(probs):
                        p = float(probs[idx].item() if hasattr(probs[idx], "item") else probs[idx])
                        label += f" {p:.2f}"
                    cv2.putText(draw, label, (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(draw, f"frame {fi}  now={n}  total unique={len(seen_ids)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if vw:
                vw.write(draw)
            per_frame_counts[fi] = n

            _, jpeg = cv2.imencode('.jpg', draw, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield {
                "frame_index": fi,
                "total_frames": len(self.frames),
                "count": n,
                "unique_so_far": len(seen_ids),
                "jpeg_bytes": jpeg.tobytes(),
                "elapsed_sec": round(time.time() - t0, 1),
            }

        if vw:
            vw.release()

        self.predictor.handle_request(
            request=dict(type="close_session", session_id=sid)
        )
        torch.cuda.empty_cache()

        vals = list(per_frame_counts.values())
        result = {
            "num_frames": len(self.frames),
            "num_unique_objs": len(color_by_id),
            "avg_per_frame": round(sum(vals) / max(len(vals), 1), 2),
            "max_per_frame": max(vals) if vals else 0,
            "nonzero_frames": sum(1 for v in vals if v > 0),
            "elapsed_sec": round(time.time() - t0, 1),
            "prompt": prompt,
            "thresholds": self.get_thresholds(),
            "cancelled": cancelled,
        }
        if out_dir:
            with open(os.path.join(out_dir, "counts.json"), "w") as f:
                json.dump(result, f, indent=2)
            result["overlay_path"] = os.path.join(out_dir, "overlay.mp4")

        yield {"frame_index": -1, "result": result}

    def _print_summary(self, result, tag):
        prefix = f"[{tag}]" if tag else "[run]"
        print(f"{prefix} unique={result['num_unique_objs']} "
              f"avg={result['avg_per_frame']} max={result['max_per_frame']} "
              f"nonzero={result['nonzero_frames']}/{result['num_frames']} "
              f"elapsed={result['elapsed_sec']}s", flush=True)
