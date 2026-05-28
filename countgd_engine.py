"""CountGD Engine — Open-world object counting via text prompts.

Wraps the CountGD model (GroundingDINO-based) for chip counting.
Runs independently from SAM3 — no shared state or imports.

Usage:
    engine = CountGDEngine(
        repo_path="/home/ubuntu/CountGD",
        checkpoint="checkpoints/checkpoint_fsc147_best.pth",
    )
    result = engine.count(image_bgr, prompt="poker chip")
    # result = {"count": 12, "boxes": [[cx,cy,w,h], ...], "scores": [...]}
"""
import sys
import time
import random
import argparse

import cv2
import numpy as np
import torch
from PIL import Image


class CountGDEngine:
    def __init__(self, repo_path: str, checkpoint: str, device="cuda",
                 bert_path: str = None, config: str = None):
        self.device = device
        self.repo_path = repo_path

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        self._build_model(checkpoint, bert_path, config)
        self._build_transform()
        print(f"[countgd] model ready on {device}", flush=True)

    def _build_model(self, checkpoint, bert_path, config):
        from util.slconfig import SLConfig

        cfg_path = config or f"{self.repo_path}/config/cfg_fsc147_vit_b.py"
        cfg = SLConfig.fromfile(cfg_path)

        bert = bert_path or f"{self.repo_path}/checkpoints/bert-base-uncased"
        cfg.merge_from_dict({"text_encoder_type": bert})

        cfg_dict = cfg._cfg_dict.to_dict()
        args = argparse.Namespace(**cfg_dict)
        args.device = self.device

        from models.registry import MODULE_BUILD_FUNCS
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        self.model, _, _ = build_func(args)

        ckpt_path = checkpoint if "/" in checkpoint else f"{self.repo_path}/{checkpoint}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def _build_transform(self):
        import datasets_inference.transforms as T
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    def count(self, image_bgr: np.ndarray, prompt: str = "poker chip",
              box_threshold: float = 0.23) -> dict:
        """Count objects in a BGR image using a text prompt.

        Args:
            image_bgr: OpenCV BGR image (H, W, 3), uint8.
            prompt: Text describing what to count.
            box_threshold: Confidence threshold for detections.

        Returns:
            {"count": int, "boxes": list of [cx, cy, w, h] normalized,
             "scores": list of float}
        """
        t0 = time.time()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        input_image, target = self.transform(
            pil_image, {"exemplars": torch.tensor([])}
        )
        input_image = input_image.to(self.device)
        input_exemplar = target["exemplars"].to(self.device)

        caption = prompt.strip()
        if not caption.endswith("."):
            caption += " ."

        with torch.no_grad():
            outputs = self.model(
                input_image.unsqueeze(0),
                [input_exemplar],
                [torch.tensor([0]).to(self.device)],
                captions=[caption],
            )

        logits = outputs["pred_logits"][0].sigmoid()
        boxes = outputs["pred_boxes"][0]

        max_scores = logits.max(dim=-1)[0]
        mask = max_scores > box_threshold

        filtered_boxes = boxes[mask].cpu().numpy().tolist()
        filtered_scores = max_scores[mask].cpu().numpy().tolist()

        elapsed = time.time() - t0
        return {
            "count": len(filtered_boxes),
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    def count_stacks(self, frame_bgr: np.ndarray, stack_bboxes: list,
                     prompt: str = "poker chip",
                     box_threshold: float = 0.23) -> dict:
        """Count chips in multiple stack crops from a single frame.

        Args:
            frame_bgr: Full frame (H, W, 3) BGR.
            stack_bboxes: List of [x0, y0, x1, y1] pixel coordinates for each stack.
            prompt: Text prompt for counting.
            box_threshold: Confidence threshold.

        Returns:
            {"stacks": [{"bbox": [...], "count": int, "boxes": [...], "scores": [...]}],
             "total_chips": int, "elapsed_ms": float}
        """
        t0 = time.time()
        results = []
        total = 0

        for bbox in stack_bboxes:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(frame_bgr.shape[1], x1)
            y1 = min(frame_bgr.shape[0], y1)

            if x1 <= x0 or y1 <= y0:
                results.append({"bbox": bbox, "count": 0, "boxes": [], "scores": []})
                continue

            crop = frame_bgr[y0:y1, x0:x1]
            r = self.count(crop, prompt=prompt, box_threshold=box_threshold)
            r["bbox"] = bbox
            total += r["count"]
            results.append(r)

        elapsed = time.time() - t0
        return {
            "stacks": results,
            "total_chips": total,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    def count_from_sam3_result(self, frame_bgr: np.ndarray, masks, obj_ids,
                               prompt: str = "poker chip",
                               box_threshold: float = 0.23,
                               min_mask_area: int = 100) -> dict:
        """Count chips using SAM3 mask outputs directly.

        Args:
            frame_bgr: Full frame (H, W, 3) BGR.
            masks: List/array of binary masks (N, H, W).
            obj_ids: List of object IDs corresponding to masks.
            prompt: Text prompt.
            box_threshold: Confidence threshold.
            min_mask_area: Skip masks smaller than this pixel count.

        Returns:
            {"chips_per_stack": {obj_id: count}, "total_chips": int,
             "details": {obj_id: {"count", "boxes", "scores", "bbox"}}}
        """
        chips_per_stack = {}
        details = {}
        total = 0

        for i, oid in enumerate(obj_ids):
            m = masks[i]
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            m = np.asarray(m).astype(bool)
            if m.ndim == 3:
                m = m[0]
            if not m.any() or m.sum() < min_mask_area:
                chips_per_stack[oid] = 0
                continue

            ys, xs = np.where(m)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())

            pad = max((x1 - x0), (y1 - y0)) // 10
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(frame_bgr.shape[1], x1 + pad)
            y1 = min(frame_bgr.shape[0], y1 + pad)

            crop = frame_bgr[y0:y1, x0:x1]
            r = self.count(crop, prompt=prompt, box_threshold=box_threshold)

            chips_per_stack[oid] = r["count"]
            details[oid] = {**r, "bbox": [x0, y0, x1, y1]}
            total += r["count"]

        return {
            "chips_per_stack": chips_per_stack,
            "total_chips": total,
            "details": details,
        }
