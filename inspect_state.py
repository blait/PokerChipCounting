"""Inspect what Sam3Processor state actually contains after set_text_prompt."""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def dump(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            dump(v, prefix + "." + str(k))
    elif hasattr(x, "shape"):
        print(f"{prefix}: tensor shape={tuple(x.shape)} dtype={x.dtype}")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}: list/tuple len={len(x)}")
        if x and not isinstance(x[0], (int, float, str)):
            dump(x[0], prefix + "[0]")
    else:
        s = str(x)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"{prefix}: {type(x).__name__} {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame-idx", type=int, default=60)
    ap.add_argument("--prompt", default="poker chip")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    img = Image.fromarray(rgb)

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    print("[info] building image model...", flush=True)
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    state = processor.set_image(img)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=args.prompt)

    print("=== STATE TYPE ===", type(state).__name__)
    if isinstance(state, dict):
        def walk(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"  {prefix}{k}: dict keys={list(v.keys())[:8]}")
                    walk(v, prefix + "  ")
                elif hasattr(v, "shape"):
                    print(f"  {prefix}{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    sample = ""
                    if v and hasattr(v[0], "shape"):
                        sample = f" [0].shape={tuple(v[0].shape)}"
                    elif v and not isinstance(v[0], (int, float, str)):
                        sample = f" [0]={type(v[0]).__name__}"
                    print(f"  {prefix}{k}: {type(v).__name__} len={len(v)}{sample}")
                else:
                    s = repr(v)
                    if len(s) > 120: s = s[:120] + "..."
                    print(f"  {prefix}{k}: {type(v).__name__} {s}")
        walk(state)

    # Try to actually draw something
    def lookup(d, keys):
        if not isinstance(d, dict): return None
        for k in keys:
            if k in d:
                return d[k]
        for v in d.values():
            if isinstance(v, dict):
                got = lookup(v, keys)
                if got is not None:
                    return got
        return None

    boxes = lookup(state, ["boxes", "bboxes", "pred_boxes", "det_boxes"])
    scores = lookup(state, ["scores", "confidences", "pred_scores", "det_scores"])
    masks = lookup(state, ["masks", "pred_masks", "det_masks"])
    print(f"boxes={'yes' if boxes is not None else 'no'}  scores={'yes' if scores is not None else 'no'}  masks={'yes' if masks is not None else 'no'}")

    if boxes is not None:
        b = boxes.detach().cpu() if hasattr(boxes, "detach") else torch.tensor(boxes)
        print("boxes raw[0:3]:", b[:3])
        print("boxes max/min per col:", b.float().min(0).values, b.float().max(0).values)

    # Draw detections on image (assuming normalized cxcywh or xywh)
    out = bgr.copy()
    if boxes is not None and len(boxes) > 0:
        bb = boxes.detach().cpu().float().numpy() if hasattr(boxes, "detach") else np.asarray(boxes, dtype=np.float32)
        ss = None
        if scores is not None:
            ss = scores.detach().cpu().float().numpy() if hasattr(scores, "detach") else np.asarray(scores, dtype=np.float32)
        # Try two interpretations
        for i, row in enumerate(bb):
            # cxcywh normalized -> xyxy pixel
            cx, cy, w, h = row[:4]
            if max(row[:4]) <= 1.5:  # normalized
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
            else:
                # assume xyxy pixel
                x1, y1, x2, y2 = map(int, row[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if ss is not None and i < len(ss):
                cv2.putText(out, f"{ss[i]:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(args.out, out)
        print(f"[done] {args.out}  (drew {len(bb)} boxes)")
    else:
        print("no boxes field found directly on state")


if __name__ == "__main__":
    main()
