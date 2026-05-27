"""Quick prompt sweep on frame 0 of the poker video using SAM3 image predictor."""
import argparse
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def extract_results(state):
    """Try hard to read boxes/scores/masks from Sam3Processor state."""
    candidates = {}
    for attr in ("boxes", "bboxes", "box", "pred_boxes"):
        if hasattr(state, attr):
            candidates["boxes"] = getattr(state, attr)
            break
    for attr in ("scores", "confidences", "pred_scores"):
        if hasattr(state, attr):
            candidates["scores"] = getattr(state, attr)
            break
    for attr in ("masks", "mask", "pred_masks"):
        if hasattr(state, attr):
            candidates["masks"] = getattr(state, attr)
            break
    if not candidates and isinstance(state, dict):
        candidates = state
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "poker chip",
            "poker chips",
            "poker chip stack",
            "stack of poker chips",
            "casino chip",
            "gambling chip",
            "chip stack",
            "pile of chips",
            "token",
        ],
    )
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, bgr = cap.read()
    cap.release()
    assert ok, "failed to read frame"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    W, H = img.size
    print(f"[info] frame {args.frame_idx}: {W}x{H}", flush=True)

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    print("[info] building image model...", flush=True)
    model = build_sam3_image_model(bpe_path=bpe_path)

    report = []
    for thr in args.thresholds:
        processor = Sam3Processor(model, confidence_threshold=thr)
        state = processor.set_image(img)
        for prompt in args.prompts:
            processor.reset_all_prompts(state)
            s = processor.set_text_prompt(state=state, prompt=prompt)
            keys = extract_results(s)

            # Normalize to count
            n = 0
            boxes = keys.get("boxes") if isinstance(keys, dict) else None
            if boxes is not None:
                try:
                    n = len(boxes)
                except Exception:
                    try:
                        n = boxes.shape[0]
                    except Exception:
                        n = -1
            line = f"threshold={thr:<4}  prompt={prompt!r:35s} -> {n} detections"
            print(line, flush=True)
            report.append(line)

            # Plot if anything detected (fallback: plot anyway for first thr)
            if n > 0 or thr == args.thresholds[0]:
                try:
                    from sam3.visualization_utils import plot_results
                    plt.close("all")
                    fig = plt.figure(figsize=(10, 6))
                    plot_results(img, s)
                    plt.title(f"thr={thr} prompt={prompt!r}  n={n}")
                    tag = prompt.replace(" ", "_")
                    out = os.path.join(args.out_dir, f"thr{thr}_{tag}.png")
                    plt.savefig(out, dpi=120, bbox_inches="tight")
                    plt.close("all")
                except Exception as e:
                    print(f"  (plot skipped: {e})", flush=True)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report))
    print(f"[done] report -> {args.out_dir}/report.txt", flush=True)


if __name__ == "__main__":
    main()
