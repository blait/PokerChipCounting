"""Unified sweep — loads model ONCE, iterates configs by changing attrs.

Usage:
    python3 /tmp/sweep.py --video /home/ubuntu/dealer.mp4 --out-root /home/ubuntu/sam3-runs/sweep_v3

Runs from /tmp/ to avoid namespace shadow. Copy there before running:
    cp /home/ubuntu/sam3_engine.py /tmp/
    cp /home/ubuntu/sweep.py /tmp/
    cd /tmp && python3 sweep.py --video ...
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from sam3_engine import SAM3Engine

CONFIGS = [
    dict(tag="A_s03_n07", score_threshold_detection=0.3, new_det_thresh=0.7),
    dict(tag="B_s03_n05", score_threshold_detection=0.3, new_det_thresh=0.5),
    dict(tag="C_s02_n07", score_threshold_detection=0.2, new_det_thresh=0.7),
    dict(tag="D_s02_n05", score_threshold_detection=0.2, new_det_thresh=0.5),
    dict(tag="E_s01_n05", score_threshold_detection=0.1, new_det_thresh=0.5),
    dict(tag="F_s01_n03", score_threshold_detection=0.1, new_det_thresh=0.3),
    dict(tag="G_s03_n07_nms03", score_threshold_detection=0.3, new_det_thresh=0.7, det_nms_thresh=0.3),
    dict(tag="H_s02_n05_nms03", score_threshold_detection=0.2, new_det_thresh=0.5, det_nms_thresh=0.3),
]

PROMPT = "stack of poker chips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--configs-json", help="Path to JSON file with custom configs (overrides built-in)")
    args = ap.parse_args()

    configs = CONFIGS
    if args.configs_json:
        with open(args.configs_json) as f:
            configs = json.load(f)

    os.makedirs(args.out_root, exist_ok=True)
    engine = SAM3Engine(video_path=args.video)

    print(f"\n{'='*60}")
    print(f"Sweep: {len(configs)} configs, prompt='{args.prompt}'")
    print(f"{'='*60}\n")

    summary = []
    for i, cfg in enumerate(configs):
        tag = cfg.pop("tag", f"run_{i:02d}")
        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- [{i+1}/{len(configs)}] {tag} ---")
        engine.set_thresholds(**cfg)

        try:
            result = engine.run(prompt=args.prompt, out_dir=out_dir, tag=tag)
            result["tag"] = tag
            result["ok"] = True
        except Exception as e:
            print(f"[ERROR] {tag}: {e}", flush=True)
            result = {"tag": tag, "ok": False, "error": str(e), **cfg}

        summary.append(result)
        with open(os.path.join(args.out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    _print_table(summary, args.out_root)


def _print_table(summary, out_root):
    lines = ["tag | unique | avg | max | nonzero | elapsed | ok"]
    lines.append("--- | --- | --- | --- | --- | --- | ---")
    for e in summary:
        lines.append("{} | {} | {} | {} | {} | {}s | {}".format(
            e.get("tag", "?"),
            e.get("num_unique_objs", "-"),
            e.get("avg_per_frame", "-"),
            e.get("max_per_frame", "-"),
            e.get("nonzero_frames", "-"),
            e.get("elapsed_sec", "-"),
            e.get("ok", False),
        ))
    table = "\n".join(lines)
    print(f"\n{'='*60}\n{table}\n{'='*60}")
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
