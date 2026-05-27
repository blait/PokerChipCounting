"""Aggressive SAM3 sweep on dealer — run on big-GPU instance (L40S 4x 184GB).

More permissive thresholds than the first sweep. Fresh tracklets allowed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time

MODEL_BUILDER = "/home/ubuntu/sam3/sam3/model_builder.py"
MAKE_OVERLAY = "/tmp/make_overlay.py"

CONFIGS = [
    dict(tag="a1_poker_s01_n03", prompt="poker chip",         score=0.1, new_det=0.3),
    dict(tag="a2_poker_s01_n05", prompt="poker chip",         score=0.1, new_det=0.5),
    dict(tag="a3_poker_s02_n03", prompt="poker chip",         score=0.2, new_det=0.3),
    dict(tag="a4_poker_s02_n04", prompt="poker chip",         score=0.2, new_det=0.4),
    dict(tag="a5_chips_s02_n05", prompt="poker chips",        score=0.2, new_det=0.5),
    dict(tag="a6_casino_s02_n05",prompt="casino chip",        score=0.2, new_det=0.5),
    dict(tag="a7_token_s02_n05", prompt="gambling token",     score=0.2, new_det=0.5),
    dict(tag="a8_poker_s005_n03",prompt="poker chip",         score=0.05,new_det=0.3),
]


def patch_thresholds(score, new_det):
    with open(MODEL_BUILDER) as f:
        src = f.read()
    src = re.sub(r"score_threshold_detection=[0-9.]+,", f"score_threshold_detection={score},", src)
    src = re.sub(r"new_det_thresh=[0-9.]+,", f"new_det_thresh={new_det},", src)
    with open(MODEL_BUILDER, "w") as f:
        f.write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    summary = []
    for cfg in CONFIGS:
        tag = cfg["tag"]
        out_dir = os.path.join(args.out_root, tag)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n===== {tag} prompt='{cfg['prompt']}' score={cfg['score']} new_det={cfg['new_det']} =====", flush=True)
        patch_thresholds(cfg["score"], cfg["new_det"])
        t0 = time.time()
        cmd = [
            "python3", MAKE_OVERLAY,
            "--video", args.video,
            "--out", os.path.join(out_dir, "overlay.mp4"),
            "--counts", os.path.join(out_dir, "counts.json"),
            "--prompt", cfg["prompt"],
        ]
        log_path = os.path.join(out_dir, "run.log")
        with open(log_path, "w") as lf:
            env = dict(os.environ)
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0
        ok = r.returncode == 0 and os.path.exists(os.path.join(out_dir, "counts.json"))
        entry = dict(cfg, elapsed_sec=int(elapsed), ok=ok)
        if ok:
            with open(os.path.join(out_dir, "counts.json")) as f:
                d = json.load(f)
            vals = list(d["per_frame_counts"].values())
            entry.update(
                num_frames=d["num_frames"],
                num_unique_objs=d["num_unique_objs"],
                avg_per_frame=round(sum(vals) / max(len(vals), 1), 2),
                max_per_frame=max(vals) if vals else 0,
                nonzero_frames=sum(1 for v in vals if v > 0),
            )
        summary.append(entry)
        with open(os.path.join(args.out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> ok={ok} elapsed={elapsed:.0f}s unique={entry.get('num_unique_objs','-')} avg={entry.get('avg_per_frame','-')}", flush=True)

    lines = ["tag | prompt | score | new_det | unique | avg | max | nonzero | elapsed | ok"]
    for e in summary:
        lines.append(
            "{tag} | {prompt} | {score} | {new_det} | {u} | {avg} | {m} | {nz} | {el}s | {ok}".format(
                tag=e["tag"], prompt=e["prompt"], score=e["score"], new_det=e["new_det"],
                u=e.get("num_unique_objs", "-"), avg=e.get("avg_per_frame", "-"),
                m=e.get("max_per_frame", "-"), nz=e.get("nonzero_frames", "-"),
                el=e["elapsed_sec"], ok=e["ok"],
            )
        )
    with open(os.path.join(args.out_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
