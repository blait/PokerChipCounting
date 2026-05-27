"""Sweep thresholds with prompt fixed to 'stack of poker chips'."""
import argparse, json, os, re, shutil, subprocess, time

MODEL_BUILDER = "/home/ubuntu/sam3/sam3/model_builder.py"
MAKE_OVERLAY = "/tmp/make_overlay.py"
PROMPT = "stack of poker chips"

CONFIGS = [
    dict(tag="s1_s005_n03", score=0.05, new_det=0.3),
    dict(tag="s2_s01_n03",  score=0.1,  new_det=0.3),
    dict(tag="s3_s01_n05",  score=0.1,  new_det=0.5),
    dict(tag="s4_s02_n03",  score=0.2,  new_det=0.3),
    dict(tag="s5_s02_n05",  score=0.2,  new_det=0.5),
    dict(tag="s6_s02_n07",  score=0.2,  new_det=0.7),
    dict(tag="s7_s03_n05",  score=0.3,  new_det=0.5),
    dict(tag="s8_s03_n07",  score=0.3,  new_det=0.7),
]


def patch(score, new_det):
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
        print(f"\n===== {tag} score={cfg['score']} new_det={cfg['new_det']} =====", flush=True)
        patch(cfg["score"], cfg["new_det"])
        t0 = time.time()
        cmd = ["python3", MAKE_OVERLAY,
               "--video", args.video,
               "--out", os.path.join(out_dir, "overlay.mp4"),
               "--counts", os.path.join(out_dir, "counts.json"),
               "--prompt", PROMPT]
        with open(os.path.join(out_dir, "run.log"), "w") as lf:
            env = dict(os.environ); env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0
        ok = r.returncode == 0 and os.path.exists(os.path.join(out_dir, "counts.json"))
        entry = dict(cfg, prompt=PROMPT, elapsed_sec=int(elapsed), ok=ok)
        if ok:
            with open(os.path.join(out_dir, "counts.json")) as f:
                d = json.load(f)
            vals = list(d["per_frame_counts"].values())
            entry.update(
                num_frames=d["num_frames"],
                num_unique_objs=d["num_unique_objs"],
                avg_per_frame=round(sum(vals)/max(len(vals),1), 2),
                max_per_frame=max(vals) if vals else 0,
                nonzero_frames=sum(1 for v in vals if v > 0),
            )
        summary.append(entry)
        with open(os.path.join(args.out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> ok={ok} elapsed={elapsed:.0f}s unique={entry.get('num_unique_objs','-')} avg={entry.get('avg_per_frame','-')}", flush=True)

    lines = ["tag | score | new_det | unique | avg | max | nonzero | elapsed | ok"]
    for e in summary:
        lines.append("{tag} | {score} | {new_det} | {u} | {avg} | {m} | {nz} | {el}s | {ok}".format(
            tag=e["tag"], score=e["score"], new_det=e["new_det"],
            u=e.get("num_unique_objs","-"), avg=e.get("avg_per_frame","-"),
            m=e.get("max_per_frame","-"), nz=e.get("nonzero_frames","-"),
            el=e["elapsed_sec"], ok=e["ok"]))
    with open(os.path.join(args.out_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
