# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Poker chip counting from diagonal-angle video. Pipeline: **SAM3** (stack tracking/segmentation) → per-stack crop → **CountGD** (individual chip counting) → color classification → chip totals. Full background, rationale, and related work in `SETUP.md`.

The chip-counting application code does not exist yet — the repo currently holds a SAM3 checkout, a local venv, and `SETUP.md`. Work happens on a remote AWS GPU instance; this machine is mostly for editing, planning, and SSH-ing.

## Where work actually runs

Real model execution is on a remote **g5.8xlarge** (A10G 24GB). Always activate the system PyTorch venv there — don't try to run SAM3 on the Mac.

```bash
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@54.86.183.205
source /opt/pytorch/bin/activate    # PyTorch 2.10 + CUDA 13.0 + flash-attn 2.8.3
```

The public IP in `SETUP.md` changes on stop/start — check before assuming. Instance ID `i-08bc229c35862d86b`, region `us-east-1`.

## Repository layout

- `sam3/` — upstream `facebookresearch/sam3` checkout (editable install target). Reference `sam3/examples/*.ipynb` (especially `sam3_video_predictor_example.ipynb` and `sam3.1_video_predictor_example.ipynb`) for the SAM3 video-mode API this project will drive.
- `sam3env/` — local Python venv. Useful for linting/imports; GPU work happens remotely.
- `SETUP.md` — canonical source for infra, env versions, instance recreation steps, and incident history. Update it when any of those change.

## Operational rules (read before acting)

These come from prior incidents — see `SETUP.md` "삽질 기록" for the full stories.

- **Never terminate the EC2 instance** without asking. If the NVIDIA driver appears broken, try recovery first (`sudo apt-get install -y linux-headers-$(uname -r)` → `sudo dkms install nvidia/<ver> -k $(uname -r)` → `sudo modprobe nvidia`). Terminate is the last resort, not the first.
- **Do not reboot / upgrade the kernel.** `unattended-upgrades` is disabled and `apt-mark hold linux-*` is set on purpose — a kernel bump desyncs the NVIDIA DKMS module and kills the instance. Don't undo these.
- **flash-attn rebuilds must use `MAX_JOBS=4 NVCC_THREADS=2` under `nohup`.** Default build settings OOM (needs ~`MAX_JOBS × NVCC_THREADS × 5GB`) and take down sshd. SAM3 runs without flash-attn via PyTorch's `scaled_dot_product_attention` fallback, so don't block on it.
- **If your SSH IP changes**, add it to security group `sg-0b5336cb7dcbf2fe3` (port 22) before debugging "connection refused" further.

## Current status

Model access to `facebook/sam3` on HuggingFace was pending approval as of `SETUP.md`; verify on the instance (`hf download facebook/sam3 ...`) before assuming checkpoints are local. CountGD is not yet installed. No chip-counting code exists yet.
