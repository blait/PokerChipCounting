# Poker Chip Counting — SAM3 Video Pipeline

대각선 앵글 포커 영상에서 칩 뭉치(stack)를 자동으로 인식하고 추적하는 파이프라인.

**SAM3** (Segment Anything Model 3) 비디오 프레딕터를 활용하여:
1. 텍스트 프롬프트(`"stack of poker chips"`)로 칩 뭉치 감지
2. 프레임 간 tracklet 유지 (같은 뭉치 = 같은 ID)
3. 마스크 오버레이 + 바운딩 박스 시각화
4. 누적 카운팅 (총 unique 뭉치 수)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Local Mac (편집/계획/SSH)                        │
│   ├─ sam3_engine.py, webui_server.py 편집        │
│   └─ 브라우저로 http://<gpu-ip>:8000 접속        │
└─────────────────┬───────────────────────────────┘
                  │ SSH / HTTP
┌─────────────────▼───────────────────────────────┐
│ AWS GPU Instance (g5.8xlarge, A10G 24GB)        │
│   ├─ SAM3 모델 (editable install)               │
│   ├─ sam3_engine.py — 모델 1회 로드, 반복 실행    │
│   ├─ webui_server.py — FastAPI + WebSocket       │
│   └─ /tmp/static/index.html — 브라우저 UI        │
└─────────────────────────────────────────────────┘
```

---

## Quick Start (이미 인스턴스가 있을 때)

```bash
# 1. SSH 접속
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@<PUBLIC_IP>

# 2. Web UI 시작
source /opt/pytorch/bin/activate
cd /tmp
cp /home/ubuntu/sam3_engine.py /home/ubuntu/webui_server.py /tmp/
mkdir -p /tmp/static && cp /home/ubuntu/static/index.html /tmp/static/
PYTORCH_ALLOC_CONF=expandable_segments:True \
  nohup python3 webui_server.py --video /home/ubuntu/dealer_540p.mp4 --port 8000 \
  > /home/ubuntu/webui.log 2>&1 &

# 3. 브라우저에서 접속
open http://<PUBLIC_IP>:8000
```

---

## 처음부터 구축하기 (Full Setup Guide)

### 1단계: AWS 인스턴스 생성

| 항목 | 값 |
|---|---|
| Instance Type | `g5.8xlarge` (A10G 24GB, vCPU 32, RAM 128GB) |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) |
| Storage | 100GB gp3 |
| Key Pair | 새로 생성 또는 기존 `.pem` 파일 |
| Security Group | SSH(22) + Custom TCP(8000) 열기 |

```bash
# 키페어 생성
aws ec2 create-key-pair --key-name sam3-chipcounting \
  --query 'KeyMaterial' --output text > ~/.ssh/sam3-chipcounting.pem
chmod 600 ~/.ssh/sam3-chipcounting.pem

# 인스턴스 시작 (AMI ID는 리전마다 다름 — 콘솔에서 확인)
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type g5.8xlarge \
  --key-name sam3-chipcounting \
  --security-group-ids sg-xxxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --region us-east-1

# Security Group에 포트 추가
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" --region us-east-1
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
  --protocol tcp --port 8000 --cidr "0.0.0.0/0" --region us-east-1
```

### 2단계: 초기 환경 설정

```bash
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@<PUBLIC_IP>

# 커널 자동 업데이트 차단 (NVIDIA 드라이버 깨짐 방지!!!)
sudo systemctl disable unattended-upgrades
sudo systemctl stop unattended-upgrades
sudo apt-mark hold linux-image-* linux-headers-* linux-modules-*

# GPU 확인
nvidia-smi
```

### 3단계: SAM3 설치

```bash
# venv 활성화 (AMI에 포함됨)
source /opt/pytorch/bin/activate

# SAM3 클론 및 설치
cd /home/ubuntu
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"
pip install einops ninja

# HuggingFace 모델 다운로드 (승인 필요: https://huggingface.co/facebook/sam3)
pip install huggingface_hub
huggingface-cli login --token <YOUR_HF_TOKEN>
huggingface-cli download facebook/sam3 --local-dir /home/ubuntu/sam3-checkpoints
```

### 4단계: Flash Attention (선택사항)

> ⚠️ 빌드에 1시간+ 소요. SAM3는 Flash Attention 없이도 동작함 (PyTorch scaled_dot_product_attention fallback).

```bash
# 반드시 nohup + 메모리 제한으로 실행 (아니면 OOM으로 SSH 죽음!)
nohup bash -c "MAX_JOBS=4 NVCC_THREADS=2 pip install flash-attn --no-build-isolation" \
  > /home/ubuntu/flash-attn-build.log 2>&1 &
# 진행 확인: tail -f /home/ubuntu/flash-attn-build.log
```

### 5단계: 비디오 준비

```bash
# 로컬에서 비디오 업로드
scp -i ~/.ssh/sam3-chipcounting.pem your_video.mp4 ubuntu@<IP>:/home/ubuntu/

# 4K 영상은 반드시 다운스케일! (A10G 24GB에서 4K = OOM)
ffmpeg -i dealer.mp4 -vf scale=960:540 dealer_540p.mp4
ffmpeg -i dealer.mp4 -vf scale=1280:720 dealer_720p.mp4
```

### 6단계: 엔진 파일 배포

```bash
# 로컬에서 GPU 서버로 복사
scp -i ~/.ssh/sam3-chipcounting.pem sam3_engine.py webui_server.py ubuntu@<IP>:/home/ubuntu/
scp -i ~/.ssh/sam3-chipcounting.pem static/index.html ubuntu@<IP>:/home/ubuntu/static/

# 서버에서 FastAPI 의존성 설치
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@<IP>
source /opt/pytorch/bin/activate
pip install fastapi "uvicorn[standard]" websockets
```

### 7단계: Web UI 시작

```bash
# /tmp/ 에서 실행해야 함 (namespace shadow 버그 방지)
cd /tmp
cp /home/ubuntu/sam3_engine.py /home/ubuntu/webui_server.py /tmp/
mkdir -p /tmp/static && cp /home/ubuntu/static/index.html /tmp/static/

PYTORCH_ALLOC_CONF=expandable_segments:True \
  nohup python3 webui_server.py --video /home/ubuntu/dealer_540p.mp4 --port 8000 \
  > /home/ubuntu/webui.log 2>&1 &

# 확인
tail -f /home/ubuntu/webui.log
# "[engine] predictor ready" + "Uvicorn running on http://0.0.0.0:8000" 나오면 성공
```

브라우저에서 `http://<PUBLIC_IP>:8000` 접속.

---

## 파일 설명

| 파일 | 역할 |
|---|---|
| `sam3_engine.py` | SAM3 모델 래퍼. 모델 1회 로드, threshold 변경, 배치/스트리밍 실행 |
| `webui_server.py` | FastAPI 서버. REST API + WebSocket 실시간 프레임 스트리밍 |
| `static/index.html` | 브라우저 UI. 슬라이더, 실시간 프레임 뷰, 결과 비교 |
| `sweep.py` | 배치 sweep 스크립트. 여러 config를 한번에 실행 |
| `make_overlay.py` | 단독 실행용 오버레이 생성기 (CLI) |
| `interactive.ipynb` | Jupyter 노트북 (대화형 실험) |
| `SETTINGS_REFERENCE.md` | 모든 SAM3 threshold 설정값 레퍼런스 |

---

## SAM3 Engine 사용법

### Python API

```python
from sam3_engine import SAM3Engine

# 모델 로드 (1회, ~10초)
engine = SAM3Engine(video_path="/home/ubuntu/dealer_540p.mp4")

# Threshold 확인/변경
engine.get_thresholds()
engine.set_thresholds(score_threshold_detection=0.2, new_det_thresh=0.5)

# 배치 실행 (전체 propagation 후 결과 반환)
result = engine.run(prompt="stack of poker chips", out_dir="/tmp/run1")
# result: {num_unique_objs, avg_per_frame, max_per_frame, elapsed_sec, ...}

# 스트리밍 실행 (프레임마다 yield)
for frame_data in engine.run_streaming(prompt="stack of poker chips"):
    if frame_data["frame_index"] == -1:
        final_result = frame_data["result"]
    else:
        jpeg_bytes = frame_data["jpeg_bytes"]  # 실시간 오버레이 JPEG

# 비디오 변경 (모델 재로드 없음)
engine.set_video("/home/ubuntu/other_video_540p.mp4")
```

### 배치 Sweep

```bash
cd /tmp
python3 sweep.py --video /home/ubuntu/dealer_540p.mp4 --out-root /home/ubuntu/sam3-runs/sweep_v4
```

`sweep.py`의 `CONFIGS` 리스트를 수정하여 테스트할 설정 조합을 정의.

### Web UI

슬라이더로 threshold 조절 → Run 클릭 → 실시간 프레임 스트리밍 확인 → 결과 비교.

---

## 주요 설정값 (Thresholds)

| 설정 | 기본값 | 역할 |
|---|---|---|
| `score_threshold_detection` | 0.3 | 감지 confidence 최소값. 낮추면 더 많이 잡지만 노이즈 증가 |
| `new_det_thresh` | 0.7 | 새 tracklet 생성 문턱. 낮추면 새 ID 쉽게 생성 (over-count 위험) |
| `det_nms_thresh` | 0.1 | NMS IoU 문턱. 올리면 겹친 박스 둘 다 유지 |
| `assoc_iou_thresh` | 0.1 | 이전 프레임과 매칭 IoU. 올리면 매칭 엄격 |
| `max_trk_keep_alive` | 30 | 미감지 tracklet 유지 프레임 수 |
| `suppress_overlapping_*` | 0.7 | 겹침 기반 억제 문턱 |

자세한 설명: [`SETTINGS_REFERENCE.md`](SETTINGS_REFERENCE.md)

---

## Troubleshooting (삽질 기록)

### 1. NVIDIA 드라이버 깨짐 — `nvidia-smi` 실패

**원인:** 커널이 자동 업데이트되면서 DKMS 모듈과 불일치.

**해결:**
```bash
sudo apt-get install -y linux-headers-$(uname -r)
sudo dkms install nvidia/580.126.16 -k $(uname -r)
sudo modprobe nvidia
nvidia-smi  # 확인
```

**예방:** `unattended-upgrades` 비활성화 + `apt-mark hold linux-*`.

### 2. Flash Attention 빌드 → SSH 죽음

**원인:** 기본 빌드 설정이 `MAX_JOBS × NVCC_THREADS × 5GB` 메모리 소비 → 시스템 RAM 고갈 → OOM killer가 sshd 종료.

**해결:** `MAX_JOBS=4 NVCC_THREADS=2` + `nohup` 백그라운드 실행.

**참고:** Flash Attention 없이도 SAM3 동작함 (성능 약간 저하).

### 3. 4K 비디오 OOM

**원인:** 3840×2160 프레임 + SAM3 내부 1008px 리사이즈 + tracklet memory bank = A10G 24GB 초과.

**해결:** `ffmpeg`로 540p 또는 720p로 다운스케일.

| 해상도 | 안정성 |
|---|---|
| 4K (3840×2160) | OOM 확정 |
| 720p (1280×720) | 높은 threshold만 가능 |
| 540p (960×540) | 안정 (추천) |
| 360p (640×360) | 가장 안전, 작은 뭉치 놓침 |

### 4. Tracklet 누적 OOM (긴 영상)

**원인:** `new_det_thresh` 낮추면 tracklet이 계속 새로 생성 → 각 tracklet이 GPU memory bank 점유 → monotone 증가 → OOM.

**해결:**
- `new_det_thresh` ≥ 0.5 유지
- `PYTORCH_ALLOC_CONF=expandable_segments:True` 항상 설정
- 519프레임(~20초) + threshold 0.3/0.7 조합이 A10G에서 안정적

### 5. `sam3.__file__ = None` — namespace package shadow

**원인:** `/home/ubuntu/`에서 실행하면 `sam3/` 디렉토리가 패키지로 인식되면서 설치된 sam3와 충돌.

**해결:** 항상 `/tmp/`에서 실행.

```bash
cp sam3_engine.py webui_server.py /tmp/
cd /tmp && python3 webui_server.py ...
```

### 6. SSH 접속 안 됨

**원인 1:** IP 변경됨 (집 공유기 재시작 등).
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" --region us-east-1
```

**원인 2:** OOM으로 sshd 죽음.
```bash
aws ec2 reboot-instances --instance-ids i-xxxxxxxxx --region us-east-1
# 2-3분 대기 후 재접속
```

**원인 3:** stop/start 후 IP 변경됨.
```bash
aws ec2 describe-instances --instance-ids i-xxxxxxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### 7. `BFloat16 and Float` dtype 에러

**원인:** `torch.autocast("cuda", dtype=torch.bfloat16)` 컨텍스트가 thread-local이라 새 스레드에서 적용 안 됨.

**해결:** `run()` / `run_streaming()` 메서드 시작 시 autocast 재진입:
```python
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
```

### 8. Detections = 0 (아무것도 감지 안 됨)

**원인:** `score_threshold_detection` 기본값(0.5)이 너무 높음.

**해결:** 0.3 또는 0.2로 낮추기. 프롬프트도 영향:
- `"stack of poker chips"` → 뭉치 단위 (추천)
- `"poker chip"` → 개별 칩 단위 (over-count)

### 9. 비디오 오버레이에 아무것도 안 그려짐

**원인:** SAM3 출력 dict 키가 `masks`가 아니라 `out_binary_masks`.

**올바른 키:**
```python
outputs.get("out_obj_ids")
outputs.get("out_binary_masks")
outputs.get("out_probs")
outputs.get("out_boxes_xywh")
```

---

## 비용 참고

| 인스턴스 | GPU | 시간당 | 메모 |
|---|---|---|---|
| g5.8xlarge | A10G 24GB | ~$2.45 | 기본 선택 |
| g6e.2xlarge | L40S 48GB | ~$2.52 | 메모리 2배, 같은 가격대 |
| g6e.12xlarge | L40S×4 192GB | ~$10.49 | 4K 원본 처리 가능 |

**사용하지 않을 때 반드시 stop!** (terminate 아님)

```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxxx --region us-east-1
# 재시작
aws ec2 start-instances --instance-ids i-xxxxxxxxx --region us-east-1
# IP 변경됨 — 확인 필요
```

---

## CountGD — 개별 칩 카운팅 (별도 서버)

SAM3가 **스택 단위**로 감지하는 반면, CountGD는 각 스택 내 **개별 칩 수**를 센다.
두 서버는 완전 독립적으로 운영되므로, 기존 SAM3 인프라에 영향 없음.

### Architecture

```
[SAM3 Server :8000]         [CountGD Server :8001]
  스택 감지/추적              개별 칩 카운팅
  (기존 EC2 유지)            (별도 EC2 인스턴스)
         │                          │
         └── SAM3 결과(bbox) ──────►┘  (또는 독립 사용)
```

### CountGD 서버 설치

```bash
# 별도 GPU 인스턴스에서:
chmod +x setup_countgd.sh
./setup_countgd.sh
```

체크포인트 수동 다운로드가 필요할 수 있음:
- [CountGD checkpoints (Google Drive)](https://drive.google.com/drive/folders/1zXXhm3XH5S2VD2_LgQ8UJfPHnJlm0YhE)

### CountGD 서버 실행

```bash
cd /tmp
cp ~/chipcounting/countgd_engine.py ~/chipcounting/countgd_server.py /tmp/
mkdir -p /tmp/countgd_static
cp ~/chipcounting/static/countgd.html /tmp/countgd_static/
python3 countgd_server.py \
    --countgd-repo /home/ubuntu/CountGD \
    --checkpoint /home/ubuntu/CountGD/checkpoints/checkpoint_fsc147_best.pth \
    --port 8001
```

### API

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/api/count` | POST (multipart) | 단일 이미지 칩 카운팅 |
| `/api/count-stacks` | POST (JSON) | SAM3 bbox 결과로 스택별 카운팅 |
| `/ws/count-video` | WebSocket | 실시간 프레임 카운팅 |
| `/health` | GET | 헬스체크 |

**Single image:**
```bash
curl -X POST http://localhost:8001/api/count \
  -F "file=@chip_stack.jpg" \
  -F "prompt=poker chip" \
  -F "box_threshold=0.23"
```

**SAM3 stacks:**
```bash
curl -X POST http://localhost:8001/api/count-stacks \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "...", "stacks": [[100,50,300,400],[350,50,550,400]], "prompt": "poker chip"}'
```

### 파이프라인 연동 (SAM3 → CountGD)

SAM3 실행 결과의 bbox를 CountGD `/api/count-stacks`로 전달:

```python
import requests, base64, cv2

# SAM3 결과에서 스택 bbox 추출 후
frame = cv2.imread("frame.jpg")
_, buf = cv2.imencode('.jpg', frame)
b64 = base64.b64encode(buf).decode()

resp = requests.post("http://<countgd-server>:8001/api/count-stacks", json={
    "image_b64": b64,
    "stacks": [[100, 50, 300, 400], [350, 50, 550, 400]],
    "prompt": "poker chip",
    "box_threshold": 0.23,
})
print(resp.json())
# {"total_chips": 24, "stacks": [{"count": 12, ...}, {"count": 12, ...}]}
```

---

## 향후 계획

- [x] **CountGD** 통합: 감지된 각 뭉치(stack) 내 개별 칩 카운팅
- [ ] **색상 분류**: 칩 색상별 금액 매핑
- [ ] **겹친 스택 문제 개선**: `det_nms_thresh`, `image_size` 튜닝
- [ ] **실시간 처리**: 라이브 카메라 입력 지원
- [ ] **SAM3↔CountGD 자동 연동**: 파이프라인 오케스트레이터

---

## License

This project uses [SAM3](https://github.com/facebookresearch/sam3) by Meta AI (Apache 2.0 License).
CountGD by Visual Geometry Group, Oxford (NeurIPS 2024).
