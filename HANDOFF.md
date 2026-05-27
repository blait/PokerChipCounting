# HANDOFF — 세션 단절 대비 인수인계

**업데이트:** 2026-05-04. 세션이 끊겨도 다음 Claude Code 세션(또는 사람)이 이 문서만 보고 이어서 작업할 수 있게 작성.

---

## 1. 지금 상태 한 줄 요약

dealer 영상에 SAM3 video predictor를 돌려 감지 설정값을 스윕 중. 3번의 스윕(v1/v2/stack) 완료. 겹친 스택 하나만 잡히는 문제 남음. 다음 후보는 `image_size↑` / `det_nms_thresh↑` / `suppress_overlapping↑` 등 아직 안 만져본 knob.

---

## 2. 인프라 — 지금 살아있는 것들

| 이름 | ID | 타입 | IP | GPU | 시간당 | 상태 |
|---|---|---|---|---|---|---|
| sam3-chipcounting (g5) | `i-08bc229c35862d86b` | g5.8xlarge | 54.86.183.205 | A10G 24GB | $2.45 | running |
| sam3-g6e12xl (new) | `i-060c3e59bd75d6823` | g6e.12xlarge | 34.207.102.234 | 4× L40S 46GB | $10.49 | running |

**합쳐서 시간당 ~$13**. g5는 기록용/백업, 실제 실험은 g6e에서.

- Region: `us-east-1`
- Key pair: `sam3-chipcounting` (`~/.ssh/sam3-chipcounting.pem`)
- Security Group: `sg-0b5336cb7dcbf2fe3` — SSH 22 open only to whitelisted IPs
- AMI (g6e origin): `ami-06e4adfd722988d24` (g5에서 스냅샷)

**중요 운영 규칙 (기존 feedback):**
- **인스턴스 terminate 금지** — 사용자 확인 필수
- **커널/드라이버 건드리지 말 것** — `unattended-upgrades` disabled + `apt-mark hold linux-*`
- IP 바뀌면 SG에 current IP 추가: `aws ec2 authorize-security-group-ingress --group-id sg-0b5336cb7dcbf2fe3 --protocol tcp --port 22 --cidr "${MY_IP}/32" --region us-east-1`

---

## 3. SSH 접속

```bash
# g6e (현재 주력)
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@34.207.102.234

# g5 (백업)
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@54.86.183.205
```

Python venv 활성화: `source /opt/pytorch/bin/activate` (PyTorch 2.10+cu130, Python 3.13)

---

## 4. 리모트 파일 구조 (양쪽 동일)

```
/home/ubuntu/
  sam3/                          ← repo (editable install)
    └─ sam3/model_builder.py     ← ★ threshold 편집 대상
  sam3-checkpoints/              ← HF 가중치 (6.5GB)
    ├─ sam3.pt
    └─ model.safetensors
  dealer.mp4                     ← 원본 4K (g6e에만, scp로 올림)
  dealer_540p.mp4                ← 960×540 스윕용
  dealer_360p.mp4                ← 640×360 (g5에만, OOM 테스트 흔적)
  pokervideo_720p.mp4            ← 1280×720
  sam3-runs/                     ← 모든 결과
```

`/tmp/`에는 실행 스크립트가 있어야 함 (없으면 재업로드):
- `/tmp/make_overlay.py`
- `/tmp/sweep_configs.py` (v1)
- `/tmp/sweep_aggressive.py` (v2)
- `/tmp/sweep_stackprompt.py` (stack)

**주의:** 인스턴스 재부팅 시 `/tmp` 날아감. 로컬에서 `/Users/hyeonsup/chipcounting/*.py`를 `scp`로 다시 올리면 됨. 또한 **스크립트는 `/home/ubuntu/`에 두면 안 됨** (sam3 namespace package shadow 버그, 아래 §8 참고).

---

## 5. 핵심 흐름 — "설정 바꿔 돌려" 한 사이클

### (a) threshold 편집 (리모트 파일 직접 수정)

```bash
ssh ubuntu@<ip> 'sed -i -E \
  "s/score_threshold_detection=[0-9.]+,/score_threshold_detection=0.3,/g; \
   s/new_det_thresh=[0-9.]+,/new_det_thresh=0.7,/g" \
  /home/ubuntu/sam3/sam3/model_builder.py'
```

- 두 번 치환되는 이유: `model_builder.py`에 같은 block이 `apply_temporal_disambiguation=True/False` 각각 하나씩 (현재는 True 경로만 실제 사용).
- `pip install -e`라 파일 수정이 다음 python 프로세스에 즉시 반영.

### (b) 실행 (nohup 백그라운드)

```bash
ssh ubuntu@<ip> '
  cd /tmp && source /opt/pytorch/bin/activate &&
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  nohup python3 /tmp/make_overlay.py \
    --video /home/ubuntu/dealer_540p.mp4 \
    --out /home/ubuntu/sam3-runs/myrun/overlay.mp4 \
    --counts /home/ubuntu/sam3-runs/myrun/counts.json \
    --prompt "poker chip" \
    > /home/ubuntu/sam3-runs/myrun.log 2>&1 &
  echo PID=$!
'
```

- `cd /tmp` 필수 — 그래야 `sam3` 파이썬 패키지가 repo 루트(`/home/ubuntu/sam3`)를 namespace package로 잘못 잡지 않음.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` 는 OOM 완화. 항상 켜는 걸 추천.

### (c) 모니터링

```bash
# 진행률
ssh ubuntu@<ip> 'grep -oE "[0-9]+/[0-9]+" /home/ubuntu/sam3-runs/myrun.log | tail -1'

# 프로세스 살아있는지
ssh ubuntu@<ip> 'pgrep -af make_overlay | head -1'

# GPU 사용량
ssh ubuntu@<ip> 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

### (d) 결과 회수

```bash
scp -i ~/.ssh/sam3-chipcounting.pem -r \
    ubuntu@<ip>:/home/ubuntu/sam3-runs/myrun \
    /Users/hyeonsup/chipcounting/sam3-runs/
```

---

## 6. Sweep — 리모트 python이 threshold patch + exec를 루프

`sweep_*.py` 중 하나를 선택해 실행:

```bash
ssh ubuntu@34.207.102.234 '
  cd /tmp && source /opt/pytorch/bin/activate &&
  nohup python3 /tmp/sweep_stackprompt.py \
    --video /home/ubuntu/dealer_540p.mp4 \
    --out-root /home/ubuntu/sam3-runs/my_sweep \
    > /home/ubuntu/sam3-runs/my_sweep.log 2>&1 &
  echo PID=$!
'
```

스크립트 내부:
1. `CONFIGS` 리스트의 각 config에 대해
2. `patch_thresholds(score, new_det)` — regex로 `model_builder.py` 수정
3. `subprocess.run([python3, make_overlay.py, --prompt=..., ...])`
4. 결과 파싱해 `summary.json` / `summary.txt` 실시간 업데이트

**sweep 3종의 차이:**
| 스크립트 | prompt | threshold 조합 |
|---|---|---|
| `sweep_configs.py` | 다양한 프롬프트 | score 0.2~0.3, new_det 0.5~0.7 (보수적) |
| `sweep_aggressive.py` | 다양한 프롬프트 | score 0.05~0.2, new_det 0.3~0.5 (공격적) |
| `sweep_stackprompt.py` | `stack of poker chips` 고정 | score 0.05~0.3, new_det 0.3~0.7 전방위 |

새 sweep 하려면 이 중 하나 복사해서 `CONFIGS` 리스트만 바꾸면 됨.

---

## 7. 결과 로컬 위치

```
/Users/hyeonsup/chipcounting/sam3-runs/
├─ run3/overlay_cum.mp4              ← pokervideo thr=0.2 (unique 20)
├─ dealer/overlay_cum.mp4            ← dealer thr=0.5 (unique 13)
├─ dealer_thr03/overlay.mp4          ← dealer thr=0.3 (unique 14)
├─ dealer_sweep/                     ← Sweep v1 (6개)
├─ dealer_sweep_v2/                  ← Sweep v2 공격적 (8개)
└─ dealer_sweep_stack/               ← Sweep stack (8개, 최신)
```

각 sweep 폴더에 `summary.txt`, `summary.json`, `<tag>/overlay.mp4`, `<tag>/counts.json`.

---

## 8. 과거 삽질 요약 — 반복하지 말 것

### 8.1 Namespace package shadow
**증상:** `sam3.__file__ = None`, `bpe_simple_vocab_16e6.txt.gz` not found.
**원인:** CWD가 `/home/ubuntu`면 거기의 `sam3/` (repo 루트, `__init__.py` 없음) 가 namespace package로 잡혀 실제 `/home/ubuntu/sam3/sam3/` 패키지를 가림.
**해결:** 스크립트를 `/tmp/`에서 실행 (`cd /tmp && python3 /tmp/...`).

### 8.2 4K OOM / 긴 영상 OOM
**증상:** `torch.OutOfMemoryError` 중간에 죽음.
**원인:** A10G 24GB에서 1920+ 해상도 / tracklet 많은 긴 영상.
**해결:**
- 해상도 다운스케일: `ffmpeg -i in.mp4 -vf scale=960:-2 -c:v libx264 -preset fast -crf 22 out.mp4`
- g6e.12xlarge로 마이그레이션 (L40S 46GB×4)
- `PYTORCH_ALLOC_CONF=expandable_segments:True`
- `new_det_thresh` 너무 낮추지 말기 (tracklet 수 폭증 = 메모리 누적)

### 8.3 `detections = 0` (frame 0에 아무것도 안 잡힘)
**원인:** `build_sam3_video_model` 내부 `score_threshold_detection=0.5` 하드코딩. 이미지 predictor와 달리 kwargs로 못 넘김.
**해결:** `sed`로 숫자 직접 수정 (§5-a).

### 8.4 Overlay에 박스 안 그려짐 (원본과 동일)
**원인:** video predictor output 키는 `out_binary_masks`, `out_boxes_xywh`, `out_probs`, `out_obj_ids`. 내 코드가 `masks`/`obj_ids`로 읽고 있었음.
**해결:** `make_overlay.py`에서 `out["out_binary_masks"]` 등 올바른 키 사용.

### 8.5 `mat1 and mat2 must have the same dtype` (bfloat16 vs float)
**원인:** SAM3 image predictor는 autocast 필요.
**해결:** 스크립트 상단에:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
```

### 8.6 g6e NVIDIA 드라이버 실패
**증상:** AMI로 g6e 부팅 직후 `nvidia-smi` 실패 — DKMS는 `6.17.0-1009/1010` 용인데 부팅된 커널은 `1012-aws`.
**해결 (SETUP.md 절차):**
```bash
sudo apt-get install -y linux-headers-$(uname -r)
sudo dkms install nvidia/580.126.16 -k $(uname -r)
sudo modprobe nvidia
```
실행 중 SSH 끊기지만 재연결하면 복구되어 있음.

### 8.7 SSH 접속 실패 (IP 변경)
로컬 IP 바뀌면 SG에 추가:
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-0b5336cb7dcbf2fe3 \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" --region us-east-1
```

---

## 9. 현재 실험 결과 요약 (dealer_540p.mp4 기준)

**v1 (보수적):** unique 7~17, c2가 최고 (poker chip / score 0.3 / new_det 0.5 → unique 17)
**v2 (공격적):** unique 8~52, over-count 경향. a3(score 0.2 / new_det 0.3) = 52
**stack (프롬프트 고정):** unique 7~18, s1(stack of poker chips / score 0.05 / new_det 0.3) = 18

**남은 문제:** dealer 영상 프레임 24~40 구간에서 두 스택이 겹쳐있는데 한 쪽만 감지됨. threshold 조정만으로는 해결 안 됨.

**다음 실험 후보 (아직 안 건드린 파라미터들):**
- `det_nms_thresh` 0.1 → 0.3~0.5 (겹친 박스 허용)
- `image_size` 1008 → 1280 또는 1536 (세부 구분)
- `suppress_overlapping_based_on_recent_occlusion_threshold` 0.7 → 0.9
- `assoc_iou_thresh` 0.1 → 0.05/0.3

---

## 10. 현재 `model_builder.py` 상태 (둘 다)

```python
# sam3/sam3/model_builder.py:746 (apply_temporal_disambiguation=True 경로)
score_threshold_detection=0.3,
assoc_iou_thresh=0.1,
det_nms_thresh=0.1,
new_det_thresh=0.7,
hotstart_delay=15,
# ... 나머지 기본값
```

최근 sweep_stackprompt.py 의 마지막 iteration(s8) 값인 score=0.3, new_det=0.7이 그대로 남아있음. 새 실험 시작 전에 원하는 값으로 치환 필요.

---

## 11. 로컬 파일 인덱스

```
/Users/hyeonsup/chipcounting/
├─ CLAUDE.md                     ← 프로젝트 개요 (초기 세션용)
├─ SETUP.md                      ← 인프라 세팅 매뉴얼
├─ JOURNAL.md                    ← 지금까지 진행 기록 (14섹션)
├─ SETTINGS_REFERENCE.md         ← threshold/kwargs 전체 설명
├─ HANDOFF.md                    ← (이 파일)
├─ make_overlay.py               ← 영상→overlay.mp4 생성 (누적 카운터 포함)
├─ run_sam3_video.py             ← 초기 버전 (old, make_overlay.py가 상위호환)
├─ sweep_configs.py              ← v1 스윕
├─ sweep_aggressive.py           ← v2 공격적 스윕
├─ sweep_stackprompt.py          ← stack 프롬프트 스윕
├─ inspect_state.py              ← image predictor state 덤프 유틸
├─ prompt_sweep.py               ← image 모드 프롬프트 테스트 유틸
├─ pokervideo.mp4                ← 원본 샘플 (5.77초, 3840x2160)
├─ pokervideo/dealer.mp4         ← dealer 원본 (20.8초, 3840x2160)
├─ sam3/ sam3env/                ← repo 체크아웃 + venv (로컬 IDE용)
└─ sam3-runs/                    ← 실험 결과 복사본
```

---

## 12. 당장 할 일 (다음 세션이 이어받을 수 있게)

1. **겹친 스택 분리 실험** — 아직 안 만져본 knob 중 `det_nms_thresh` 및 `image_size`부터. 각 값을 `model_builder.py`에서 수정 후 `make_overlay.py` 재실행 필요. (sweep 스크립트 확장: `sweep_nms.py` 같은 이름으로 `det_nms_thresh` 치환하는 버전 만들기)
2. **g5 stop 결정** — 더 이상 안 쓰면 `aws ec2 stop-instances --instance-ids i-08bc229c35862d86b --region us-east-1`로 과금 중단. 사용자 확인 필요.
3. **CountGD 설치** — 다음 파이프라인 단계. conda env (python 3.9.19) + gcc-11 + GroundingDINO CUDA ops 빌드 + Google Drive 체크포인트. SETUP.md § "CountGD" 참고 예정.
4. **`/tmp` 스크립트 복구** — 세션 재개 시 자동으로 안 되니 항상 먼저 확인: `ssh ubuntu@34.207.102.234 'ls /tmp/*.py'`. 없으면 로컬에서 `scp *.py ubuntu@34:/tmp/`.

---

## 13. 긴급 체크리스트 (세션 재개 시)

```bash
# 1. 인스턴스 살아있나
aws ec2 describe-instances --instance-ids i-060c3e59bd75d6823 i-08bc229c35862d86b \
  --region us-east-1 --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text

# 2. IP 변경됐나 (.ssh SG 확인)
MY_IP=$(curl -s https://checkip.amazonaws.com); echo "$MY_IP"
aws ec2 describe-security-groups --group-ids sg-0b5336cb7dcbf2fe3 --region us-east-1 \
  --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text

# 3. SSH 접속 테스트
ssh -i ~/.ssh/sam3-chipcounting.pem -o ConnectTimeout=10 ubuntu@34.207.102.234 \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'

# 4. 스크립트 확인 + (필요시) 재업로드
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@34.207.102.234 'ls /tmp/*.py' || \
  scp -i ~/.ssh/sam3-chipcounting.pem \
    /Users/hyeonsup/chipcounting/{make_overlay,sweep_configs,sweep_aggressive,sweep_stackprompt}.py \
    ubuntu@34.207.102.234:/tmp/

# 5. 현재 threshold 확인
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@34.207.102.234 \
  'grep -n "score_threshold_detection\|new_det_thresh" /home/ubuntu/sam3/sam3/model_builder.py | head -4'
```
