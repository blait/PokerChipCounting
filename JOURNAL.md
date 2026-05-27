# Poker Chip Counting — 프로젝트 진행 기록

SAM3 기반 포커 칩 뭉치 감지/트래킹 — 문제 정의, 인프라 구축, 삽질, 실험 요약.

---

## 1. 초기 목표와 접근

**목표:** 대각 앵글 포커 영상에서 칩 뭉치를 트래킹하고, 뭉치 내부 개별 칩 수/종류를 카운팅.

**파이프라인 구상 (SETUP.md 기준):**
```
영상 → SAM3 (뭉치 세그멘테이션 + 트래킹)
     → 뭉치 크롭
     → CountGD (개별 칩 카운팅)
     → 색상 분류 → 금액 매핑
```

**왜 이 조합:**
- SAM3: 뭉치 단위 트래킹 강점, 개별 칩 카운팅은 약함
- CountGD: 밀집 객체 카운팅 SOTA, zero-shot 텍스트 프롬프트
- 학습 데이터 없이 동작 가능

---

## 2. 인프라 세팅 — g5.8xlarge

| 항목 | 값 |
|---|---|
| Instance | `i-08bc229c35862d86b` g5.8xlarge |
| GPU | NVIDIA A10G **24GB** |
| Region | us-east-1 |
| AMI | Deep Learning OSS PyTorch 2.10 (Ubuntu 24.04) |
| Python | 3.13 (venv `/opt/pytorch`) |
| CUDA | 13.0, PyTorch 2.10.0+cu130 |
| Flash-Attn | 2.8.3 |
| 시간당 | $2.45 |

### 설치 삽질 (SETUP.md 기준)

**Flash-attn 빌드 OOM:**
- 기본 빌드는 `MAX_JOBS × NVCC_THREADS × 5GB` 메모리 필요
- g5.2xlarge (32GB)에서 OOM → sshd 죽고 SSH 불가
- **해결:** g5.8xlarge (128GB RAM) + `MAX_JOBS=4 NVCC_THREADS=2` + nohup (1시간 소요)

**NVIDIA 드라이버 깨짐:**
- `unattended-upgrades`가 커널 자동 업데이트 → DKMS 모듈 불일치
- **근본 해결:** `unattended-upgrades` 비활성화 + `apt-mark hold linux-*`
- **사후 복구:** `sudo apt-get install linux-headers-$(uname -r)` → `sudo dkms install nvidia/580.126.16 -k $(uname -r)` → `sudo modprobe nvidia`

---

## 3. SAM3 모델 + 체크포인트

### HuggingFace
- `facebook/sam3` 모델 접근 승인 (계정: blait8595)
- 체크포인트 다운로드: `hf download facebook/sam3 --local-dir /home/ubuntu/sam3-checkpoints`
- 6.5GB (sam3.pt 3.4GB + model.safetensors 3.4GB + tokenizer/config)

### SAM3 패키지
- `pip install -e ".[notebooks]"` (editable install)
- 버전 0.1.0

---

## 4. 첫 실행 — 버그와 해결

### 4.1. Namespace package shadow
**증상:** `sam3.__file__ = None`, `pkg_resources.resource_filename`이 bpe 경로 못 찾음.

**원인:** 스크립트를 `/home/ubuntu/run_sam3_video.py`로 실행 → CWD가 `/home/ubuntu` → 거기 있는 `sam3/` (repo 루트, `__init__.py` 없음)이 namespace package로 인식되어 실제 패키지 가림.

**해결:** 스크립트를 `/tmp/`로 옮기고 `cd /tmp && python3 /tmp/run_sam3_video.py`로 실행.

### 4.2. 4K 영상 OOM
**증상:** 비디오 propagate 중 OOM (`score_threshold_detection=0.5` 기본)

**원인:** pokervideo.mp4 = 3840×2160. A10G 24GB 부족.

**해결:** ffmpeg로 1280×720 다운스케일.

### 4.3. `add_prompt`에서 detections = 0
**증상:** `"poker chip stack"` 프롬프트 넣어도 `len(outputs['masks'])==0`.

**원인:** 비디오 predictor 내부 `score_threshold_detection=0.5` 하드코딩. 이미지 모델은 프레임 0에 대해 출력 내지만 필터링되어 사라짐.

**해결:** `sam3/model_builder.py`의 `build_sam3_video_model`에서 threshold 직접 수정:
- `score_threshold_detection=0.5 → 0.2`
- `new_det_thresh=0.7 → 0.3`
→ pokervideo unique **20개** 트래킹.

### 4.4. Overlay 박스 안 그려지는 버그
**증상:** `prepare_masks_for_visualization`는 제대로 그리는데, 내가 짠 overlay 루프는 원본과 동일한 영상.

**원인:** SAM3 video predictor의 raw output 키는 `out_binary_masks`, `out_boxes_xywh`, `out_probs`, `out_obj_ids`. 내 코드가 `masks`, `obj_ids`를 읽고 있었음.

**해결:** 키 이름 수정 + bbox/label 그리기. pokervideo: 프레임 173/173 감지 확인.

### 4.5. 이미지 predictor dtype 오류
**증상:** `RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`

**원인:** 노트북 예제가 스크립트 상단에 `torch.autocast("cuda", dtype=torch.bfloat16).__enter__()` 걸어놓음. 이걸 안 해서 dtype 불일치.

**해결:** 스크립트 상단에 autocast 추가.

### 4.6. 프레임 프롬프트 스윕 (이미지 모드)
**sam3-runs/sweep** — 이미지 predictor로 프레임 60 한 장 감지 테스트:

| threshold | `poker chip` | `poker chip stack` | `casino chip` | `pile of chips` |
|---|---|---|---|---|
| 0.1 | 65 | 22 | 38 | 17 |
| 0.3 | 11 | 10 | 10 | 5 |
| 0.5 | 9 | 8 | 9 | 0 |

`poker chip` + threshold 0.3이 가장 깔끔.

---

## 5. 누적 카운터 추가

**문제:** overlay에 프레임별 "now=n"만 표시, 영상 전체에서 총 몇 개 뭉치 봤는지 모름.

**해결:** `seen_ids` 집합 유지 + `first_seen_frame` 기록 → `frame {N}  now={now}  total unique={len(seen_ids)}`.

---

## 6. dealer 영상 — OOM과의 사투

**pokervideo:** 173프레임, 5.77초, A10G 24GB에서 무사히 완료.

**dealer:** 519프레임, 20.8초, 뭉치 많음 — **반복 OOM**.

| 시도 | 해상도 | 설정 | 결과 |
|---|---|---|---|
| 1 | 1280×720 | score=0.5, new_det=0.7 | 처음부터 OOM |
| 2 | 960×540 | score=0.2, new_det=0.3 | 423/519에서 OOM |
| 3 | 640×360 | score=0.2, new_det=0.3 + expandable_segments | 423/519에서 OOM |
| 4 | 960×540 | **score=0.5, new_det=0.7** | **완료** (unique 13) |

**교훈:** `new_det_thresh` 낮추면 tracklet 계속 새로 생김 → GPU 메모리 monotone 누적 → OOM. A10G 24GB에서 긴 영상(500+프레임) + 낮은 threshold = 실패.

---

## 7. Tracklet / 설정값 이해

**Tracklet = "한 오브젝트의 영상 속 인생"** — 같은 ID로 묶인 프레임별 마스크 체인.

**두 threshold 역할:**
- `score_threshold_detection` (기본 0.5): detector가 "이건 칩이다"라고 인정할 confidence 최소값. 이걸 넘어야 후보가 됨.
- `new_det_thresh` (기본 0.7): 후보 중 기존 tracklet과 매칭 안 되는 것에 대해 "새 ID 시작"할지 판정. 더 엄격한 문턱.

**tracklet 수 = GPU 메모리와 직결** (각 tracklet이 memory bank를 GPU에 유지).

---

## 8. Sweep v1 — 기본 (g5.8xlarge)

dealer_540p.mp4 × 6가지 설정:

| tag | prompt | score | new_det | unique | avg | max | 시간 |
|---|---|---|---|---|---|---|---|
| c1 | poker chip | 0.2 | 0.7 | 14 | 5.49 | 9 | 5.3m |
| **c2** | poker chip | 0.3 | 0.5 | **17** | 6.34 | 12 | 5.2m |
| c3 | poker chip stack | 0.3 | 0.7 | 11 | 3.35 | 10 | 3.7m |
| c4 | stack of poker chips | 0.3 | 0.7 | 9 | 3.29 | 8 | 3.7m |
| c5 | pile of poker chips | 0.3 | 0.7 | 7 | 2.29 | 7 | 3.3m |
| c6 | casino chip | 0.3 | 0.7 | 14 | 5.79 | 9 | 4.9m |

OOM 없이 전부 완료.

---

## 9. 인프라 마이그레이션 — g6e.12xlarge

**동기:** dealer 같은 영상에 공격적 threshold를 쓰려면 A10G 24GB로는 부족.

### 9.1. GPU 옵션 비교
| 타입 | GPU | 시간당 | 설명 |
|---|---|---|---|
| g5.8xlarge (기존) | A10G 24GB | $2.45 | baseline |
| g6e.2xlarge | L40S 48GB | $2.52 | 메모리 2배, CPU/RAM은 줄어듦 |
| g6e.12xlarge | **L40S×4 184GB** | **$10.49** | 선택 |
| p4d.24xlarge | A100×8 320GB | $32 | 과잉 |
| p5en.48xlarge | H200×8 | $64 | Capacity Block 필요 |

### 9.2. 마이그레이션 절차
1. **AMI 스냅샷:** `aws ec2 create-image --no-reboot ...` → `ami-06e4adfd722988d24`
2. **available 확인:** 약 15분 대기
3. **새 인스턴스 시작:** g6e.12xlarge, 동일 SG/키페어
4. **새 IP:** `34.207.102.234`

### 9.3. NVIDIA 드라이버 복구
**증상:** `nvidia-smi` 실패, DKMS는 1009/1010 커널용인데 부팅된 커널은 **1012-aws**.

**해결:** SETUP.md의 사후 복구 절차 그대로:
```bash
sudo apt-get install -y linux-headers-$(uname -r)
sudo dkms install nvidia/580.126.16 -k $(uname -r)
sudo modprobe nvidia
```
SSH 세션이 apt 수행 중에 끊어졌지만 자동 리부팅 후 **4× L40S 46GB 전부 인식**.

### 9.4. Security Group
- 작업 중 로컬 IP 여러 번 바뀜 (`15.248.5.81 → 15.248.5.82 → 218.55.253.116`)
- 매번 SG sg-0b5336cb7dcbf2fe3에 현재 IP 추가

---

## 10. Sweep v2 — 공격적 (g6e.12xlarge)

dealer_540p.mp4 × 8가지 공격적 설정:

| tag | prompt | score | new_det | unique | avg | max | 시간 |
|---|---|---|---|---|---|---|---|
| a1 | poker chip | 0.1 | 0.3 | 45 | 13.56 | 28 | 16m ⚠️ |
| a2 | poker chip | 0.1 | 0.5 | 18 | 7.9 | 12 | 3.1m |
| a3 | poker chip | 0.2 | 0.3 | **52** | 13.3 | 30 | 3.3m |
| a4 | poker chip | 0.2 | 0.4 | 27 | 9.42 | 21 | 2.7m |
| a5 | poker chips | 0.2 | 0.5 | 8 | 2.64 | 8 | 1.7m |
| a6 | casino chip | 0.2 | 0.5 | 19 | 6.65 | 13 | 2m |
| a7 | gambling token | 0.2 | 0.5 | 25 | 9.03 | 18 | 3.1m |
| a8 | poker chip | 0.05 | 0.3 | 48 | 15.5 | 32 | 3.7m |

**관찰:**
- 낮은 threshold (a1, a3, a8) → unique 45~52지만 **over-count** (하나의 뭉치가 여러 ID로 쪼개짐)
- L40S에서도 OOM 없이 전부 완료 — 메모리 2배가 효과
- a1이 16분 걸린 이유: tracklet 45개가 후반 프레임에서 누적되어 처리 속도 급락

---

## 11. Sweep "stack of poker chips" — 뭉치 단위 프롬프트

프롬프트를 `stack of poker chips`로 고정, threshold만 다양화:

| tag | score | new_det | unique | avg | max | 시간 |
|---|---|---|---|---|---|---|
| **s1** | 0.05 | 0.3 | **18** | 5.17 | 14 | 2.3m |
| s2 | 0.1 | 0.3 | 18 | 4.25 | 13 | 1.9m |
| s3 | 0.1 | 0.5 | 11 | 3.34 | 10 | 1.9m |
| s4 | 0.2 | 0.3 | 17 | 3.7 | 13 | 1.7m |
| s5 | 0.2 | 0.5 | 11 | 3.34 | 10 | 1.6m |
| s6 | 0.2 | 0.7 | 9 | 3.29 | 8 | 1.6m |
| s7 | 0.3 | 0.5 | 11 | 3.34 | 10 | 1.6m |
| s8 | 0.3 | 0.7 | 9 | 3.29 | 8 | 1.6m |

**관찰:**
- `stack of` 프롬프트는 뭉치 단위로 감지 → unique 수가 7~18로 절제됨 (v2의 8~52보다 훨씬 깔끔)
- over-count 감소

---

## 12. 남은 문제

1. **겹친 두 스택 구분 불가:** dealer 1~2초 구간(프레임 24~40)에서 수직으로 겹친 두 스택 중 하나만 인식. 텍스트 프롬프트만으로는 해결 어려움. 해결 후보:
   - 해상도 올리기 (720p/1080p, L40S로 가능)
   - 포인트 프롬프트 (사용자는 자동 감지를 선호)
   - 개별 단위 강조 프롬프트
2. **Over-count vs miss-count 균형:** 공격적 threshold는 많이 잡지만 하나의 뭉치가 여러 ID로 쪼개짐.
3. **CountGD 미설치:** 뭉치 단위 트래킹은 되지만 **개별 칩 수 카운팅**은 다음 단계.

---

## 13. 파일/경로 레퍼런스

**로컬:**
- `/Users/hyeonsup/chipcounting/`
  - `CLAUDE.md`, `SETUP.md` — 프로젝트 문서
  - `run_sam3_video.py` — 초기 video predictor 러너
  - `make_overlay.py` — overlay + 누적 카운트 생성
  - `sweep_configs.py`, `sweep_aggressive.py`, `sweep_stackprompt.py` — 스윕 드라이버
  - `inspect_state.py`, `prompt_sweep.py` — 디버깅 유틸
  - `pokervideo.mp4` — 5.77초 샘플
  - `pokervideo/dealer.mp4` — 20.8초 dealer 영상
  - `sam3-runs/` — 실험 결과
    - `run3/overlay_cum.mp4` (pokervideo, thr=0.2 → unique 20)
    - `dealer/overlay_cum.mp4` (dealer, thr=0.5 → unique 13)
    - `dealer_sweep/` — Sweep v1 (6개)
    - `dealer_sweep_v2/` — Sweep v2 공격적 (8개)
    - `dealer_sweep_stack/` — Sweep stack 프롬프트 (8개)

**원격 (g6e.12xlarge, 34.207.102.234):**
- `/home/ubuntu/sam3/` — repo
- `/home/ubuntu/sam3-checkpoints/` — 모델 가중치
- `/home/ubuntu/dealer_540p.mp4`, `/home/ubuntu/pokervideo_720p.mp4`
- `/home/ubuntu/sam3-runs/` — 스윕 결과
- `/tmp/make_overlay.py`, `/tmp/sweep_*.py` — 실행 스크립트

**인스턴스:**
- g5.8xlarge (i-08bc229c35862d86b, 54.86.183.205) — 계속 켜둠
- g6e.12xlarge (i-060c3e59bd75d6823, 34.207.102.234) — 현재 작업
- AMI: ami-06e4adfd722988d24

---

## 14. 다음 단계 후보

1. **겹친 스택 분리** — 해상도 높이거나 다른 프롬프트 전략
2. **CountGD 설치** — 개별 칩 카운팅 (conda env + gcc-11 + CUDA ops 빌드)
3. **뭉치 단위 크롭 저장** — CountGD 입력 준비
4. **색상 분류** — 칩 종류별 금액 매핑
5. **g5 stop** — 과금 절약 (마이그레이션 완료됨)
