# SAM3 Video Predictor 설정값 레퍼런스

`sam3/sam3/model_builder.py`의 `build_sam3_video_model()` 내부에 하드코딩된 파라미터들.
현재는 kwargs로 외부에서 못 바꿔서 `sed`로 직접 수정해야 함.

---

## 공간: 프레임마다 detector가 박스를 뱉으면 tracker가 어떻게 처리하는지

```
프레임 N 입장
 ├─ detector → 박스들 + score
 ├─ score_threshold_detection 필터 (score < 값은 버림)
 ├─ det_nms_thresh로 겹치는 박스 제거
 ├─ 남은 박스들 ─── 기존 tracklet과 IoU 비교 (assoc_iou_thresh)
 │                    ├─ 매칭 → 그 tracklet 업데이트
 │                    └─ 안 매칭 + score ≥ new_det_thresh → 새 tracklet 시작
 └─ tracklet 유지 관리 (max_trk_keep_alive, suppress_*)
```

---

## 가장 중요한 4개

### `score_threshold_detection` (현재 0.3)
**뜻:** detector 출력 중 이 confidence 미만은 전부 버림. "이게 칩이다"라고 인정할 최소 점수.

**올리면 (0.5~0.8):**
- 오검출 감소
- 진짜 있는 뭉치도 점수 낮으면 놓침
- tracklet 수 급감 → GPU 메모리 안정

**내리면 (0.05~0.2):**
- 애매한 뭉치도 감지 → 가려진 뭉치, 작은 칩까지 잡힘
- 노이즈(카드 무늬, 테이블 패턴) 오검출 증가
- tracklet 증가 → GPU 메모리 가파르게 증가

**실험 데이터 (dealer 영상):**
| score | unique | 관찰 |
|---|---|---|
| 0.5 | 13 | 안정, 놓친 뭉치 존재 |
| 0.3 | 17 | 균형 |
| 0.2 | 27~52 | over-count 시작 |
| 0.05 | 48 | 노이즈 많음 |

---

### `new_det_thresh` (현재 0.7)
**뜻:** 기존 tracklet과 매칭 안 되는 detection이 "새로운 뭉치"로 인정받을 문턱. `score_threshold_detection`보다 엄격.

**올리면 (0.7~0.9):**
- 새 ID 생성 보수적 → tracklet 수 안정
- 가려졌다 다시 보이는 뭉치가 기존 ID에 붙을 확률↑
- 진짜 새로 등장한 뭉치를 놓칠 수 있음

**내리면 (0.3~0.5):**
- 새 ID 쉽게 생성 → unique 수 급증
- GPU 메모리 monotone 증가 (OOM 위험)
- 가려졌다 돌아온 뭉치도 새 ID로 잡힘 (over-count)

**실험 데이터:**
| new_det | unique |
|---|---|
| 0.7 | 9~14 |
| 0.5 | 11~18 |
| 0.3 | 17~52 |

---

### `assoc_iou_thresh` (현재 0.1, 안 바꿔봄)
**뜻:** 현재 프레임 detection과 이전 프레임 tracklet 위치의 IoU가 이 값 이상이면 "같은 오브젝트"로 판단.

**올리면 (0.3~0.5):**
- 매칭 조건 엄격 → 같은 뭉치가 조금 움직여도 매칭 실패 → 새 ID로 생성
- tracklet 분할 심해짐

**내리면 (0.05~0.1):**
- 느슨한 매칭 → 이동 큰 뭉치도 같은 ID 유지
- 너무 낮으면 다른 뭉치끼리도 합쳐질 위험

---

### `det_nms_thresh` (현재 0.1, 안 바꿔봄)
**뜻:** detector가 같은 뭉치에 여러 박스 뱉을 때, IoU가 이 이상이면 낮은 score 박스 제거 (Non-Maximum Suppression).

**올리면 (0.3~0.5):**
- 겹친 박스도 둘 다 유지 → **겹친 두 스택 문제 완화 가능성**
- 하지만 detector가 애초에 두 스택을 다른 박스로 내야 함

**내리면 (0.05):**
- 중복 박스 공격적 제거 → 같은 위치 하나만 살아남음

---

## 트래킹 수명 관리

### `max_trk_keep_alive` (현재 30)
**뜻:** 감지 안 된 상태로 몇 프레임까지 tracklet 살려둘지. 30프레임 = 1초(30fps 기준).

**올리면:**
- 일시적으로 가려진 뭉치도 ID 유지
- 죽은 tracklet이 메모리 오래 차지 → OOM 위험

**내리면:**
- 잠깐 가려지면 바로 tracklet 죽음 → 다시 보일 때 새 ID (over-count)

---

### `min_trk_keep_alive` (현재 -1)
-1은 "사용 안 함"이라는 뜻. 보통은 tracklet이 최소 몇 프레임 살아 있어야 삭제 가능한지를 정함.

---

### `init_trk_keep_alive` (현재 30)
**뜻:** tracklet이 처음 만들어졌을 때 초기에 부여받는 수명. 새 tracklet이 첫 프레임에 바로 죽지 않도록.

---

### `hotstart_delay` (현재 15)
**뜻:** 영상 시작 후 처음 N프레임은 "hotstart 구간"으로 간주, 엄격하게 판정. 카메라 흔들림/초기 노이즈 대응.

**올리면:** 초반에 tracklet 적게 생성 → 시작부 장면에 뭉치 있어도 놓칠 수 있음.

---

### `hotstart_unmatch_thresh` (현재 8), `hotstart_dup_thresh` (현재 8)
hotstart 구간 내에서의 매칭 실패/중복 허용 횟수. 튜닝 영향 적음.

---

## 마스크 품질 관리

### `suppress_overlapping_based_on_recent_occlusion_threshold` (현재 0.7)
**뜻:** 최근 가려진 적 있는 tracklet이 다른 마스크와 겹치면 억제. 0.7 이상 겹치면 숨김.

**올리면 (0.9):** 거의 억제 안 함 → 중복 마스크 많아질 수 있음.
**내리면 (0.3):** 조금만 겹쳐도 억제 → 겹친 뭉치 사라짐 (현재 겹친 스택 문제와 연관 가능성).

---

### `fill_hole_area` (현재 16)
**뜻:** 마스크 내부 작은 구멍(< N픽셀)은 채워버림.

**올리면:** 마스크 깨끗해짐, 실제 공간 왜곡.
**내리면/0:** 모든 구멍 유지, raw 마스크.

---

### `recondition_every_nth_frame` (현재 16)
**뜻:** N프레임마다 고신뢰 detection으로 tracklet의 마스크/메모리를 재조정.

**올리면:** 재조정 덜 함 → tracklet drift.
**내리면 (8, 4):** 자주 재조정 → 최신 detection 반영 잘 됨, 연산 ↑.

---

### `suppress_det_close_to_boundary` (현재 False)
영상 경계에 가까운 detection 버릴지. 포커 영상에선 끌 것.

---

### `masklet_confirmation_enable` (현재 False)
켜면 새 tracklet이 N프레임 연속 감지돼야만 확정. False면 첫 프레임부터 바로 카운트.

---

### `decrease_trk_keep_alive_for_empty_masklets` (현재 False)
빈 마스크 tracklet 수명을 단축할지.

---

## 입력/전처리

### `image_size` (현재 1008)
**뜻:** 모델 내부에서 리사이즈할 크기. 1008px로 짧은 변 맞춤.

**올리면 (1280, 1536):**
- 작은 칩/세부 구분 잘 됨
- VRAM 제곱으로 증가, 느려짐
- **겹친 스택 문제 해결에 유효할 수 있음**

**내리면 (640, 768):**
- 빠름, 메모리 적게 씀
- 작은 뭉치 놓침

---

### `image_mean`, `image_std` (0.5, 0.5, 0.5)
입력 정규화. 건드릴 일 없음.

---

## 사용자 입력 (스크립트 레벨)

### `prompt` (텍스트)
현재 실험에선 `poker chip`, `poker chip stack`, `stack of poker chips`, `casino chip` 등 테스트.

**관찰:**
- `poker chip` → 개별 칩 단위로 잡음 (over-count 경향)
- `poker chip stack`, `stack of poker chips` → 뭉치 단위로 잡음 (깔끔)
- `casino chip`, `gambling token` → `poker chip`과 비슷하지만 살짝 적게 잡음
- `pile of chips` → 감지 수 적음

---

### 포인트/박스 프롬프트 (옵션)
사용자가 좌표 지정해서 "여기도 추적해"라고 강제. 자동 감지와 함께 사용 가능.

---

## 실행 환경

### `PYTORCH_ALLOC_CONF=expandable_segments:True`
GPU 메모리 파편화 완화. OOM 경계에서 도움. 항상 켜는 걸 추천.

### `image_size` × 프레임 수 × 평균 tracklet 수 ≈ VRAM
대략의 메모리 공식. dealer 영상(519프레임 + tracklet 20+)에서 A10G 24GB로는 1280px 불가.

---

## 튜닝 시나리오별 추천

### 뭉치를 최대한 많이 잡고 싶음
- `score_threshold_detection=0.1`
- `new_det_thresh=0.3`
- prompt=`poker chip`
- 주의: over-count, GPU 메모리 높음

### 깔끔하게 뭉치 단위만
- `score_threshold_detection=0.3`
- `new_det_thresh=0.7`
- prompt=`stack of poker chips` 또는 `poker chip stack`

### GPU 작은데 긴 영상
- `score_threshold_detection=0.4`
- `new_det_thresh=0.7`
- `image_size=640`
- `PYTORCH_ALLOC_CONF=expandable_segments:True`

### 겹친 뭉치 구분 개선
- `image_size=1280` 또는 1536
- `det_nms_thresh=0.3` (실험 필요)
- `suppress_overlapping_based_on_recent_occlusion_threshold=0.9`
- 마지막 수단: 포인트 프롬프트

---

## 아직 안 만져본 설정들 (실험 여지)

- `assoc_iou_thresh` — 매칭 엄격도
- `det_nms_thresh` — 중복 박스 허용도 (**겹친 스택 문제에 유력**)
- `image_size` — 해상도 (**세부 구분에 유력**)
- `suppress_overlapping_based_on_recent_occlusion_threshold` — 겹침 처리
- `recondition_every_nth_frame` — tracklet drift 보정 주기
