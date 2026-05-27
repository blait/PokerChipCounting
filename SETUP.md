# Poker Chip Counting with Meta SAM3

## 프로젝트 목표
포커 플레이 비디오(대각 앵글)에서 칩 뭉치를 트래킹하고 개별 칩 개수/종류를 카운팅

## 아키텍처

```
포커 비디오 (대각 앵글)
  ↓
SAM3 (비디오 모드) → 칩 뭉치 세그멘테이션 + 트래킹
  ↓
뭉치별 크롭 (매 프레임)
  ↓
CountGD → 크롭된 뭉치에서 개별 칩 카운팅
  ↓
색상 분류 → 칩 종류별 금액 매핑
  ↓
총 칩 수 + 금액 산출
```

### SAM3 프롬프트 방식
- **텍스트 프롬프트**: `"poker chip stack"` → 자동 감지
- **포인트 프롬프트**: 좌표 지정으로 놓친 칩 추가 / 오인식 제거
- **박스 프롬프트**: 바운딩박스로 영역 지정
- 세 가지를 혼합 사용 가능

### 왜 SAM3 + CountGD 조합인가
- SAM3: 뭉치 트래킹 우수, 개별 칩 카운팅 불가
- CountGD: 밀집 객체 개별 카운팅 특화, 텍스트 프롬프트 기반 (zero-shot)
- 두 모델 조합으로 학습 데이터 없이 동작 가능

## 참고 논문/사례

| 논문 | 학회 | 핵심 |
|------|------|------|
| CountGD + SAM2 자동 라벨링 (심명보 외) | 제어로봇시스템학회 2025 | CountGD 카운팅 + SAM2 세그멘테이션 조합, 농업 도메인 |
| PseCo | CVPR 2024 | Point → SAM 세그멘테이션 → CLIP 분류 파이프라인 |
| OCCAM | arXiv 2026 | SAM2 + 클러스터링, 학습 없이 멀티클래스 카운팅 |
| OmniCount | AAAI 2025 | SAM + 시맨틱 프라이어, 다중 카테고리 동시 카운팅 |
| CountGD | NeurIPS 2024 | GroundingDINO 기반 오픈월드 카운팅, SOTA |
| Crowd Counting + SAM | ECCV 2024 | SAM + GMM 군중 카운팅 |

- **포커 칩 카운팅은 학술적으로 거의 미개척** (GitHub에 MVP 1개만 존재: TianYiY1103/pkr.img, CV 미구현)

## AWS 인스턴스 정보

| 항목 | 값 |
|------|-----|
| Instance ID | `i-08bc229c35862d86b` |
| Type | `g5.8xlarge` (32 vCPU, 128GB RAM, A10G 24GB VRAM) |
| AMI | `ami-077c6fac5ef663f46` (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10, Ubuntu 24.04) |
| Region | us-east-1 |
| Public IP | `54.86.183.205` (stop/start 시 변경됨) |
| 커널 | `6.17.0-1010-aws` (apt-mark hold로 고정) |
| Key Pair | `sam3-chipcounting` (`~/.ssh/sam3-chipcounting.pem`) |
| Security Group | `sg-0b5336cb7dcbf2fe3` (sam3-dev-sg, SSH 22번 포트) |
| EBS | 100GB gp3 |

### SSH 접속
```bash
ssh -i ~/.ssh/sam3-chipcounting.pem ubuntu@54.86.183.205
```

## 인스턴스 환경

| 구성 | 버전 |
|------|------|
| Python | 3.13 (venv: `/opt/pytorch/bin/activate`) |
| PyTorch | 2.10.0+cu130 |
| CUDA | 13.0 |
| Flash Attention | 2.8.3 |
| SAM3 | 0.1.0 (`/home/ubuntu/sam3`) |
| HuggingFace | 로그인 완료 (토큰 저장됨) |

### PyTorch 환경 활성화
```bash
source /opt/pytorch/bin/activate
```

## 인스턴스 재생성 절차

```bash
# 1. 인스턴스 생성 (user-data로 커널 업데이트 차단)
cat <<'USERDATA' > /tmp/sam3-userdata.sh
#!/bin/bash
systemctl stop unattended-upgrades
systemctl disable unattended-upgrades
apt-mark hold linux-image-* linux-headers-* linux-modules-*
USERDATA

aws ec2 run-instances \
  --image-id ami-077c6fac5ef663f46 \
  --instance-type g5.8xlarge \
  --key-name sam3-chipcounting \
  --security-group-ids sg-0b5336cb7dcbf2fe3 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sam3-chipcounting}]' \
  --user-data file:///tmp/sam3-userdata.sh \
  --region us-east-1

# 2. SSH 접속 후 커널 업데이트 차단 확인 및 보강
sudo systemctl disable unattended-upgrades
sudo apt-mark hold linux-image-* linux-headers-* linux-modules-* linux-aws

# 3. SAM3 설치
source /opt/pytorch/bin/activate
cd /home/ubuntu
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"
pip install einops ninja

# 4. Flash Attention 설치 (nohup 필수, 약 1시간 소요)
nohup bash -c "source /opt/pytorch/bin/activate && MAX_JOBS=4 NVCC_THREADS=2 pip install flash-attn --no-build-isolation > /home/ubuntu/flash-attn-build.log 2>&1" &
# 진행 확인: tail -f /home/ubuntu/flash-attn-build.log

# 5. HuggingFace 로그인 및 모델 다운로드
hf auth login --token <YOUR_TOKEN>
hf download facebook/sam3 --local-dir /home/ubuntu/sam3-checkpoints
```

## 삽질 기록 및 교훈

### flash-attn 빌드 OOM 문제
- flash-attn 소스 빌드는 `MAX_JOBS × NVCC_THREADS × 5GB` 메모리 필요
- 기본값으로 빌드 시 g5.2xlarge (32GB)에서 OOM → sshd 죽음 → SSH 접속 불가
- **해결:** 128GB 인스턴스(g5.8xlarge) 사용 + `MAX_JOBS=4 NVCC_THREADS=2` + nohup 백그라운드
- flash-attn 없이도 SAM3 동작 가능 (PyTorch 내장 `scaled_dot_product_attention` fallback)

### NVIDIA 드라이버 깨짐
- `unattended-upgrades`가 커널을 자동 업데이트 → nvidia DKMS 모듈과 불일치
- 예: 드라이버는 `6.17.0-1009-aws`용인데 커널이 `6.17.0-1010-aws`로 업데이트됨
- **근본 해결:** user-data로 `unattended-upgrades` 비활성화 + `apt-mark hold linux-*`
- **사후 복구 (인스턴스 살아있을 때):**
  ```bash
  sudo apt-get install -y linux-headers-$(uname -r)
  sudo dkms install nvidia/580.126.16 -k $(uname -r)
  sudo modprobe nvidia
  nvidia-smi  # 확인
  ```

### Security Group
- SSH 접근은 현재 IP 기준으로 제한됨
- IP 변경 시 Security Group 인바운드 규칙 업데이트 필요:
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-0b5336cb7dcbf2fe3 \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" --region us-east-1
```

## 비용 참고
- g5.8xlarge: ~$3.25/hr
- **사용하지 않을 때 인스턴스 종료(terminate) 권장**
- stop/start 시 IP 변경됨 + 드라이버 복구 필요할 수 있음 (위 사후 복구 참고)

## 현재 상태
- [x] AWS 인스턴스 세팅
- [x] SAM3 설치
- [x] Flash Attention 2.8.3 빌드/설치
- [x] HuggingFace 로그인
- [x] 커널 자동 업데이트 차단
- [x] HuggingFace `facebook/sam3` 모델 접근 승인 완료
- [x] 모델 체크포인트 다운로드 (`/home/ubuntu/sam3-checkpoints`, 6.5GB)
- [ ] CountGD 설치
- [ ] 포커 칩 감지/세그멘테이션 테스트
- [ ] 칩 종류별 분류 및 카운팅 로직 구현
