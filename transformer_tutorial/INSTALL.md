# 🚀 완벽한 설치 및 실행 가이드

## 목차
1. [사전 요구사항](#사전-요구사항)
2. [단계별 설치](#단계별-설치)
3. [실행 가이드](#실행-가이드)
4. [결과 확인](#결과-확인)
5. [문제해결](#문제해결)

## 사전 요구사항

### 시스템 요구사항
- **운영체제**: macOS, Linux, Windows
- **Python**: 3.8 이상 (3.9~3.11 권장)
- **메모리**: 최소 4GB RAM (8GB 권장)
- **저장공간**: 2GB 이상
- **GPU**: 선택사항 (CUDA 지원 GPU 권장)

### Python 설치 확인
```bash
# Python 버전 확인
python3 --version  # Python 3.8+ 이어야 함

# pip 설치 확인
python3 -m pip --version
```

## 단계별 설치

### Step 1: 프로젝트 디렉토리로 이동
```bash
# 터미널에서 프로젝트 디렉토리로 이동
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial

# 현재 위치 확인
pwd
ls -la  # main.py, requirements.txt 등이 보여야 함
```

### Step 2: Python 가상환경 생성
```bash
# 가상환경 생성 (transformer_env라는 이름으로)
python3 -m venv transformer_env

# 생성 확인
ls -la transformer_env/  # bin, lib, include 폴더가 보여야 함
```

### Step 3: 가상환경 활성화
```bash
# macOS/Linux에서 활성화
source transformer_env/bin/activate

# 활성화 확인 - 프롬프트가 (transformer_env)로 시작해야 함
# 예: (transformer_env) user@computer:~/path$
```

**Windows 사용자의 경우:**
```cmd
# Windows에서 활성화
transformer_env\Scripts\activate
```

### Step 4: pip 업그레이드
```bash
# pip를 최신 버전으로 업그레이드
pip install --upgrade pip

# 버전 확인
pip --version
```

### Step 5: 패키지 설치

#### 방법 A: requirements.txt 일괄 설치 (권장)
```bash
# 모든 패키지 한 번에 설치
pip install -r requirements.txt

# 설치 과정에서 오류가 나면 방법 B 사용
```

#### 방법 B: 단계별 설치 (문제 발생시)
```bash
# 1. 핵심 패키지부터 설치
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0

# 2. Transformers 관련
pip install transformers>=4.30.0 tokenizers>=0.13.0

# 3. 데이터 처리
pip install numpy>=1.24.0 pandas>=2.0.0

# 4. 시각화
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# 5. 기타 유틸리티
pip install tqdm>=4.65.0 scikit-learn>=1.3.0

# 6. 선택사항 (고급 기능)
pip install datasets>=2.12.0 accelerate>=0.20.0
pip install tensorboard>=2.13.0  # 로깅용
pip install wandb>=0.15.0        # 실험 추적용 (선택)
```

### Step 6: 설치 확인
```bash
# 핵심 패키지 임포트 테스트
python -c "
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
print('✅ 모든 핵심 패키지가 정상적으로 설치되었습니다!')
print(f'PyTorch 버전: {torch.__version__}')
print(f'Transformers 버전: {transformers.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
"
```

성공시 출력 예시:
```
✅ 모든 핵심 패키지가 정상적으로 설치되었습니다!
PyTorch 버전: 2.0.1
Transformers 버전: 4.30.2
CUDA 사용 가능: True
```

## 실행 가이드

### 첫 실행 (추천)

#### 🎯 빠른 테스트 (1-2분)
```bash
# 설치가 제대로 되었는지 빠른 확인
python main.py --version test
```

#### 🔍 기본 실행 (5-10분)
```bash
# 두 버전 모두 간단히 실행
python main.py --version both --epochs 2 --train_size 500 --eval_size 100
```

출력 예시:
```
사용 가능한 디바이스: cuda:0
=== PYTORCH 순정 TRANSFORMER 실행 ===
Creating datasets...
Train batches: 32
Val batches: 7
Creating model...
Starting training for 2 epochs...
```

### 상세 분석 실행

#### 📚 PyTorch 버전 (교육용)
```bash
# 내부 동작을 상세히 볼 수 있는 디버그 모드
python main.py --version pytorch --epochs 3 --debug --visualize \
    --batch_size 8 --train_size 1000 --eval_size 200
```

실행 중 출력 예시:
```
=== Multi-Head Attention Forward Pass ===
Input shapes - Q: torch.Size([8, 10, 256]), K: torch.Size([8, 10, 256])
1. Linear Transformations:
Q after W_q: torch.Size([8, 10, 256]), mean: 0.0234, std: 0.8901
2. Reshape for Multi-Head:
Q reshaped: torch.Size([8, 8, 10, 32])
3. Attention Output: torch.Size([8, 8, 10, 32])
```

#### 🏭 Hugging Face 버전 (실무용)
```bash
# 실무에서 사용하는 방식의 구현
python main.py --version huggingface --epochs 5 --track_internals \
    --batch_size 16 --train_size 2000 --eval_size 500
```

#### 📊 WandB 로깅 포함 (고급)
```bash
# 실험 추적 도구와 함께 실행
pip install wandb  # 아직 설치 안했다면
wandb login        # 처음 한 번만 (웹브라우저에서 API 키 입력)

python main.py --version huggingface --epochs 10 --track_internals \
    --wandb --wandb_project my-transformer-experiment
```

### 매개변수 설명

| 매개변수 | 설명 | 기본값 | 예시 |
|---------|------|--------|------|
| `--version` | 실행할 버전 | `both` | `pytorch`, `huggingface`, `both` |
| `--epochs` | 학습 에폭 수 | `5` | `--epochs 10` |
| `--batch_size` | 배치 크기 | `16` | `--batch_size 32` |
| `--hidden_size` | 모델 차원 | `256` | `--hidden_size 512` |
| `--train_size` | 훈련 데이터 크기 | `2000` | `--train_size 5000` |
| `--debug` | 상세 로그 출력 | `False` | `--debug` |
| `--visualize` | 시각화 생성 | `False` | `--visualize` |
| `--track_internals` | 내부 상태 추적 | `False` | `--track_internals` |

## 결과 확인

### 실행 완료 후 생성되는 파일들

```
transformer_tutorial/
├── results_pytorch/                 # PyTorch 버전 결과
│   ├── checkpoints/
│   │   ├── best_model.pt           # 최고 성능 모델
│   │   └── checkpoint_epoch_*.pt   # 중간 체크포인트
│   ├── visualizations/
│   │   ├── training_curves.png     # 학습 곡선
│   │   ├── attention_plots/
│   │   │   └── attention_evolution.png  # Attention 변화
│   │   └── qkv_evolution.png       # QKV 값 변화
│   └── logs/                       # 상세 로그
├── results_huggingface/            # Hugging Face 버전 결과
│   ├── pytorch_model.bin           # 저장된 모델
│   ├── config.json                 # 모델 설정
│   ├── visualizations/
│   │   ├── attention/
│   │   │   └── attention_step_*.png
│   │   └── qkv/
│   │       └── qkv_evolution.png
│   └── runs/                       # TensorBoard 로그
└── sample_visualizations/           # 분석 도구 데모 결과
    ├── sample_attention_heatmap.png
    ├── sample_multihead_attention.png
    └── sample_training_curves.png
```

### 결과 파일 확인 방법

#### 1. 시각화 결과 확인
```bash
# 생성된 이미지 파일들 확인
ls -la results_*/visualizations/
open results_pytorch/visualizations/training_curves.png  # macOS
```

#### 2. 학습 로그 확인
```bash
# 학습 과정 로그 확인
tail -n 20 results_pytorch/logs/training.log
```

#### 3. 모델 파일 확인
```bash
# 저장된 모델 크기 확인
ls -lh results_*/checkpoints/
ls -lh results_*/pytorch_model.bin
```

## 문제해결

### 자주 발생하는 문제들

#### 1. 가상환경 활성화 문제
**증상**: `(transformer_env)` 표시가 안 보임
```bash
# 해결: 전체 경로로 활성화
source /Users/gimjunseog/projects/linear-algebra/transformer_tutorial/transformer_env/bin/activate

# 확인
echo $VIRTUAL_ENV  # 경로가 나와야 함
which python       # transformer_env 내부 python이어야 함
```

#### 2. 패키지 설치 실패
**증상**: `pip install` 명령어 실패
```bash
# 해결 1: pip 업그레이드
pip install --upgrade pip setuptools wheel

# 해결 2: 캐시 삭제 후 재설치
pip cache purge
pip install --no-cache-dir -r requirements.txt

# 해결 3: 개별 설치
pip install torch  # 하나씩 설치해보기
```

#### 3. CUDA 관련 오류
**증상**: GPU 관련 에러 메시지
```bash
# 해결: CPU 모드로 실행
python main.py --version pytorch --epochs 2 --batch_size 4

# GPU 상태 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### 4. 메모리 부족
**증상**: `CUDA out of memory` 또는 시스템 메모리 부족
```bash
# 해결: 배치 크기와 모델 크기 줄이기
python main.py --version pytorch --epochs 2 \
    --batch_size 4 --hidden_size 128 --train_size 300 --eval_size 50
```

#### 5. 시각화 오류
**증상**: matplotlib 관련 에러
```bash
# 해결: 백엔드 설정
export MPLBACKEND=Agg  # GUI 없는 환경용
python main.py --version pytorch --visualize

# 또는 X11 포워딩 (SSH 사용시)
ssh -X username@hostname
```

#### 6. 임포트 에러
**증상**: `ModuleNotFoundError`
```bash
# 해결 1: 가상환경 확인
which python  # transformer_env 내부여야 함

# 해결 2: 패키지 재설치
pip uninstall torch transformers
pip install torch transformers

# 해결 3: Python path 확인
python -c "import sys; print(sys.path)"
```

### 성능 튜닝

#### 빠른 실행을 위한 설정
```bash
# 최소 설정으로 빠른 테스트
python main.py --version pytorch --epochs 1 \
    --batch_size 4 --train_size 200 --eval_size 50 \
    --hidden_size 128 --num_layers 2
```

#### 고성능 실행을 위한 설정
```bash
# GPU가 있는 경우 최대 성능
python main.py --version huggingface --epochs 20 \
    --batch_size 64 --train_size 10000 --eval_size 2000 \
    --hidden_size 512 --num_layers 8 --track_internals
```

## 실행 체크리스트

실행 전 확인:
- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 생성됨 (`transformer_env/` 폴더 존재)
- [ ] 가상환경 활성화됨 (프롬프트에 `(transformer_env)` 표시)
- [ ] 패키지 설치됨 (`pip list | grep torch` 결과 있음)
- [ ] 현재 디렉토리가 `transformer_tutorial/`
- [ ] `main.py` 파일 존재함

첫 실행 명령어:
```bash
# ✅ 이 명령어로 시작하세요!
python main.py --version both --epochs 2 --train_size 500 --eval_size 100
```

성공적인 실행의 신호:
- 터미널에 `사용 가능한 디바이스: cpu` 또는 `cuda` 출력
- `Creating datasets...` 메시지 출력
- `results_*/` 폴더 생성
- 에폭별 loss 값 출력

## 가상환경 관리

### 가상환경 종료
```bash
# 작업 완료 후 가상환경 비활성화
deactivate
```

### 가상환경 재시작
```bash
# 다음 작업시 가상환경 다시 활성화
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial
source transformer_env/bin/activate
python main.py --version test  # 빠른 동작 확인
```

### 가상환경 삭제 (필요시)
```bash
# 가상환경 완전 삭제 후 재생성
deactivate
rm -rf transformer_env
python3 -m venv transformer_env
source transformer_env/bin/activate
pip install -r requirements.txt
```

이제 완벽하게 설치하고 실행할 수 있습니다! 🚀