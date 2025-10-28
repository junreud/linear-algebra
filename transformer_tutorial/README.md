# Transformer Tutorial: "Attention is All You Need" 구현과 분석

이 프로젝트는 "Attention is All You Need" 논문의 Transformer를 **두 가지 방식**으로 구현하여 내부 동작을 상세히 분석합니다.

## ⚡ 가장 빠른 시작 방법

### 🤖 완전 자동 실행 (추천!)
```bash
# 한 번의 명령어로 설치부터 실행까지!
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial
./setup_and_run.sh
```

**Windows 사용자:**
```cmd
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial
setup_and_run.bat
```

### 🔧 수동 설정 (단계별)
```bash
# 1. 프로젝트 디렉토리로 이동
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial

# 2. 가상환경 생성 및 활성화 (한 번만)
python3 -m venv transformer_env
source transformer_env/bin/activate  # Windows: transformer_env\Scripts\activate

# 3. 패키지 설치 (한 번만)
pip install -r requirements.txt

# 4. 실행!
python main.py --version both --epochs 2 --train_size 500
```

## ⚡ 빠른 시작 (Quick Start)

### 1. 프로젝트 클론 및 이동
```bash
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial
```

### 2. Python 가상환경 생성 및 활성화
```bash
# Python 가상환경 생성
python3 -m venv transformer_env

# 가상환경 활성화
source transformer_env/bin/activate  # macOS/Linux
# 또는 Windows의 경우: transformer_env\Scripts\activate

# 가상환경 활성화 확인 (프롬프트 앞에 (transformer_env) 표시됨)
which python  # 가상환경의 python 경로가 표시되어야 함
```

### 3. 필요한 패키지 설치
```bash
# requirements.txt의 패키지들 설치
pip install --upgrade pip
pip install -r requirements.txt

# 설치 확인
pip list | grep torch
pip list | grep transformers
```

### 4. 즉시 실행 (권장)
```bash
# 🚀 기본 실행 - 두 버전 모두 간단히 테스트
python main.py --version both --epochs 2 --train_size 500 --eval_size 100

# 🔍 PyTorch 버전 상세 분석
python main.py --version pytorch --epochs 3 --debug --visualize

# 🏭 Hugging Face 버전 실무 실행
python main.py --version huggingface --epochs 3 --track_internals

# 📊 분석 도구 데모
python main.py --version analysis
```

### 5. 가상환경 종료 (작업 완료 후)
```bash
deactivate
```

## 📁 프로젝트 구조

```
transformer_tutorial/
├── pytorch_version/          # PyTorch 순정 구현
│   ├── model.py             # Transformer 모델 (from scratch)
│   ├── attention.py         # Multi-Head Attention 구현
│   ├── layers.py            # Encoder/Decoder 레이어
│   └── train.py             # 학습 스크립트
├── huggingface_version/      # Hugging Face 구현
│   ├── model.py             # HF 기반 모델
│   ├── custom_trainer.py    # 커스텀 트레이너
│   └── train.py             # 학습 스크립트
├── utils/                    # 공통 유틸리티
│   ├── data.py              # 데이터 로더
│   ├── tokenizer.py         # 토크나이저
│   ├── visualization.py     # 시각화 도구
│   └── analysis.py          # QKV 분석 도구
└── requirements.txt
```

## 📋 필수 요구사항

### 시스템 요구사항
- **Python 3.8+** (권장: Python 3.9+)
- **pip** (Python 패키지 관리자)
- **8GB RAM 이상** (모델 학습용)
- **CUDA 지원 GPU** (선택사항, 속도 향상)

### Python 버전 확인
```bash
python3 --version  # Python 3.8+ 확인
pip --version      # pip 확인
```

### 🚀 가상환경 설치 및 실행 (필수!)

**가상환경을 사용하는 이유:**
- 시스템 Python 환경 보호
- 패키지 충돌 방지
- 프로젝트별 독립적 환경 관리

**1단계: 가상환경 생성**
```bash
# 현재 디렉토리에 가상환경 생성
python3 -m venv transformer_env

# 또는 시스템 전역에 생성하고 싶다면
python3 -m venv ~/venvs/transformer_env
```

**2단계: 가상환경 활성화**
```bash
# macOS/Linux
source transformer_env/bin/activate

# Windows
transformer_env\Scripts\activate

# 활성화 확인 (프롬프트에 (transformer_env) 표시됨)
which python  # 가상환경 Python 경로 확인
```

**3단계: 패키지 설치**
```bash
# 필수 패키지 설치 (한 번만 실행)
pip install --upgrade pip
pip install -r requirements.txt

# 설치 확인
pip list | grep torch
```

**4단계: 프로그램 실행**
```bash
# 기본 실행
python main.py

# 상세 옵션으로 실행
python main.py --version both --epochs 3 --train_size 1000
```

**5단계: 가상환경 종료 (작업 완료 후)**
```bash
deactivate
```

### 🔄 재실행할 때 (다음 번부터)
```bash
# 1. 디렉토리 이동
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial

# 2. 가상환경 활성화 (매번 필요)
source transformer_env/bin/activate

# 3. 실행
python main.py --version both

# 4. 종료
deactivate
```

## 📦 주요 의존성 패키지

핵심 패키지들:
- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Transformers 4.30+**: Hugging Face 트랜스포머
- **matplotlib, seaborn**: 시각화
- **wandb**: 실험 추적 (선택사항)
- **datasets**: 데이터셋 로딩

## 🎯 실행 옵션

### A) PyTorch 순정 버전 (From-Scratch)
- **QKV 계산 과정**: Query, Key, Value 변환 추적
- **Attention 메커니즘**: Scaled Dot-Product Attention 단계별 분석
- **Multi-Head**: 각 헤드별 attention weight 시각화
- **Linear/Non-linear 변환**: 언제, 어디서 일어나는지 추적
- **데이터 흐름**: Encoder → Decoder → Output 전체 파이프라인

### B) Hugging Face 버전 (Production-Ready)
- **Hook 기반 분석**: 내부 텐서 값 실시간 추적
- **실무형 구현**: 토크나이저, 데이터셋, 트레이너 풀스택
- **성능 최적화**: Mixed Precision, Gradient Accumulation
- **분산 학습**: Multi-GPU 지원

## 🔍 주요 분석 기능

1. **QKV 변화 추적**: 각 레이어에서 Query, Key, Value 값 변화
2. **Attention Weight 시각화**: 단어 간 attention 패턴 히트맵
3. **FFN 중간값**: Feed-Forward Network 내부 활성화 분석
4. **Gradient Flow**: 역전파 과정에서 gradient 변화
5. **토큰별 임베딩**: 입력 토큰이 어떻게 변환되는지 추적

## 🚀 시작하기

### 1. 환경 설정

#### Option A: 간단한 방법 (권장)
```bash
# 프로젝트 디렉토리로 이동
cd /Users/gimjunseog/projects/linear-algebra/transformer_tutorial

# 가상환경 생성 및 활성화
python3 -m venv transformer_env
source transformer_env/bin/activate

# 패키지 설치 및 즉시 실행
pip install -r requirements.txt
python main.py --version both --epochs 2 --train_size 500
```

#### Option B: 단계별 상세 설정
```bash
# 1️⃣ Python 가상환경 생성
python3 -m venv transformer_env

# 2️⃣ 가상환경 활성화 확인
source transformer_env/bin/activate
echo "가상환경 활성화됨: $(which python)"

# 3️⃣ pip 업그레이드
pip install --upgrade pip

# 4️⃣ 필수 패키지 설치
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.30.0 tokenizers>=0.13.0
pip install matplotlib seaborn numpy pandas tqdm

# 5️⃣ 나머지 패키지 설치
pip install -r requirements.txt

# 6️⃣ 설치 확인
python -c "import torch; print(f'PyTorch 버전: {torch.__version__}')"
python -c "import transformers; print(f'Transformers 버전: {transformers.__version__}')"
```

### 2. 실행 옵션

#### 🎯 목적별 실행 가이드

**A) Transformer 기본 이해 (초보자용)**
```bash
# PyTorch 순정 구현으로 내부 동작 이해
python main.py --version pytorch --epochs 3 --debug --visualize \
    --batch_size 8 --train_size 1000 --eval_size 200
```

**B) 실무 구현 학습 (중급자용)**
```bash
# Hugging Face 버전으로 production-ready 코드 학습
python main.py --version huggingface --epochs 5 --track_internals \
    --batch_size 16 --train_size 2000 --eval_size 500
```

**C) 완전한 비교 분석 (고급자용)**
```bash
# 두 버전 모두 실행하여 차이점 분석
python main.py --version both --epochs 5 --debug --track_internals --visualize \
    --batch_size 12 --train_size 3000 --eval_size 600
```

**D) 빠른 테스트**
```bash
# 분석 도구 데모 (실제 학습 없이 시각화 확인)
python main.py --version analysis

# 모델 구조 테스트
python main.py --version test
```

### 3. WandB 로깅 (선택사항)
```bash
# WandB 계정 설정 (처음 한 번만)
pip install wandb
wandb login  # 웹브라우저에서 API 키 입력

# WandB와 함께 실행
python main.py --version huggingface --epochs 10 --track_internals \
    --wandb --wandb_project my-transformer-study
```

## � 트러블슈팅

### 가상환경 관련 문제

**Q: 가상환경이 활성화되지 않아요**
```bash
# 해결방법 1: 전체 경로로 활성화
source /Users/gimjunseog/projects/linear-algebra/transformer_tutorial/transformer_env/bin/activate

# 해결방법 2: 가상환경 재생성
rm -rf transformer_env
python3 -m venv transformer_env
source transformer_env/bin/activate
```

**Q: pip install이 실패해요**
```bash
# pip 업그레이드 후 재시도
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchvision
pip install transformers
pip install matplotlib seaborn numpy pandas
```

**Q: (transformer_env) 표시가 안 보여요**
```bash
# 가상환경 활성화 확인
echo $VIRTUAL_ENV  # 경로가 표시되어야 함
which python       # transformer_env 내의 python 경로가 표시되어야 함
```

### 실행 관련 문제

**Q: CUDA/GPU 관련 오류**
```bash
# CPU 모드로 실행
python main.py --version pytorch --epochs 2 --batch_size 4

# GPU 사용 가능 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Q: 메모리 부족 오류**
```bash
# 배치 크기와 모델 크기 줄이기
python main.py --version pytorch --epochs 2 --batch_size 4 \
    --hidden_size 128 --train_size 500 --eval_size 100
```

**Q: matplotlib 백엔드 오류**
```bash
# 환경 변수 설정 후 실행
export MPLBACKEND=Agg
python main.py --version pytorch --visualize
```

### 패키지 관련 문제

**Q: transformers 버전 호환성 문제**
```bash
# 특정 버전 설치
pip install transformers==4.30.0 tokenizers==0.13.0

# 또는 최신 버전으로 업그레이드
pip install --upgrade transformers tokenizers
```

**Q: numpy/pandas 오류**
```bash
# 호환 가능한 버전 설치
pip install numpy==1.24.0 pandas==2.0.0
```

## 🧪 검증 및 테스트

### 설치 확인
```bash
# 가상환경에서 실행
source transformer_env/bin/activate

# 기본 임포트 테스트
python -c "
import torch
import transformers
import matplotlib.pyplot as plt
import numpy as np
print('✅ 모든 패키지 정상 임포트됨')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
"
```

### 빠른 동작 테스트
```bash
# 1분 내 완료되는 간단한 테스트
python main.py --version test

# 5분 내 완료되는 기본 실행
python main.py --version pytorch --epochs 1 --train_size 200 --eval_size 50 --debug
```

## 📋 실행 체크리스트

실행 전 확인사항:
- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 생성 및 활성화됨 `(transformer_env)`
- [ ] requirements.txt 패키지 설치됨
- [ ] 현재 디렉토리가 `transformer_tutorial/`임
- [ ] 디스크 여유 공간 1GB 이상

권장 첫 실행:
```bash
# ✅ 이 명령어로 시작하세요!
python main.py --version both --epochs 2 --train_size 500 --eval_size 100 --debug
```

## 🌟 실행 후 확인할 파일들

성공적으로 실행되면 다음 파일들이 생성됩니다:

```
results_pytorch/
├── checkpoints/
│   ├── best_model.pt              # 최고 성능 모델
│   └── checkpoint_epoch_*.pt      # 중간 체크포인트
├── visualizations/
│   ├── attention_plots/
│   │   └── attention_evolution.png  # Attention 패턴 변화
│   ├── training_curves.png         # 학습 곡선
│   └── qkv_evolution.png          # QKV 값 변화
└── logs/                          # 학습 로그

results_huggingface/
├── pytorch_model.bin              # 저장된 모델
├── config.json                    # 모델 설정
├── training_args.bin              # 학습 설정
└── visualizations/
    ├── attention/
    │   └── attention_step_*.png    # 스텝별 attention
    └── qkv/
        └── qkv_evolution.png      # QKV 변화 추적
```

## � 문제 해결 (Troubleshooting)

### 자주 발생하는 문제들

**1. Python 버전 문제**
```bash
# Python 3.8+ 확인
python3 --version

# 시스템에 여러 Python이 설치된 경우
which python3
which pip3
```

**2. 가상환경 활성화 안됨**
```bash
# 가상환경이 제대로 생성되었는지 확인
ls transformer_env/bin/  # activate 파일 확인

# 권한 문제인 경우
chmod +x transformer_env/bin/activate
source transformer_env/bin/activate
```

**3. 패키지 설치 오류**
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 개별 설치 시도
pip install torch torchvision torchaudio
pip install transformers
```

**4. CUDA/GPU 문제**
```bash
# CUDA 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"

# CPU만 사용하고 싶은 경우
python main.py --device cpu
```

**5. 메모리 부족**
```bash
# 작은 배치 사이즈로 실행
python main.py --batch_size 8 --train_size 100
```

**6. 완전 초기화가 필요한 경우**
```bash
# 가상환경 삭제 후 재생성
rm -rf transformer_env
python3 -m venv transformer_env
source transformer_env/bin/activate
pip install -r requirements.txt
```

### 더 자세한 설치 가이드
📖 상세한 설치 방법은 [INSTALL.md](INSTALL.md) 파일을 참조하세요!

## �💡 다음 단계

1. **결과 분석**: 생성된 시각화 파일들을 확인하여 Transformer 내부 동작 이해
2. **코드 읽기**: `pytorch_version/`과 `huggingface_version/` 코드 비교 분석
3. **실험 확장**: 더 큰 모델, 더 많은 데이터로 실험
4. **실제 적용**: 본인의 프로젝트에 학습한 내용 적용