# Transformer Tutorial 실행 가이드

"Attention is All You Need" 논문의 Transformer를 두 가지 방식으로 구현하고 내부 동작을 상세히 분석하는 튜토리얼입니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
cd transformer_tutorial
pip install -r requirements.txt
```

### 2. 기본 실행 (추천)
```bash
# 두 버전 모두 실행 (가장 교육적)
python main.py --version both --epochs 3 --debug --track_internals --visualize

# PyTorch 순정 버전만 (구현 이해용)
python main.py --version pytorch --epochs 5 --debug --visualize

# Hugging Face 버전만 (실무용)
python main.py --version huggingface --epochs 5 --track_internals --wandb
```

### 3. 분석 도구 데모
```bash
# 시각화 및 분석 도구 테스트
python main.py --version analysis
```

## 📁 프로젝트 구조

```
transformer_tutorial/
├── main.py                   # 메인 실행 스크립트
├── requirements.txt          # 의존성 패키지
├── README.md                # 프로젝트 개요
├── USAGE.md                 # 이 파일
├── pytorch_version/         # PyTorch 순정 구현
│   ├── attention.py         # Multi-Head Attention (QKV 추적)
│   ├── layers.py           # Encoder/Decoder 레이어
│   ├── model.py            # 완전한 Transformer 모델
│   └── train.py            # 학습 스크립트
├── huggingface_version/    # Hugging Face 구현
│   ├── model.py            # HF 호환 모델 (내부 추적)
│   ├── custom_trainer.py   # 커스텀 트레이너
│   └── train.py            # 학습 스크립트
└── utils/                  # 공통 유틸리티
    ├── visualization.py    # 시각화 도구
    └── analysis.py         # 분석 도구
```

## 🎯 각 버전별 특징

### A) PyTorch 순정 버전 (Educational)
**목적**: Transformer 내부 동작 완전 이해

**주요 기능**:
- ✨ **QKV 계산 과정 상세 추적**: Query, Key, Value 변환 단계별 출력
- 🔍 **Attention 메커니즘 분석**: Scaled Dot-Product Attention 중간값 모니터링
- 📊 **Multi-Head 시각화**: 각 헤드별 attention pattern 히트맵
- ⚡ **Linear/Non-linear 변환 추적**: ReLU, Layer Norm 적용 전후 비교
- 📈 **데이터 흐름 완전 추적**: Encoder → Decoder → Output 전체 파이프라인

**실행 예시**:
```bash
# 디버그 모드로 상세 로그 확인
python main.py --version pytorch --epochs 5 --debug --visualize \
    --batch_size 8 --hidden_size 256

# 작은 모델로 빠른 테스트
python main.py --version pytorch --epochs 2 --debug \
    --hidden_size 128 --num_layers 2
```

**출력 예시**:
```
=== Multi-Head Attention Forward Pass ===
Input shapes - Q: torch.Size([2, 10, 256]), K: torch.Size([2, 10, 256])
1. Linear Transformations:
Q after W_q: torch.Size([2, 10, 256]), mean: 0.0234, std: 0.8901
2. Reshape for Multi-Head:
Q reshaped: torch.Size([2, 8, 10, 32])
3. Attention Output: torch.Size([2, 8, 10, 32])
```

### B) Hugging Face 버전 (Production)
**목적**: 실무형 구현 + 고급 분석

**주요 기능**:
- 🏭 **Production-Ready**: 토크나이저, 데이터로더, 트레이너 풀스택
- 🎣 **Hook 기반 추적**: 실시간 내부 텐서 값 모니터링
- 📊 **WandB 통합**: 학습 과정 실시간 시각화
- ⚡ **Mixed Precision**: GPU 메모리 효율적 학습
- 🔄 **분산 학습 지원**: Multi-GPU 환경 대응

**실행 예시**:
```bash
# WandB 로깅과 함께 실행
python main.py --version huggingface --epochs 10 --track_internals \
    --wandb --wandb_project my-transformer-experiment

# 큰 모델 실험
python main.py --version huggingface --epochs 5 --track_internals \
    --hidden_size 512 --num_layers 6 --batch_size 32
```

## 📊 분석 및 시각화

### 1. Attention Pattern 분석
```bash
# 실행 후 생성되는 시각화
results_pytorch/visualizations/attention/
├── attention_step_100.png    # 특정 스텝의 attention 패턴
├── attention_evolution.png   # 학습 과정에서 변화
└── multihead_comparison.png  # 헤드간 비교
```

### 2. QKV 값 변화 추적
```bash
results_huggingface/visualizations/qkv/
├── qkv_evolution.png         # 레이어별 QKV 평균값 변화
├── layer_contributions.png   # 각 레이어의 기여도
└── gradient_flow.png         # Gradient 흐름 분석
```

### 3. 학습 곡선
```bash
results_*/
├── training_curves.png       # Loss, 메트릭 곡선
├── attention_weights_*.png   # Attention weight 히트맵
└── analysis_report.json     # 종합 분석 결과
```

## 🔬 고급 분석 기능

### 1. 종합 분석 실행
```python
from utils.analysis import TransformerAnalyzer
from utils.visualization import TransformerVisualizer

# 모델 로드 후
analyzer = TransformerAnalyzer(model)
results = analyzer.comprehensive_analysis(
    input_ids, target_ids, attention_mask, tokens,
    save_path="detailed_analysis.json"
)

# 시각화
visualizer = TransformerVisualizer("my_viz")
visualizer.plot_attention_heatmap(attention_weights, tokens)
```

### 2. 커스텀 분석
```python
# QKV 통계 분석
qkv_stats = analyzer.analyze_qkv_statistics(input_ids)

# Attention pattern 분류
attention_analysis = analyzer.analyze_attention_patterns(input_ids, tokens=tokens)

# Gradient flow 분석
gradient_analysis = analyzer.analyze_gradient_flow(input_ids, target_ids)
```

## 🎓 학습 로드맵

### 초급: Transformer 기본 이해
1. **PyTorch 버전 실행**: `--version pytorch --debug`
2. **QKV 과정 관찰**: 콘솔 출력에서 각 단계별 값 변화 확인
3. **Attention 시각화**: 생성된 히트맵으로 토큰간 관계 이해

### 중급: 내부 메커니즘 분석
1. **두 버전 비교**: `--version both`로 구현 차이점 이해
2. **학습 과정 추적**: QKV evolution 그래프로 학습 역학 분석
3. **Attention pattern 분류**: Local vs Global vs Focused 패턴 구분

### 고급: 실무 응용
1. **대용량 실험**: 더 큰 모델, 데이터셋으로 확장
2. **성능 최적화**: Mixed Precision, Gradient Accumulation
3. **커스텀 분석**: 특정 태스크에 맞는 분석 도구 개발

## 🛠️ 커스터마이징

### 1. 모델 크기 조정
```bash
# 소형 모델 (빠른 실험)
python main.py --hidden_size 128 --num_layers 2 --num_heads 4

# 대형 모델 (성능 실험)
python main.py --hidden_size 512 --num_layers 8 --num_heads 16
```

### 2. 데이터셋 크기 조정
```bash
# 작은 데이터셋 (빠른 테스트)
python main.py --train_size 1000 --eval_size 200

# 큰 데이터셋 (본격 실험)
python main.py --train_size 10000 --eval_size 2000
```

### 3. 분석 설정
```bash
# 최대 디버깅
python main.py --version pytorch --debug --visualize \
    --epochs 3 --batch_size 4

# 최대 추적
python main.py --version huggingface --track_internals \
    --wandb --epochs 10
```

## 🔧 문제해결

### 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --batch_size 4 --hidden_size 128

# Gradient accumulation 사용
python main.py --version huggingface --batch_size 8 --gradient_accumulation_steps 4
```

### 느린 실행
```bash
# 작은 모델로 시작
python main.py --epochs 2 --train_size 500 --eval_size 100

# CPU에서 실행시 workers 0으로
python main.py --dataloader_num_workers 0
```

### 시각화 문제
```bash
# matplotlib backend 설정
export MPLBACKEND=Agg  # headless 환경
python main.py --visualize
```

## 📈 확장 아이디어

### 1. 실제 데이터셋 적용
- WMT 번역 데이터셋
- WikiText 언어모델링
- GLUE 태스크 fine-tuning

### 2. 고급 기법 추가
- Gradient checkpointing
- Dynamic attention
- Sparse attention patterns

### 3. 분석 도구 확장
- Attention entropy 분석
- Layer-wise learning rate
- Probing 실험

## 💡 주요 인사이트

이 튜토리얼을 통해 다음을 이해할 수 있습니다:

1. **QKV 변환**: Query, Key, Value가 어떻게 계산되고 상호작용하는지
2. **Multi-Head의 효과**: 각 헤드가 다른 패턴을 학습하는 방식
3. **Attention 진화**: 학습 과정에서 attention 패턴이 어떻게 변화하는지
4. **Layer 기여도**: 각 레이어가 최종 출력에 미치는 영향
5. **구현 차이점**: From-scratch vs Library 구현의 장단점

**핵심**: Transformer는 단순한 attention 메커니즘의 조합이지만, 그 상호작용이 만들어내는 복잡성을 이해하는 것이 중요합니다.