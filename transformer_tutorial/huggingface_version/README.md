# HuggingFace Version - Transformer Tutorial

이 디렉토리는 HuggingFace 라이브러리와 실제 데이터셋을 사용한 고급 Transformer 구현을 제공합니다.

## 🚀 주요 특징

### 1. BERT Encoder + Transformer Decoder 하이브리드 모델
- **BERT의 양방향 인코더**: 소스 텍스트를 양방향으로 인코딩
- **Transformer의 인과적 디코더**: 타겟 텍스트를 순차적으로 생성
- **크로스 어텐션**: 인코더와 디코더 간 정보 전달
- **실제 번역 작업**: T5, BART와 유사한 구조

### 2. 실제 데이터셋 지원
- **HuggingFace Datasets**: 실제 번역 데이터셋 사용
- **다양한 언어 쌍**: EN-KO, EN-FR, EN-DE 등
- **고품질 데이터**: OPUS Books, WMT, OPUS-100

## 📁 파일 구조

```
huggingface_version/
├── README.md                        # 이 파일
├── encoder_decoder_model.py         # BERT+Transformer 하이브리드 모델
├── train_encoder_decoder_hf.py      # HuggingFace 데이터셋 훈련 스크립트
├── custom_trainer.py               # 커스텀 훈련기
├── model.py                        # 기본 모델들
└── train.py                        # 기본 훈련 스크립트
```

## 🏗️ 모델 아키텍처

### BERT Encoder + Transformer Decoder

```
입력 텍스트 (소스 언어)
    ↓
🔄 BERT Encoder (양방향)
- TrackedTransformerLayer × N
- Self-attention (bidirectional)
- Position embedding
    ↓
인코더 출력 (hidden states)
    ↓
🎯 Transformer Decoder (단방향)
- DecoderLayer × N  
- Masked self-attention (causal)
- Cross-attention (encoder → decoder)
- Feed forward
    ↓
출력 텍스트 (타겟 언어)
```

### 크로스 어텐션 메커니즘

```python
# Query: 디코더의 현재 상태
# Key, Value: 인코더의 출력
cross_attention_output = CrossAttention(
    query=decoder_hidden,      # 디코더에서
    key=encoder_hidden,        # 인코더에서  
    value=encoder_hidden       # 인코더에서
)
```

## 🚀 시작하기

### 1. 환경 설정

```bash
cd transformer_tutorial
source transformer_env/bin/activate

# 필요한 패키지 설치
pip install datasets transformers nltk
```

### 2. 기본 모델 테스트

```bash
cd huggingface_version
python test_encoder_decoder.py
```

### 3. HuggingFace 데이터셋으로 훈련

#### 영어 → 한국어 번역 (OPUS Books)
```bash
python train_encoder_decoder_hf.py \
    --dataset opus_books \
    --src_lang en --tgt_lang ko \
    --epochs 5 --num_samples 5000 \
    --batch_size 4 --gradient_accumulation_steps 4
```

#### 영어 → 프랑스어 번역 (WMT16)
```bash
python train_encoder_decoder_hf.py \
    --dataset wmt16 \
    --src_lang en --tgt_lang fr \
    --epochs 10 --batch_size 8 \
    --hidden_size 512 --encoder_layers 6
```

#### 빠른 테스트 (작은 모델)
```bash
python train_encoder_decoder_hf.py \
    --dataset opus100 \
    --src_lang en --tgt_lang de \
    --num_samples 1000 --epochs 2 \
    --hidden_size 128 --encoder_layers 2 --decoder_layers 2
```

### 4. HuggingFace 훈련 테스트
```bash
python test_hf_training.py
```

## 📊 지원 데이터셋

### 번역 데이터셋

| 데이터셋 | 설명 | 언어 쌍 예시 | 품질 |
|---------|------|-------------|------|
| `opus_books` | 다국어 도서 번역 | EN-KO, EN-FR, EN-DE | ⭐⭐⭐⭐⭐ |
| `wmt16` | WMT 2016 번역 대회 | EN-FR, EN-DE, EN-RU | ⭐⭐⭐⭐⭐ |
| `wmt14` | WMT 2014 번역 대회 | EN-FR, EN-DE | ⭐⭐⭐⭐⭐ |
| `opus100` | 100개 언어 번역 | EN-XX (100+ 언어) | ⭐⭐⭐⭐ |
| `kde4` | KDE 소프트웨어 번역 | 다양한 언어 쌍 | ⭐⭐⭐ |

### 데이터셋 선택 가이드

- **높은 품질**: `opus_books`, `wmt16`, `wmt14`
- **다양한 언어**: `opus100`
- **빠른 테스트**: `kde4`
- **한국어 포함**: `opus_books`, `opus100`

## 🔧 모델 설정

### 기본 설정 (33M 파라미터)
```python
config = EncoderDecoderConfig(
    hidden_size=256,
    encoder_layers=4,
    decoder_layers=4,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    max_position_embeddings=128
)
```

### 대형 설정 (100M+ 파라미터)
```python
config = EncoderDecoderConfig(
    hidden_size=512,
    encoder_layers=8,
    decoder_layers=8,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    max_position_embeddings=256
)
```

## 📈 훈련 결과 예시

```
🚀 Encoder-Decoder 모델 훈련 시작 (HuggingFace 데이터)
📊 훈련 설정:
   - 에포크: 5
   - 배치 크기: 4  
   - Gradient Accumulation: 4
   - 실제 배치 크기: 16
   - 학습률: 0.0001
   - 디바이스: mps
   - 모델 파라미터: 33,660,929

📈 Epoch 1/5
Training Epoch 1: 100%|██████████| 313/313 [03:45<00:00,  1.39it/s]
✅ Train Loss: 8.2451, BLEU: 0.0234
✅ Val Loss: 7.8932, BLEU: 0.0389

🔍 번역 샘플:
   1. 예측: Bonjour comment allez vous aujourd hui
      정답: Hello, how are you today?
```

## 🔍 번역 테스트

훈련된 모델로 번역 테스트:

```python
from train_encoder_decoder_hf import EncoderDecoderTrainerHF
import torch

# 체크포인트 로드
checkpoint = torch.load("encoder_decoder_checkpoints_hf/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# 번역 실행
trainer = EncoderDecoderTrainerHF(model, tokenizer, ...)
translation = trainer.translate_sample("Hello, how are you?")
print(f"번역 결과: {translation}")
```

## 🆚 기존 버전과의 차이점

### 기존 버전 (`train_encoder_decoder.py`)
- ❌ 하드코딩된 20개 번역 쌍
- ❌ 제한적인 학습 데이터
- ❌ 실제 성능 평가 어려움

### HuggingFace 버전 (`train_encoder_decoder_hf.py`)
- ✅ 실제 대규모 번역 데이터셋
- ✅ 다양한 언어 쌍 지원
- ✅ BLEU 점수 평가
- ✅ 체계적인 검증 데이터
- ✅ 품질 좋은 번역 결과

## 🎯 성능 개선 팁

### 1. 데이터 품질
- 고품질 데이터셋 선택 (`opus_books`, `wmt16`)
- 충분한 데이터 양 (10K+ 샘플)
- 적절한 문장 길이 필터링

### 2. 모델 크기
- 작은 모델: 128-256 hidden size (빠른 실험)
- 중간 모델: 512 hidden size (균형잡힌 성능)
- 큰 모델: 768+ hidden size (최고 성능)

### 3. 훈련 설정
- Learning rate: 1e-4 ~ 5e-4
- Batch size: 실제 배치 크기 16-32 (gradient accumulation 활용)
- Epochs: 데이터셋 크기에 따라 5-20

### 4. 평가 지표
- BLEU 점수: 번역 품질 측정
- 손실 값: 학습 진행 모니터링
- 샘플 번역: 정성적 평가

## 🚀 고급 사용법

### 1. 커스텀 데이터셋 추가
```python
class CustomTranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer):
        # 커스텀 데이터 로딩 로직
        pass
```

### 2. 다국어 토크나이저
```python
# 다양한 토크나이저 지원
tokenizers = {
    "multilingual": "bert-base-multilingual-cased",
    "korean": "klue/bert-base",
    "french": "dbmdz/bert-base-french-europeana-cased",
    "german": "bert-base-german-cased"
}
```

### 3. 체크포인트 재개
```python
# 훈련 재개
checkpoint = torch.load("encoder_decoder_checkpoints_hf/checkpoint_epoch_3.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 📚 참고 문헌

- **BERT**: Bidirectional Encoder Representations from Transformers
- **Transformer**: Attention Is All You Need
- **T5**: Text-to-Text Transfer Transformer  
- **BART**: Denoising Sequence-to-Sequence Pre-training
- **HuggingFace Datasets**: 🤗 Datasets Library

---

💡 **팁**: 실제 번역 모델 개발에서는 이러한 하이브리드 구조가 매우 효과적입니다. BERT의 강력한 인코딩 능력과 Transformer의 생성 능력을 결합하여 실용적인 번역 시스템을 구축할 수 있습니다!