"""
Hugging Face 버전 Transformer 학습 스크립트

이 파일은 Hugging Face 생태계를 활용한 실무형 Transformer 학습을 담당합니다.
PyTorch 순정 구현과 달리, 다음과 같은 실무 기능들을 포함합니다:

주요 특징:
1. 실제 데이터셋 사용 - wikitext, imdb 등 Hugging Face Hub 데이터셋
2. 실제 토크나이저 사용 - GPT-2, BERT 등 사전 훈련된 토크나이저
3. Hugging Face Trainer 사용 - 자동 최적화, 로깅, 체크포인트 저장
4. WandB 통합 - 실험 추적 및 시각화
5. Mixed Precision - 메모리 절약 및 속도 향상
6. 내부 상태 추적 - QKV, attention weights 등 실시간 모니터링
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
import argparse
import os
import wandb
from typing import List, Dict
import numpy as np

# 우리가 만든 모듈들 가져오기
from huggingface_version.model import create_tracked_transformer
from huggingface_version.custom_trainer import (
    CustomTransformerTrainer,
    create_training_arguments,
    SimpleTokenDataset
)


def load_real_dataset(
    dataset_name: str = "wikitext", 
    dataset_config: str = "wikitext-2-raw-v1",
    num_samples: int = 10000,
    split: str = "train"
) -> Dataset:
    """
    Hugging Face Hub에서 실제 데이터셋 로딩
    
    실무에서 사용하는 실제 텍스트 데이터셋을 로드합니다.
    
    Args:
        dataset_name: 데이터셋 이름 ('wikitext', 'imdb', 'ag_news', 'amazon_polarity' 등)
        dataset_config: 데이터셋 설정 (예: 'wikitext-2-raw-v1', 'wikitext-103-raw-v1')
        num_samples: 사용할 샘플 수 (전체 데이터셋이 클 경우 제한)
        split: 데이터 분할 ('train', 'validation', 'test')
    
    Returns:
        HuggingFace Dataset 객체
    
    지원하는 데이터셋들:
    1. wikitext: 위키피디아 텍스트 (언어 모델링용)
    2. imdb: 영화 리뷰 (감정 분석용, 텍스트로도 사용 가능)
    3. ag_news: 뉴스 분류 데이터
    4. amazon_polarity: 아마존 리뷰 감정 분석
    5. bookcorpus: 책 텍스트 (대용량)
    """
    
    print(f"📚 실제 데이터셋 로딩 중: {dataset_name} ({dataset_config})")
    print(f"   분할: {split}, 최대 샘플 수: {num_samples:,}")
    
    try:
        # Hugging Face Hub에서 데이터셋 로드
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        print(f"✅ 데이터셋 로드 완료: {len(dataset):,} 샘플")
        
        # 샘플 수 제한 (메모리 및 훈련 시간 고려)
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
            print(f"📝 샘플 수 제한: {len(dataset):,} 샘플 사용")
        
        # 데이터셋 구조 확인 및 텍스트 필드 통일
        print(f"📋 데이터셋 컬럼: {list(dataset.features.keys())}")
        
        # 텍스트 필드 찾기 및 'text'로 통일
        text_field = None
        if 'text' in dataset.features:
            text_field = 'text'
        elif 'sentence' in dataset.features:
            text_field = 'sentence'
        elif 'review' in dataset.features:
            text_field = 'review'
        elif 'content' in dataset.features:
            text_field = 'content'
        else:
            # 첫 번째 문자열 필드를 텍스트로 사용
            for key, feature in dataset.features.items():
                if feature.dtype == 'string':
                    text_field = key
                    break
        
        if text_field is None:
            raise ValueError(f"텍스트 필드를 찾을 수 없습니다. 사용 가능한 필드: {list(dataset.features.keys())}")
        
        if text_field != 'text':
            dataset = dataset.rename_column(text_field, 'text')
            print(f"🔄 텍스트 필드 '{text_field}' → 'text'로 이름 변경")
        
        # 빈 텍스트 제거
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x['text'] and len(x['text'].strip()) > 10)
        print(f"🧹 빈 텍스트 제거: {original_len:,} → {len(dataset):,} 샘플")
        
        # 샘플 텍스트 출력
        print(f"\n📄 샘플 텍스트 (처음 3개):")
        for i in range(min(3, len(dataset))):
            text = dataset[i]['text'][:100]  # 처음 100자만
            print(f"   {i+1}. {text}{'...' if len(dataset[i]['text']) > 100 else ''}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        print(f"🔄 기본 synthetic 데이터셋으로 fallback")
        return create_synthetic_dataset(num_samples)


def create_synthetic_dataset(num_samples: int = 10000) -> Dataset:
    """
    간단한 synthetic 데이터셋 생성 (fallback용)
    
    실제 데이터셋 로딩이 실패했을 때 사용하는 백업 함수입니다.
    """
    
    # 간단한 템플릿 문장들
    templates = [
        "The {animal} {verb} in the {location} during {time}.",
        "A {adjective} {object} was {verb} by the {person}.",
        "In the {location}, {person} {verb} {adjective} {object}.",
        "During {time}, the {animal} {verb} {adjective}ly.",
        "The {person} saw a {adjective} {animal} near the {location}.",
        "Yesterday, {person} visited the {location} and saw many {animal}s.",
        "The {adjective} {object} belongs to {person} who lives in {location}.",
        "Every {time}, {animal} {verb} near the old {object} in the {location}."
    ]
    
    # 단어 목록
    words = {
        'animal': ['cat', 'dog', 'bird', 'fish', 'horse', 'tiger', 'lion', 'elephant', 'rabbit', 'deer'],
        'verb': ['runs', 'jumps', 'flies', 'swims', 'walks', 'sleeps', 'eats', 'plays', 'sits', 'stands'],
        'location': ['park', 'forest', 'beach', 'mountain', 'garden', 'lake', 'field', 'city', 'village', 'river'],
        'time': ['morning', 'evening', 'night', 'afternoon', 'dawn', 'dusk', 'midnight', 'noon', 'sunrise', 'sunset'],
        'adjective': ['big', 'small', 'beautiful', 'fast', 'slow', 'bright', 'dark', 'quiet', 'loud', 'colorful'],
        'object': ['book', 'car', 'house', 'tree', 'flower', 'stone', 'bridge', 'chair', 'table', 'window'],
        'person': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
    }
    
    import random
    
    texts = []
    for _ in range(num_samples):
        template = random.choice(templates)
        text = template
        for placeholder, word_list in words.items():
            if f'{{{placeholder}}}' in text:
                text = text.replace(f'{{{placeholder}}}', random.choice(word_list))
        texts.append(text)
    
    dataset = Dataset.from_dict({'text': texts})
    return dataset


def create_tokenizer(model_name: str = "gpt2", vocab_size: int = None):
    """
    실제 사전 훈련된 토크나이저 생성
    토크나이저가 잘 훈련되었고 아니고의 차이는 
        간단: 단어 단위로만 분할
        훈련된: 의미 있는 서브워드로 분할
        "tokenization" → ["token", "ization"]
        "tokenizer" → ["token", "izer"]
        "tokenizing" → ["token", "izing"]
    이런식으로 단어의 파생된 부분을 이해하냐 못하냐가 될 수도 있고, 기호가 들어간 문장 등도 잘 구분하게됨
    이 토크나이저의 학습은 일반적인 딥러닝 훈련과는 아예 다름.
    단순 통계 기반 알고리즘으로 미분, 로스율 계산이 없이 다량의 데이터(라벨데이터 필요없응)만으로 학습 가능.

    Args:
        model_name: 사용할 토크나이저 모델 ('gpt2', 'bert-base-uncased', 'distilbert-base-uncased' 등)
        vocab_size: 어휘 사전 크기 (None이면 원본 크기 사용)
    
    Returns:
        AutoTokenizer 객체
    
    지원하는 토크나이저들:
    1. gpt2: GPT-2 토크나이저 (BPE 기반)
    2. bert-base-uncased: BERT 토크나이저 (WordPiece 기반)
    3. distilbert-base-uncased: DistilBERT 토크나이저
    4. t5-small: T5 토크나이저 (SentencePiece 기반)
    """
    
    print(f"🔤 토크나이저 로딩 중: {model_name}")
    
    try:
        # Hugging Face에서 사전 훈련된 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 패딩 토큰 설정 (GPT-2는 기본적으로 패딩 토큰이 없음)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"📝 패딩 토큰을 EOS 토큰으로 설정: '{tokenizer.eos_token}'")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"📝 새로운 패딩 토큰 추가: '[PAD]'")
        
        print(f"✅ 토크나이저 로드 완료:")
        print(f"   어휘 사전 크기: {len(tokenizer):,}")
        print(f"   패딩 토큰: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        
        # 토크나이저 테스트
        test_text = "Hello, this is a test sentence for tokenization."
        test_tokens = tokenizer.tokenize(test_text)
        test_ids = tokenizer.encode(test_text)
        
        print(f"\n🧪 토크나이저 테스트:")
        print(f"   입력: '{test_text}'")
        print(f"   토큰: {test_tokens[:10]}{'...' if len(test_tokens) > 10 else ''}")
        print(f"   ID: {test_ids[:10]}{'...' if len(test_ids) > 10 else ''}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ 토크나이저 로딩 실패: {e}")
        print(f"🔄 기본 간단 토크나이저로 fallback")
        return create_simple_tokenizer(vocab_size or 1000)


class SimpleTokenizer: # 보험용, 라이브러리 미설치, 인터넷 연결문제 등 발생 시  실행
    """간단한 토크나이저 (fallback용)"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # 간단한 vocab 구축
        self.vocab = {
            '<pad>': self.pad_token_id,
            '<unk>': self.unk_token_id,
            '<bos>': self.bos_token_id,
            '<eos>': self.eos_token_id,
        }
        
        # 추가 토큰들
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'is', 'was', 'are', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'cat', 'dog', 'bird', 'fish',
            'runs', 'jumps', 'flies', 'swims', 'walks', 'park', 'forest', 'beach', 'big', 'small'
        ]
        
        for word in common_words:
            if len(self.vocab) < vocab_size:
                self.vocab[word] = len(self.vocab)
        
        # 역방향 매핑
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def __len__(self):
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할"""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: int = None, 
               padding: str = None, truncation: bool = False, return_tensors: str = None):
        """텍스트 인코딩"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['<bos>'] + tokens + ['<eos>']
        
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Attention mask
        attention_mask = [1] * len(token_ids)
        
        # Padding
        if padding == 'max_length' and max_length:
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        result = {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 변환"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<unk>')
            if skip_special_tokens and token in ['<pad>', '<bos>', '<eos>', '<unk>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)


def create_simple_tokenizer(vocab_size: int = 1000):
    """간단한 토크나이저 생성 (fallback용)"""
    return SimpleTokenizer(vocab_size)


def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int = 128, num_proc: int = 4):
    """
    데이터셋 전처리 - 토크나이제이션 및 패딩
    
    Args:
        dataset: 원본 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        num_proc: 병렬 처리 프로세스 수
    
    Returns:
        전처리된 데이터셋
    """
    
    def tokenize_function(examples):
        """배치 토크나이제이션 함수"""
        # 텍스트 리스트를 토크나이즈
        if hasattr(tokenizer, 'encode_batch'):  # 실제 HF 토크나이저
            return tokenizer(
                examples['text'],       # 리스트 입력
                truncation=True,        # 최대 길이 초과 시 자르기
                padding='max_length',   # 최대 길이에 맞게 패딩
                max_length=max_length,  # 최대 길이 설정
                return_tensors=None     # 리스트로 반환
            )
        else:  # 간단한 토크나이저
            results = {'input_ids': [], 'attention_mask': []}
            for text in examples['text']:
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True
                )
                results['input_ids'].append(encoded['input_ids'])
                results['attention_mask'].append(encoded['attention_mask'])
            return results
    
    print(f"🔄 데이터셋 토크나이제이션 중...")
    print(f"   최대 길이: {max_length}")
    print(f"   병렬 프로세스: {num_proc}")
    
    # 토크나이제이션 실행
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,                       # 배치 단위 처리
        num_proc=num_proc,                  # 병렬 처리 프로세스 수
        remove_columns=dataset.column_names,  # 원본 텍스트 제거함
        desc="토크나이제이션 진행 중"
    )
    """
    텍스트: ["Hello world", "This is a test", "Short"]
    토크나이제이션 후:
        {
            'input_ids': [
                [15496, 995, 0, 0, 0],     # "Hello world" + 패딩      패딩이 존재하는 이유: GPU/CPU는 같은 길이의 시퀀스만 처리 가능
                [1212, 318, 257, 1332, 0], # "This is a test" + 패딩
                [17896, 0, 0, 0, 0]        # "Short" + 패딩
            ],
            'attention_mask': [
                [1, 1, 0, 0, 0],           # 처음 2개만 유효
                [1, 1, 1, 1, 0],           # 처음 4개만 유효
                [1, 0, 0, 0, 0]            # 처음 1개만 유효
            ]
        }
    """
    # labels 컬럼 추가 (언어 모델링용 - input_ids와 동일) 위의 토크나이제이션 후의 데이터에 labels 컬럼이 추가됨. 
    # input_ids와 동일한 값 가짐. 모델 훈련 시 로스 계산에 사용됨.
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    print(f"✅ 토크나이제이션 완료: {len(tokenized_dataset):,} 샘플")
    
    # 샘플 출력
    print(f"\n📄 토크나이제이션 샘플:")
    sample = tokenized_dataset[0]
    print(f"   input_ids 길이: {len(sample['input_ids'])}")
    print(f"   input_ids: {sample['input_ids'][:20]}...")
    print(f"   attention_mask: {sample['attention_mask'][:20]}...")
    
    if hasattr(tokenizer, 'decode'): # 사람이 읽기 편하게. skip_special_tokens=True: 패딩, BOS, EOS 토큰 무시
        decoded = tokenizer.decode(sample['input_ids'][:50], skip_special_tokens=True) 
        print(f"   디코딩된 텍스트: '{decoded[:100]}...'")
    
    return tokenized_dataset


def setup_wandb(project_name: str = "transformer-tutorial", config: dict = None):
    """WandB 설정"""
    try:
        # WandB 초기화
        wandb.init(
            project=project_name,
            config=config,
            name=f"tracked-transformer-{wandb.util.generate_id()}"
        )
        print("✅ WandB 초기화 완료")
        return True
    except Exception as e:
        print(f"❌ WandB 초기화 실패: {e}")
        return False


def main():
    """메인 학습 함수"""
    parser = argparse.ArgumentParser(description='Hugging Face Transformer 학습')
    
    # 데이터셋 설정
    parser.add_argument('--dataset_name', type=str, default='wikitext', 
                       help='데이터셋 이름 (wikitext, imdb, ag_news 등)')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', 
                       help='데이터셋 설정')
    parser.add_argument('--train_size', type=int, default=8000, help='훈련 데이터 크기')
    parser.add_argument('--eval_size', type=int, default=2000, help='검증 데이터 크기')
    
    # 토크나이저 설정
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', 
                       help='토크나이저 이름 (gpt2, bert-base-uncased 등)')
    parser.add_argument('--max_length', type=int, default=128, help='최대 시퀀스 길이') # 입력 시퀀스의 최대 길이, 배치 내에서 max_length 길이로 패딩됨
    
    # 모델 설정
    parser.add_argument('--vocab_size', type=int, default=None, help='어휘 사전 크기 (자동 추정)')
    parser.add_argument('--hidden_size', type=int, default=256, help='숨김 차원')
    parser.add_argument('--num_layers', type=int, default=6, help='레이어 수')
    parser.add_argument('--num_heads', type=int, default=8, help='어텐션 헤드 수')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=3, help='에폭 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--output_dir', type=str, default='./hf_results', help='출력 디렉토리')
    
    # 추적 및 로깅 설정
    parser.add_argument('--track_internals', action='store_true', help='내부 상태 추적')
    parser.add_argument('--wandb', action='store_true', help='WandB 사용')
    parser.add_argument('--wandb_project', type=str, default='hf-transformer-tutorial', 
                       help='WandB 프로젝트 이름')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"HUGGING FACE TRANSFORMER 학습 시작")
    print(f"{'='*80}")
    print(f"📊 설정:")
    print(f"   데이터셋: {args.dataset_name} ({args.dataset_config})")
    print(f"   토크나이저: {args.tokenizer_name}")
    print(f"   모델: {args.hidden_size}d, {args.num_layers}L, {args.num_heads}H")
    print(f"   학습: {args.epochs} epochs, lr={args.learning_rate}")
    print(f"   내부 추적: {args.track_internals}")
    
    # 1. 데이터셋 로딩
    print(f"\n📚 1단계: 데이터셋 로딩")
    dataset = load_real_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.train_size + args.eval_size,
        split="train"
    )
    
    # 2. 토크나이저 생성
    print(f"\n🔤 2단계: 토크나이저 생성")
    tokenizer = create_tokenizer(args.tokenizer_name, args.vocab_size)
    
    # 어휘 사전 크기 자동 설정
    if args.vocab_size is None:
        args.vocab_size = len(tokenizer)
        print(f"📝 어휘 사전 크기 자동 설정: {args.vocab_size:,}")
    
    # 3. 데이터셋 전처리
    print(f"\n🔄 3단계: 데이터셋 전처리")
    tokenized_dataset = preprocess_dataset(dataset, tokenizer, args.max_length)
    
    # 훈련/검증 분할 || 본적 없는 데이터로 검증하기 위해 분할. evaluation dataset은 train dataset과 겹치지 않도록
    train_dataset = tokenized_dataset.select(range(args.train_size))
    eval_dataset = tokenized_dataset.select(range(args.train_size, args.train_size + args.eval_size))
    
    print(f"✅ 데이터 분할 완료:")
    print(f"   훈련: {len(train_dataset):,} 샘플")
    print(f"   검증: {len(eval_dataset):,} 샘플")
    
    # 4. 모델 생성
    print(f"\n🤖 4단계: 모델 생성")
    model = create_tracked_transformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        track_internal_states=args.track_internals
    )
    
    print(f"✅ 모델 생성 완료:")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   총 파라미터: {total_params:,}")
    print(f"   훈련 가능: {trainable_params:,}")
    
    # 5. WandB 설정 (선택사항)
    if args.wandb:
        print(f"\n📊 5단계: WandB 설정")
        config = {
            'dataset_name': args.dataset_name,
            'tokenizer_name': args.tokenizer_name,
            'vocab_size': args.vocab_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'total_params': total_params,
        }
        setup_wandb(args.wandb_project, config)
    
    # 6. 훈련 인수 생성
    print(f"\n⚙️ 6단계: 훈련 설정")
    training_args = create_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.wandb else None
    )
    
    # 7. 데이터 콜레이터 생성
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer if hasattr(tokenizer, 'pad_token_id') else None,
        mlm=False,  # 자기회귀 언어 모델링 (GPT 스타일)
    )
    
    # 8. 트레이너 생성
    print(f"\n🏃 7단계: 트레이너 생성")
    trainer = CustomTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        track_internal_states=args.track_internals
    )
    
    # 9. 학습 시작
    print(f"\n🚀 8단계: 학습 시작")
    print(f"{'='*80}")
    
    trainer.train()
    
    # 10. 결과 저장
    print(f"\n💾 9단계: 결과 저장")
    trainer.save_model()
    trainer.save_state()
    
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(args.output_dir)
        print(f"✅ 토크나이저 저장: {args.output_dir}")
    
    print(f"\n🎉 학습 완료!")
    print(f"   결과 저장 위치: {args.output_dir}")
    print(f"   체크포인트: {args.output_dir}/checkpoint-*")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()