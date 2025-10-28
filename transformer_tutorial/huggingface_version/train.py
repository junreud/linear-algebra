"""
Hugging Face 버전 Transformer 학습 스크립트

이 파일은 Hugging Face 생태계를 활용한 실무형 Transformer 학습을 담당합니다.
PyTorch 순정 구현과 달리, 다음과 같은 실무 기능들을 포함합니다:

주요 특징:
1. Hugging Face Trainer 사용 - 자동 최적화, 로깅, 체크포인트 저장
2. WandB 통합 - 실험 추적 및 시각화
3. 자동 토크나이저 - 텍스트 전처리 자동화
4. 데이터셋 API - 효율적인 데이터 로딩
5. Mixed Precision - 메모리 절약 및 속도 향상
6. Gradient Accumulation - 큰 배치 사이즈 시뮬레이션
"""
import torch  # PyTorch 기본 라이브러리
import torch.nn as nn  # 신경망 모듈들
from transformers import AutoTokenizer, DataCollatorForLanguageModeling  # HF 토크나이저와 데이터 콜레이터
from datasets import Dataset, load_dataset  # HF 데이터셋 라이브러리
import argparse  # 명령행 인자 파싱
import os  # 운영체제 인터페이스
import wandb  # Weights & Biases - 실험 추적 도구
from typing import List, Dict  # 타입 힌트

# 우리가 만든 모듈들 가져오기
from huggingface_version.model import create_tracked_transformer  # 커스텀 Transformer 모델
from huggingface_version.custom_trainer import (  # 커스텀 트레이너 관련
    CustomTransformerTrainer,    # 내부 상태 추적이 가능한 트레이너
    create_training_arguments,   # 훈련 설정 생성 함수
    SimpleTokenDataset          # 간단한 토큰 데이터셋
)


def create_simple_dataset(num_samples: int = 10000, vocab_size: int = 1000) -> Dataset:
    """
    간단한 언어 모델링 데이터셋 생성
    
    실무에서는 wikitext, openwebtext, C4 등의 대용량 텍스트 데이터를 사용하지만,
    여기서는 학습 과정 이해를 위해 간단한 패턴 기반 데이터셋을 생성합니다.
    
    Args:
        num_samples: 생성할 문장 수
        vocab_size: 어휘 사전 크기 (실제로는 사용되지 않음, 호환성용)
    
    Returns:
        HuggingFace Dataset 객체
    
    생성 과정:
    1. 문장 템플릿 정의
    2. 단어 목록 정의
    3. 랜덤하게 조합해서 문장 생성
    4. Dataset 형태로 반환
    """
    
    # 간단한 템플릿 문장들 - 실제 언어의 문법 구조를 간단히 모방
    templates = [
        "The {animal} {verb} in the {location} during {time}.",      # 동물이 장소에서 시간에 행동
        "A {adjective} {object} was {verb} by the {person}.",        # 형용사 물체가 사람에 의해 동사
        "In the {location}, {person} {verb} {adjective} {object}.",  # 장소에서 사람이 형용사 물체를 동사
        "During {time}, the {animal} {verb} {adjective}ly.",         # 시간에 동물이 형용사로 동사
        "The {person} saw a {adjective} {animal} near the {location}.",  # 사람이 장소 근처에서 형용사 동물을 봄
    ]
    
    # 각 카테고리별 단어 목록 - 실제 데이터의 다양성을 시뮬레이션
    words = {
        'animal': ['cat', 'dog', 'bird', 'fish', 'horse', 'tiger', 'lion', 'elephant'],
        'verb': ['runs', 'jumps', 'flies', 'swims', 'walks', 'sleeps', 'eats', 'plays'],
        'location': ['park', 'forest', 'beach', 'mountain', 'garden', 'lake', 'field', 'city'],
        'time': ['morning', 'evening', 'night', 'afternoon', 'dawn', 'dusk', 'midnight', 'noon'],
        'adjective': ['big', 'small', 'beautiful', 'fast', 'slow', 'bright', 'dark', 'quiet'],
        'object': ['book', 'car', 'house', 'tree', 'flower', 'stone', 'bridge', 'chair'],
        'person': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
    }
    
    import random  # 랜덤 조합을 위한 모듈
    
    texts = []  # 생성된 텍스트들을 저장할 리스트
    for _ in range(num_samples):
        template = random.choice(templates)
        
        # 템플릿의 플레이스홀더를 랜덤 단어로 채우기
        text = template
        for placeholder, word_list in words.items():
            if f'{{{placeholder}}}' in text:
                text = text.replace(f'{{{placeholder}}}', random.choice(word_list))
        
        texts.append(text)
    
    # Hugging Face Dataset 형식으로 변환
    dataset = Dataset.from_dict({'text': texts})
    return dataset


class CustomTokenizer:
    """
    간단한 토크나이저 (실제로는 BPE, SentencePiece 등 사용)
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
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
        
        for i, word in enumerate(common_words):
            if len(self.vocab) < vocab_size:
                self.vocab[word] = len(self.vocab)
        
        # 역방향 매핑
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할"""
        import re
        # 간단한 토큰화 (실제로는 더 정교함)
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """토큰을 ID로 변환"""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: int = None, 
               padding: str = None, truncation: bool = False) -> Dict:
        """텍스트 인코딩"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['<bos>'] + tokens + ['<eos>']
        
        token_ids = self.convert_tokens_to_ids(tokens)
        
        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Padding
        attention_mask = [1] * len(token_ids)
        if padding == 'max_length' and max_length:
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def __call__(self, text, **kwargs):
        """호출 가능한 인터페이스"""
        encoding = self.encode(text, **kwargs)
        return {
            'input_ids': torch.tensor([encoding['input_ids']]),
            'attention_mask': torch.tensor([encoding['attention_mask']])
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 변환"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<unk>')
            if skip_special_tokens and token in ['<pad>', '<bos>', '<eos>', '<unk>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)


class HFDataset(torch.utils.data.Dataset):
    """
    Hugging Face 스타일 데이터셋
    """
    
    def __init__(self, texts: List[str], tokenizer: CustomTokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(encoding['input_ids'], dtype=torch.long),  # For language modeling
        }


def setup_wandb(project_name: str = "transformer-tutorial", config: dict = None):
    """WandB 설정"""
    try:
        wandb.init(
            project=project_name,
            config=config,
            name=f"tracked-transformer-{wandb.util.generate_id()}"
        )
        print("WandB initialized successfully")
        return True
    except Exception as e:
        print(f"WandB initialization failed: {e}")
        return False


def create_datasets(tokenizer: CustomTokenizer, train_size: int = 8000, eval_size: int = 2000):
    """훈련/검증 데이터셋 생성"""
    
    print("Creating datasets...")
    
    # 텍스트 데이터 생성
    dataset = create_simple_dataset(train_size + eval_size)
    texts = dataset['text']
    
    # 훈련/검증 분할
    train_texts = texts[:train_size]
    eval_texts = texts[train_size:train_size + eval_size]
    
    # 데이터셋 생성
    train_dataset = HFDataset(train_texts, tokenizer)
    eval_dataset = HFDataset(eval_texts, tokenizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with Hugging Face')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./hf_results', help='Output directory')
    parser.add_argument('--track_internals', action='store_true', help='Track internal states')
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--wandb_project', type=str, default='transformer-tutorial', help='WandB project name')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--train_size', type=int, default=8000, help='Training dataset size')
    parser.add_argument('--eval_size', type=int, default=2000, help='Evaluation dataset size')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # WandB 설정
    wandb_enabled = False
    if args.wandb:
        config = vars(args)
        wandb_enabled = setup_wandb(args.wandb_project, config)
    
    # 토크나이저 생성
    print("Creating tokenizer...")
    tokenizer = CustomTokenizer(vocab_size=args.vocab_size)
    
    # 모델 생성
    print("Creating model...")
    model = create_tracked_transformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        track_internal_states=args.track_internals
    )
    
    # LM head 추가
    model.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 데이터셋 생성
    train_dataset, eval_dataset = create_datasets(
        tokenizer, 
        train_size=args.train_size, 
        eval_size=args.eval_size
    )
    
    # Training arguments
    training_args = create_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # GPU에서만 FP16 사용
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        report_to=["wandb"] if wandb_enabled else [],
    )
    
    # 트레이너 생성
    trainer = CustomTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        track_internals=args.track_internals,
        wandb_project=args.wandb_project if wandb_enabled else None,
    )
    
    # 학습 시작
    print(f"\nStarting training...")
    print(f"Training arguments:")
    for key, value in vars(training_args).items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    try:
        # 학습 실행
        trainer.train()
        
        # 최종 평가
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        
        # 시각화
        if args.track_internals:
            print("\nGenerating visualizations...")
            viz_dir = os.path.join(args.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            trainer.log_attention_patterns(os.path.join(viz_dir, "attention"))
            trainer.plot_qkv_evolution(os.path.join(viz_dir, "qkv"))
        
        # 모델 저장
        print(f"\nSaving final model...")
        trainer.save_model()
        
        # 간단한 생성 테스트
        print("\nTesting generation...")
        test_input = "The cat runs in the"
        input_encoding = tokenizer.encode(test_input, add_special_tokens=True, max_length=20, padding='max_length')
        input_ids = torch.tensor([input_encoding['input_ids']], device=device)
        attention_mask = torch.tensor([input_encoding['attention_mask']], device=device)
        
        model.eval()
        with torch.no_grad():
            # 단순 generation (실제로는 더 정교한 방법 사용)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model.lm_head(outputs.last_hidden_state)
            predicted_ids = logits.argmax(dim=-1)
            
            generated_text = tokenizer.decode(predicted_ids[0].cpu().tolist())
            print(f"Input: {test_input}")
            print(f"Generated: {generated_text}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 정리
        trainer.cleanup()
        print("\nTraining completed!")


if __name__ == "__main__":
    main()