"""
BERT Encoder + Transformer Decoder 모델 학습 스크립트

이 스크립트는 Encoder-Decoder 모델을 학습시킵니다.
간단한 번역 작업 (영어 → 한국어)으로 테스트해봅니다.

사용법:
    python train_encoder_decoder.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder_decoder_model import BERTEncoderTransformerDecoderModel, EncoderDecoderConfig
import os
import json
from tqdm import tqdm
import numpy as np


class SimpleTranslationDataset(Dataset):
    """
    간단한 번역 데이터셋
    
    영어 → 한국어 번역을 위한 더미 데이터셋입니다.
    실제로는 더 큰 병렬 코퍼스를 사용해야 합니다.
    """
    
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 간단한 번역 예시들 (실제로는 큰 데이터셋 사용)
        self.translation_pairs = [
            ("Hello world", "안녕하세요 세계"),
            ("How are you?", "어떻게 지내세요?"),
            ("I love programming", "저는 프로그래밍을 좋아합니다"),
            ("The weather is nice", "날씨가 좋습니다"),
            ("Good morning", "좋은 아침입니다"),
            ("Thank you very much", "정말 감사합니다"),
            ("See you later", "나중에 봐요"),
            ("I am learning Korean", "저는 한국어를 배우고 있습니다"),
            ("This is a beautiful day", "오늘은 아름다운 날입니다"),
            ("Can you help me?", "도와주실 수 있나요?"),
            ("I like to read books", "저는 책 읽기를 좋아합니다"),
            ("The food is delicious", "음식이 맛있습니다"),
            ("Where is the library?", "도서관이 어디에 있나요?"),
            ("I want to learn more", "더 배우고 싶습니다"),
            ("Happy birthday", "생일 축하합니다"),
            ("What time is it?", "몇 시인가요?"),
            ("I am from Korea", "저는 한국에서 왔습니다"),
            ("Nice to meet you", "만나서 반갑습니다"),
            ("Have a good day", "좋은 하루 보내세요"),
            ("I understand", "이해합니다"),
        ]
        
        # 데이터 확장 (학습용으로 더 많은 예시 생성)
        self.data = self.translation_pairs * 10  # 10배 복제
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        english, korean = self.data[idx]
        
        # Encoder 입력 (영어)
        encoder_inputs = self.tokenizer(
            english,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Decoder 입력 및 레이블 (한국어)
        # Teacher Forcing: 입력은 [BOS] + 토큰들, 레이블은 토큰들 + [EOS]
        korean_with_bos = f"[CLS] {korean}"  # BOS 토큰 추가
        korean_with_eos = f"{korean} [SEP]"  # EOS 토큰 추가
        
        decoder_inputs = self.tokenizer(
            korean_with_bos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = self.tokenizer(
            korean_with_eos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(0),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(0),
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze(0),
            'decoder_attention_mask': decoder_inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0),
        }


class EncoderDecoderTrainer:
    """
    Encoder-Decoder 모델 학습기
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset=None,
        batch_size=8,
        learning_rate=5e-4,
        num_epochs=10,
        device='cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 데이터 로더
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # macOS에서 안정성을 위해 0으로 설정
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # 옵티마이저 및 스케줄러
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # 데이터를 디바이스로 이동
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 순전파
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels'],
                return_dict=True
            )
            
            loss = outputs.loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 (안정성을 위해)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 손실 기록
            total_loss += loss.item()
            num_batches += 1
            
            # 진행률 표시 업데이트
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """검증"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels'],
                    return_dict=True
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """전체 학습 루프"""
        print(f"🚀 Encoder-Decoder 모델 학습 시작!")
        print(f"📊 학습 설정:")
        print(f"   - 에포크: {self.num_epochs}")
        print(f"   - 배치 크기: {self.train_loader.batch_size}")
        print(f"   - 학습률: {self.optimizer.param_groups[0]['lr']}")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\n📈 Epoch {epoch + 1}/{self.num_epochs}")
            
            # 학습
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
                
                print(f"✅ Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
                    print(f"💾 최고 성능 모델 저장 (Val Loss: {val_loss:.4f})")
            else:
                print(f"✅ Train Loss: {train_loss:.4f}")
            
            # 주기적으로 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        print(f"\n🎉 학습 완료!")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename):
        """체크포인트 저장"""
        os.makedirs("encoder_decoder_checkpoints", exist_ok=True)
        filepath = os.path.join("encoder_decoder_checkpoints", filename)
        
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.model.config,
        }, filepath)
    
    def generate_sample(self, source_text, max_length=50):
        """샘플 번역 생성"""
        self.model.eval()
        
        with torch.no_grad():
            # 입력 토크나이징
            inputs = self.tokenizer(
                source_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                do_sample=False  # Greedy decoding
            )
            
            # 디코딩
            generated_text = self.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            return generated_text


def main():
    """메인 실행 함수"""
    
    # 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"🖥️  사용 디바이스: {device}")
    
    # 토크나이저 로드 (BERT 토크나이저 사용)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    print(f"📝 토크나이저 로드 완료: {tokenizer.__class__.__name__}")
    
    # 모델 설정
    config = EncoderDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_intermediate_size=2048,
        decoder_intermediate_size=2048,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    
    # 모델 생성
    model = BERTEncoderTransformerDecoderModel(config)
    print(f"🤖 모델 생성 완료")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 데이터셋 생성
    train_dataset = SimpleTranslationDataset(tokenizer, max_length=128)
    val_dataset = SimpleTranslationDataset(tokenizer, max_length=128)  # 동일한 데이터 사용 (실제로는 분리해야 함)
    
    print(f"📚 데이터셋 로드 완료")
    print(f"   - 학습 데이터: {len(train_dataset)}개")
    print(f"   - 검증 데이터: {len(val_dataset)}개")
    
    # 학습기 생성
    trainer = EncoderDecoderTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,  # 작은 배치 크기로 시작
        learning_rate=1e-4,
        num_epochs=5,
        device=device
    )
    
    # 학습 전 샘플 생성
    print(f"\n🔍 학습 전 샘플 생성:")
    sample_text = "Hello world"
    generated = trainer.generate_sample(sample_text)
    print(f"   입력: {sample_text}")
    print(f"   출력: {generated}")
    
    # 학습 실행
    train_losses, val_losses = trainer.train()
    
    # 학습 후 샘플 생성
    print(f"\n🔍 학습 후 샘플 생성:")
    generated = trainer.generate_sample(sample_text)
    print(f"   입력: {sample_text}")
    print(f"   출력: {generated}")
    
    # 여러 샘플 테스트
    test_samples = [
        "How are you?",
        "Thank you very much",
        "Good morning",
        "I love programming"
    ]
    
    print(f"\n🧪 다양한 샘플 테스트:")
    for sample in test_samples:
        generated = trainer.generate_sample(sample)
        print(f"   {sample} → {generated}")
    
    print(f"\n💾 체크포인트 저장 위치: encoder_decoder_checkpoints/")


if __name__ == "__main__":
    main()