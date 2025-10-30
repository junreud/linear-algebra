"""
BERT Encoder + Transformer Decoder 학습 스크립트 (HuggingFace 데이터셋 버전)

실제 번역 데이터셋(WMT, OPUS 등)을 사용하여 Encoder-Decoder 모델을 학습합니다.

사용법:
    python train_encoder_decoder_hf.py --dataset opus_books --src_lang en --tgt_lang ko --epochs 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from transformers import AutoTokenizer
from datasets import load_dataset
from encoder_decoder_model import BERTEncoderTransformerDecoderModel, EncoderDecoderConfig
import os
import json
from tqdm import tqdm
import numpy as np
import argparse
from typing import Dict, List, Optional, Union


class HuggingFaceTranslationDataset(Dataset):
    """
    HuggingFace 번역 데이터셋
    
    지원 데이터셋:
    - opus_books: 다국어 도서 번역 (고품질)
    - wmt16: WMT 2016 번역 대회 데이터
    - wmt14: WMT 2014 번역 대회 데이터  
    - opus100: 100개 언어 번역 데이터
    - kde4: KDE 소프트웨어 번역
    """
    
    def __init__(
        self,
        dataset_name: str = "opus_books",
        src_lang: str = "en",
        tgt_lang: str = "ko", 
        tokenizer_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        num_samples: Optional[int] = None,
        split: str = "train"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        print(f"📚 HuggingFace 번역 데이터셋 로딩: {dataset_name} ({src_lang} → {tgt_lang})")
        
        # 데이터셋별 로딩 방식
        if dataset_name == "opus_books":
            dataset = load_dataset("opus_books", f"{src_lang}-{tgt_lang}", split=split)
            
        elif dataset_name.startswith("wmt"):
            # WMT 데이터셋
            if dataset_name == "wmt16":
                dataset = load_dataset("wmt16", f"{src_lang}-{tgt_lang}", split=split)
            elif dataset_name == "wmt14":
                dataset = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split=split)
            else:
                raise ValueError(f"지원하지 않는 WMT 데이터셋: {dataset_name}")
                
        elif dataset_name == "opus100":
            dataset = load_dataset("opus100", f"{src_lang}-{tgt_lang}", split=split)
            
        elif dataset_name == "kde4":
            dataset = load_dataset("kde4", f"{src_lang}-{tgt_lang}", split=split)
            
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
        
        # 샘플 수 제한
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        print(f"✅ 데이터셋 로드 완료: {len(dataset)}개 번역 쌍")
        
        # 번역 쌍 추출
        self.translation_pairs = []
        
        for item in tqdm(dataset, desc="번역 쌍 추출"):
            if dataset_name == "opus_books":
                src_text = item['translation'][src_lang]
                tgt_text = item['translation'][tgt_lang]
            elif dataset_name.startswith("wmt"):
                src_text = item['translation'][src_lang]
                tgt_text = item['translation'][tgt_lang]
            elif dataset_name == "opus100":
                src_text = item['translation'][src_lang]
                tgt_text = item['translation'][tgt_lang]
            elif dataset_name == "kde4":
                src_text = item['translation'][src_lang]
                tgt_text = item['translation'][tgt_lang]
            
            # 빈 텍스트나 너무 짧은 텍스트 스킵
            if not src_text.strip() or not tgt_text.strip():
                continue
            if len(src_text.split()) < 3 or len(tgt_text.split()) < 3:
                continue
            if len(src_text) > 500 or len(tgt_text) > 500:  # 너무 긴 텍스트 스킵
                continue
                
            self.translation_pairs.append((src_text.strip(), tgt_text.strip()))
        
        print(f"✅ 유효한 번역 쌍: {len(self.translation_pairs)}개")
    
    def __len__(self):
        return len(self.translation_pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.translation_pairs[idx]
        
        # Encoder 입력 (소스 언어)
        encoder_inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Decoder 입력 및 레이블 (타겟 언어)
        # Teacher Forcing: 입력은 [BOS] + 토큰들, 레이블은 토큰들 + [EOS]
        tgt_with_bos = f"[CLS] {tgt_text}"  # BOS 토큰 추가
        tgt_with_eos = f"{tgt_text} [SEP]"  # EOS 토큰 추가
        
        decoder_inputs = self.tokenizer(
            tgt_with_bos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = self.tokenizer(
            tgt_with_eos,
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
            'src_text': src_text,
            'tgt_text': tgt_text,
        }


class EncoderDecoderTrainerHF:
    """HuggingFace 데이터셋을 사용한 Encoder-Decoder 모델 훈련기"""
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset=None,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        device='cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # 데이터 로더
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # macOS 호환성
            pin_memory=True if device != 'cpu' else False
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device != 'cpu' else False
            )
        
        # 옵티마이저 및 스케줄러
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay
        )
        
        total_steps = len(self.train_loader) * 10  # 추정치
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_bleu_scores = []
        self.val_bleu_scores = []
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """간단한 BLEU 점수 계산"""
        try:
            from nltk.translate.bleu_score import corpus_bleu
            from nltk.tokenize import word_tokenize
            
            # 토크나이징
            tokenized_preds = [word_tokenize(pred.lower()) for pred in predictions]
            tokenized_refs = [[word_tokenize(ref.lower())] for ref in references]
            
            # BLEU 점수 계산
            bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
            return bleu_score
        except ImportError:
            # NLTK가 없으면 간단한 단어 일치율 계산
            total_matches = 0
            total_words = 0
            
            for pred, ref in zip(predictions, references):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                matches = len(pred_words & ref_words)
                total_matches += matches
                total_words += len(ref_words)
            
            return total_matches / max(total_words, 1)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        predictions = []
        references = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
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
            
            # Gradient Accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 통계 업데이트
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # BLEU 점수 계산용 예측 수집 (일부만)
            if batch_idx % 20 == 0:  # 20배치마다 BLEU 계산
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=50,
                        do_sample=False
                    )
                    
                    batch_predictions = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    batch_references = batch['tgt_text']
                    
                    predictions.extend(batch_predictions)
                    references.extend(batch_references)
            
            # Gradient 업데이트
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 진행률 표시
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # BLEU 점수 계산
        bleu_score = self.calculate_bleu(predictions, references) if predictions else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'bleu': bleu_score
        }
    
    def validate(self) -> Optional[Dict[str, float]]:
        """검증"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
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
                
                # 번역 생성
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=50,
                    do_sample=False
                )
                
                batch_predictions = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                batch_references = batch['tgt_text']
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        val_loss = total_loss / num_batches
        bleu_score = self.calculate_bleu(predictions, references)
        
        return {
            'loss': val_loss,
            'bleu': bleu_score,
            'sample_predictions': predictions[:5],
            'sample_references': references[:5]
        }
    
    def train(self, num_epochs: int, save_dir: str = "encoder_decoder_checkpoints_hf"):
        """전체 훈련 루프"""
        print(f"🚀 Encoder-Decoder 모델 훈련 시작 (HuggingFace 데이터)")
        print(f"📊 훈련 설정:")
        print(f"   - 에포크: {num_epochs}")
        print(f"   - 배치 크기: {self.train_loader.batch_size}")
        print(f"   - Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   - 실제 배치 크기: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"   - 학습률: {self.optimizer.param_groups[0]['lr']}")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}")
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_bleu = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
            
            # 훈련
            train_metrics = self.train_epoch(epoch + 1)
            self.train_losses.append(train_metrics['loss'])
            self.train_bleu_scores.append(train_metrics['bleu'])
            
            # 검증
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
                self.val_bleu_scores.append(val_metrics['bleu'])
                
                print(f"✅ Train Loss: {train_metrics['loss']:.4f}, BLEU: {train_metrics['bleu']:.4f}")
                print(f"✅ Val Loss: {val_metrics['loss']:.4f}, BLEU: {val_metrics['bleu']:.4f}")
                
                # 샘플 번역 출력
                print(f"\n🔍 번역 샘플:")
                for i, (pred, ref) in enumerate(zip(val_metrics['sample_predictions'][:3], 
                                                   val_metrics['sample_references'][:3])):
                    print(f"   {i+1}. 예측: {pred}")
                    print(f"      정답: {ref}")
                
                # 최고 성능 모델 저장
                if val_metrics['bleu'] > best_val_bleu:
                    best_val_bleu = val_metrics['bleu']
                    self.save_checkpoint(os.path.join(save_dir, f"best_model.pt"), epoch + 1)
                    print(f"💾 최고 성능 모델 저장 (BLEU: {val_metrics['bleu']:.4f})")
            else:
                print(f"✅ Train Loss: {train_metrics['loss']:.4f}, BLEU: {train_metrics['bleu']:.4f}")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"), epoch + 1)
        
        print(f"\n🎉 훈련 완료!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_bleu_scores': self.train_bleu_scores,
            'val_bleu_scores': self.val_bleu_scores
        }
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_bleu_scores': self.train_bleu_scores,
            'val_bleu_scores': self.val_bleu_scores,
            'config': self.model.config,
        }, filepath)
    
    def translate_sample(self, source_text: str, max_length: int = 50) -> str:
        """샘플 번역 생성"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                source_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                do_sample=False
            )
            
            translation = self.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            return translation


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Encoder-Decoder 번역 모델 훈련 (HuggingFace 데이터)")
    
    # 데이터셋 설정
    parser.add_argument('--dataset', type=str, default='opus_books',
                       choices=['opus_books', 'wmt16', 'wmt14', 'opus100', 'kde4'],
                       help='사용할 HuggingFace 번역 데이터셋')
    parser.add_argument('--src_lang', type=str, default='en', help='소스 언어 코드')
    parser.add_argument('--tgt_lang', type=str, default='ko', help='타겟 언어 코드')
    parser.add_argument('--num_samples', type=int, default=10000, help='사용할 샘플 수')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased',
                       help='사용할 토크나이저')
    
    # 모델 설정
    parser.add_argument('--hidden_size', type=int, default=256, help='숨김 차원')
    parser.add_argument('--encoder_layers', type=int, default=4, help='인코더 레이어 수')
    parser.add_argument('--decoder_layers', type=int, default=4, help='디코더 레이어 수')
    parser.add_argument('--attention_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--max_length', type=int, default=128, help='최대 시퀀스 길이')
    
    # 훈련 설정
    parser.add_argument('--epochs', type=int, default=5, help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                       help='그래디언트 누적 스텝')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='가중치 감쇠')
    parser.add_argument('--save_dir', type=str, default='encoder_decoder_checkpoints_hf',
                       help='체크포인트 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"🖥️  사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"📝 토크나이저: {args.tokenizer}")
    
    # 모델 설정
    config = EncoderDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.attention_heads,
        decoder_attention_heads=args.attention_heads,
        max_position_embeddings=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    
    # 모델 생성
    model = BERTEncoderTransformerDecoderModel(config)
    print(f"🤖 Encoder-Decoder 모델 생성 완료")
    print(f"📊 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 데이터셋 로드
    print(f"📚 번역 데이터셋 로드: {args.dataset} ({args.src_lang} → {args.tgt_lang})")
    
    try:
        train_dataset = HuggingFaceTranslationDataset(
            dataset_name=args.dataset,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            num_samples=args.num_samples,
            split="train"
        )
        
        # 검증 데이터셋
        val_dataset = None
        try:
            val_dataset = HuggingFaceTranslationDataset(
                dataset_name=args.dataset,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
                num_samples=min(1000, args.num_samples // 10),
                split="validation"
            )
            print(f"📚 검증 데이터셋 로드 완료: {len(val_dataset)}개")
        except:
            print("⚠️  검증 데이터셋을 찾을 수 없어 검증 없이 진행")
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        print("💡 대안: 다른 언어 쌍이나 데이터셋을 시도해보세요")
        print("   예: --dataset opus100 --src_lang en --tgt_lang fr")
        return
    
    # 훈련기 생성
    trainer = EncoderDecoderTrainerHF(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=device
    )
    
    # 훈련 전 번역 테스트
    print(f"\n🔍 훈련 전 번역 테스트:")
    if args.src_lang == "en" and args.tgt_lang == "ko":
        test_text = "Hello, how are you today?"
    elif args.src_lang == "en" and args.tgt_lang == "fr":
        test_text = "The weather is beautiful today."
    else:
        test_text = "Good morning, how are you?"
    
    translation = trainer.translate_sample(test_text)
    print(f"   원문: {test_text}")
    print(f"   번역: {translation}")
    
    # 훈련 실행
    results = trainer.train(args.epochs, args.save_dir)
    
    # 훈련 후 번역 테스트
    print(f"\n🔍 훈련 후 번역 테스트:")
    translation = trainer.translate_sample(test_text)
    print(f"   원문: {test_text}")
    print(f"   번역: {translation}")
    
    # 다양한 문장 번역 테스트
    if args.src_lang == "en":
        test_sentences = [
            "Good morning!",
            "Thank you very much.",
            "I love this book.",
            "The weather is nice today."
        ]
    else:
        test_sentences = [
            "Hello world",
            "How are you?",
            "Nice to meet you.",
            "Have a good day."
        ]
    
    print(f"\n🧪 다양한 문장 번역 테스트:")
    for sentence in test_sentences:
        translation = trainer.translate_sample(sentence)
        print(f"   {sentence} → {translation}")
    
    # 결과 저장
    results_path = os.path.join(args.save_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 훈련 결과 저장: {results_path}")
    print(f"📁 체크포인트 저장 위치: {args.save_dir}/")


if __name__ == "__main__":
    main()