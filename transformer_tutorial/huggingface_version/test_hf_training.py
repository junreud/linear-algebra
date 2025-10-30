"""
Encoder-Decoder HuggingFace 데이터셋 훈련 테스트

실제 번역 데이터셋으로 모델을 훈련하는 예제입니다.
"""

import sys
import os

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_encoder_decoder_hf import main, HuggingFaceTranslationDataset
import torch
from transformers import AutoTokenizer


def test_dataset_loading():
    """데이터셋 로딩 테스트"""
    print("🧪 HuggingFace 번역 데이터셋 테스트")
    
    try:
        # 영어-한국어 도서 번역 데이터
        print("\n1. OPUS Books (EN-KO) 테스트...")
        dataset = HuggingFaceTranslationDataset(
            dataset_name="opus_books",
            src_lang="en",
            tgt_lang="ko",
            num_samples=100,
            split="train"
        )
        print(f"✅ OPUS Books 로드 성공: {len(dataset)}개 샘플")
        
        # 샘플 확인
        sample = dataset[0]
        print(f"📝 샘플 데이터:")
        print(f"   소스: {sample['src_text']}")
        print(f"   타겟: {sample['tgt_text']}")
        print(f"   Input IDs 형태: {sample['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ OPUS Books 실패: {e}")
        
        # 대안: 영어-프랑스어
        try:
            print("\n대안: OPUS Books (EN-FR) 테스트...")
            dataset = HuggingFaceTranslationDataset(
                dataset_name="opus_books",
                src_lang="en",
                tgt_lang="fr",
                num_samples=100,
                split="train"
            )
            print(f"✅ OPUS Books (EN-FR) 로드 성공: {len(dataset)}개 샘플")
            
            sample = dataset[0]
            print(f"📝 샘플 데이터:")
            print(f"   소스: {sample['src_text']}")
            print(f"   타겟: {sample['tgt_text']}")
            
        except Exception as e2:
            print(f"❌ OPUS Books (EN-FR) 실패: {e2}")
            
            # 다른 데이터셋 시도
            try:
                print("\n대안: OPUS-100 (EN-FR) 테스트...")
                dataset = HuggingFaceTranslationDataset(
                    dataset_name="opus100",
                    src_lang="en",
                    tgt_lang="fr",
                    num_samples=100,
                    split="train"
                )
                print(f"✅ OPUS-100 로드 성공: {len(dataset)}개 샘플")
                
            except Exception as e3:
                print(f"❌ 모든 데이터셋 로드 실패")
                print(f"💡 인터넷 연결을 확인하거나 다른 언어 쌍을 시도해보세요")


def test_quick_training():
    """빠른 훈련 테스트 (작은 데이터셋)"""
    print("\n🚀 빠른 훈련 테스트")
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"🖥️  디바이스: {device}")
    
    # 가상으로 명령줄 인수 설정
    import argparse
    import sys
    
    # 기존 argv 백업
    original_argv = sys.argv
    
    try:
        # 테스트용 명령줄 인수
        sys.argv = [
            'train_encoder_decoder_hf.py',
            '--dataset', 'opus_books',
            '--src_lang', 'en',
            '--tgt_lang', 'fr',  # 한국어보다 프랑스어가 더 많은 데이터 보유
            '--num_samples', '200',  # 매우 작은 데이터셋
            '--epochs', '1',
            '--batch_size', '2',
            '--gradient_accumulation_steps', '2',
            '--hidden_size', '128',  # 작은 모델
            '--encoder_layers', '2',
            '--decoder_layers', '2',
            '--attention_heads', '4',
            '--save_dir', 'test_checkpoints_hf'
        ]
        
        print("📋 테스트 설정:")
        print("   - 데이터셋: OPUS Books (EN-FR)")
        print("   - 샘플 수: 200개")
        print("   - 에포크: 1")
        print("   - 모델 크기: 작음 (128 hidden)")
        
        # 메인 함수 실행
        main()
        
    except Exception as e:
        print(f"❌ 훈련 테스트 실패: {e}")
        print(f"💡 이는 정상일 수 있습니다 (데이터셋 접근 제한 등)")
        
    finally:
        # argv 복원
        sys.argv = original_argv


def usage_examples():
    """사용 예시"""
    print("\n📚 사용 예시:")
    print("\n1. 영어-한국어 번역 (OPUS Books):")
    print("   python train_encoder_decoder_hf.py \\")
    print("       --dataset opus_books \\")
    print("       --src_lang en --tgt_lang ko \\")
    print("       --epochs 5 --num_samples 5000")
    
    print("\n2. 영어-프랑스어 번역 (WMT16):")
    print("   python train_encoder_decoder_hf.py \\")
    print("       --dataset wmt16 \\")
    print("       --src_lang en --tgt_lang fr \\")
    print("       --epochs 10 --batch_size 8")
    
    print("\n3. 다국어 번역 (OPUS-100):")
    print("   python train_encoder_decoder_hf.py \\")
    print("       --dataset opus100 \\")
    print("       --src_lang en --tgt_lang de \\")
    print("       --hidden_size 512 --encoder_layers 6")
    
    print("\n4. 작은 모델 빠른 테스트:")
    print("   python train_encoder_decoder_hf.py \\")
    print("       --dataset kde4 \\")
    print("       --src_lang en --tgt_lang fr \\")
    print("       --num_samples 1000 --epochs 2 \\")
    print("       --hidden_size 128 --encoder_layers 2")


if __name__ == "__main__":
    print("🧪 Encoder-Decoder HuggingFace 데이터셋 훈련 테스트\n")
    
    # 데이터셋 로딩 테스트
    test_dataset_loading()
    
    # 사용 예시
    usage_examples()
    
    # 빠른 훈련 테스트 (선택사항)
    print(f"\n❓ 빠른 훈련 테스트를 실행하시겠습니까? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes', '예']:
        test_quick_training()
    else:
        print("✅ 테스트 완료!")
        print("💡 실제 훈련을 위해서는 train_encoder_decoder_hf.py를 직접 실행하세요.")