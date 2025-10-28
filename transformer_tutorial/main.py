"""
Transformer Tutorial - 메인 실행 스크립트
"""
import argparse
import sys
import os
import torch

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_pytorch_version(args):
    """PyTorch 순정 버전 실행"""
    print("=" * 60)
    print("PYTORCH 순정 TRANSFORMER 실행")
    print("=" * 60)
    
    # PyTorch 버전 import
    from pytorch_version.train import main as pytorch_main
    
    # Arguments 변환
    pytorch_args = [
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.learning_rate),
        '--d_model', str(args.hidden_size),
        '--save_dir', args.output_dir + '_pytorch',
    ]
    
    if args.debug:
        pytorch_args.append('--debug')
    if args.visualize:
        pytorch_args.append('--visualize')
    
    # Arguments 설정
    sys.argv = ['train.py'] + pytorch_args
    
    try:
        pytorch_main()
        print("\nPyTorch 버전 실행 완료!")
    except Exception as e:
        print(f"PyTorch 버전 실행 중 오류: {e}")


def run_huggingface_version(args):
    """Hugging Face 버전 실행"""
    print("=" * 60)
    print("HUGGING FACE TRANSFORMER 실행")
    print("=" * 60)
    
    # Hugging Face 버전 import
    from huggingface_version.train import main as hf_main
    
    # Arguments 변환
    hf_args = [
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--hidden_size', str(args.hidden_size),
        '--num_layers', str(args.num_layers),
        '--num_heads', str(args.num_heads),
        '--output_dir', args.output_dir + '_huggingface',
        '--vocab_size', str(args.vocab_size),
        '--train_size', str(args.train_size),
        '--eval_size', str(args.eval_size),
    ]
    
    if args.track_internals:
        hf_args.append('--track_internals')
    if args.wandb:
        hf_args.append('--wandb')
        hf_args.extend(['--wandb_project', args.wandb_project])
    
    # Arguments 설정
    sys.argv = ['train.py'] + hf_args
    
    try:
        hf_main()
        print("\nHugging Face 버전 실행 완료!")
    except Exception as e:
        print(f"Hugging Face 버전 실행 중 오류: {e}")


def run_analysis_demo():
    """분석 도구 데모 실행"""
    print("=" * 60)
    print("TRANSFORMER 분석 도구 데모")
    print("=" * 60)
    
    try:
        from utils.analysis import analyze_sample_model
        from utils.visualization import create_sample_visualizations
        
        print("\n1. 분석 도구 데모 실행...")
        analyze_sample_model()
        
        print("\n2. 시각화 도구 데모 실행...")
        create_sample_visualizations()
        
        print("\n분석 도구 데모 완료!")
        
    except Exception as e:
        print(f"분석 도구 데모 실행 중 오류: {e}")


def run_comparison_test():
    """두 버전 비교 테스트"""
    print("=" * 60)
    print("PYTORCH vs HUGGING FACE 비교 테스트")
    print("=" * 60)
    
    try:
        # PyTorch 버전 테스트
        print("\n1. PyTorch 버전 테스트...")
        from pytorch_version.model import test_transformer
        test_transformer()
        
        # Hugging Face 버전 테스트
        print("\n2. Hugging Face 버전 테스트...")
        from huggingface_version.model import test_tracked_transformer
        test_tracked_transformer()
        
        print("\n비교 테스트 완료!")
        
    except Exception as e:
        print(f"비교 테스트 실행 중 오류: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Tutorial - Attention is All You Need 구현 및 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  # PyTorch 순정 버전 실행
  python main.py --version pytorch --epochs 10 --debug --visualize
  
  # Hugging Face 버전 실행  
  python main.py --version huggingface --epochs 5 --track_internals --wandb
  
  # 두 버전 모두 실행
  python main.py --version both --epochs 3
  
  # 분석 도구 데모
  python main.py --version analysis
  
  # 비교 테스트
  python main.py --version test
        """
    )
    
    # 실행 버전 선택
    parser.add_argument(
        '--version', 
        choices=['pytorch', 'huggingface', 'both', 'analysis', 'test'],
        default='both',
        help='실행할 버전 선택 (default: both)'
    )
    
    # 모델 설정
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train_size', type=int, default=2000, help='Training dataset size')
    parser.add_argument('--eval_size', type=int, default=500, help='Evaluation dataset size')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    # PyTorch 버전 전용 옵션
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (PyTorch)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization (PyTorch)')
    
    # Hugging Face 버전 전용 옵션
    parser.add_argument('--track_internals', action='store_true', help='Track internal states (HuggingFace)')
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging (HuggingFace)')
    parser.add_argument('--wandb_project', type=str, default='transformer-tutorial', help='WandB project name')
    
    args = parser.parse_args()
    
    # 디바이스 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 가능한 디바이스: {device}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 설정 출력
    print("\n실행 설정:")
    print(f"  버전: {args.version}")
    print(f"  에폭: {args.epochs}")
    print(f"  배치 크기: {args.batch_size}")
    print(f"  학습률: {args.learning_rate}")
    print(f"  모델 크기: {args.hidden_size}")
    print(f"  출력 디렉토리: {args.output_dir}")
    
    if args.version == 'pytorch':
        run_pytorch_version(args)
        
    elif args.version == 'huggingface':
        run_huggingface_version(args)
        
    elif args.version == 'both':
        print("\n두 버전 모두 실행합니다...\n")
        run_pytorch_version(args)
        print("\n" + "="*60 + "\n")
        run_huggingface_version(args)
        
    elif args.version == 'analysis':
        run_analysis_demo()
        
    elif args.version == 'test':
        run_comparison_test()
    
    print("\n" + "="*60)
    print("TRANSFORMER TUTORIAL 완료!")
    print("="*60)
    print("\n주요 결과 파일:")
    print(f"  - 모델 체크포인트: {args.output_dir}_*/")
    print(f"  - 시각화 결과: {args.output_dir}_*/visualizations/")
    print(f"  - 분석 결과: {args.output_dir}_*/analysis/")
    
    print("\n다음 단계:")
    print("  1. 시각화 결과를 확인하여 attention 패턴 분석")
    print("  2. QKV 값 변화 그래프로 학습 과정 이해")
    print("  3. 두 버전의 구현 차이점 비교")
    print("  4. 실제 데이터셋으로 확장 실험")


if __name__ == "__main__":
    main()