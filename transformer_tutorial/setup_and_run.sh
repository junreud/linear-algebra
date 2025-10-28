#!/bin/bash

# Transformer Tutorial 자동 설치 및 실행 스크립트

set -e  # 에러 발생시 스크립트 중단

echo "🚀 Transformer Tutorial 자동 설치 및 실행 스크립트"
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 현재 디렉토리 확인
CURRENT_DIR=$(pwd)
PROJECT_DIR="/Users/gimjunseog/projects/linear-algebra/transformer_tutorial"

echo "현재 디렉토리: $CURRENT_DIR"

# 프로젝트 디렉토리로 이동
if [ "$CURRENT_DIR" != "$PROJECT_DIR" ]; then
    echo "프로젝트 디렉토리로 이동: $PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Python 버전 확인
echo -e "\n${BLUE}1. Python 버전 확인${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 버전: $PYTHON_VERSION"

if ! python3 -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
    echo -e "${RED}❌ Python 3.8 이상이 필요합니다.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 버전 OK${NC}"

# 가상환경 생성 확인
echo -e "\n${BLUE}2. 가상환경 설정${NC}"
if [ ! -d "transformer_env" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv transformer_env
    echo -e "${GREEN}✅ 가상환경 생성 완료${NC}"
else
    echo -e "${YELLOW}⚠️  가상환경이 이미 존재합니다${NC}"
fi

# 가상환경 활성화
echo "가상환경 활성화 중..."
source transformer_env/bin/activate

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ 가상환경 활성화 완료: $(basename $VIRTUAL_ENV)${NC}"
else
    echo -e "${RED}❌ 가상환경 활성화 실패${NC}"
    exit 1
fi

# pip 업그레이드
echo -e "\n${BLUE}3. pip 업그레이드${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip 업그레이드 완료${NC}"

# 패키지 설치 확인
echo -e "\n${BLUE}4. 패키지 설치 확인${NC}"

# torch 설치 확인
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✅ PyTorch 이미 설치됨: $TORCH_VERSION${NC}"
else
    echo "PyTorch 설치 중..."
    pip install torch>=2.0.0 torchvision>=0.15.0 --quiet
    echo -e "${GREEN}✅ PyTorch 설치 완료${NC}"
fi

# requirements.txt 설치
echo "필수 패키지 설치 중..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ 패키지 설치 완료${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt 파일이 없습니다. 개별 패키지 설치 중...${NC}"
    pip install transformers>=4.30.0 numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0 pandas>=2.0.0 tqdm>=4.65.0 --quiet
fi

# 설치 확인
echo -e "\n${BLUE}5. 설치 확인${NC}"
python -c "
import torch
import transformers
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경 대응
import matplotlib.pyplot as plt
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ CUDA 사용 가능: {torch.cuda.is_available()}')
print(f'✅ 모든 패키지 정상 설치됨')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 모든 패키지 정상 설치 확인${NC}"
else
    echo -e "${RED}❌ 패키지 설치 확인 실패${NC}"
    exit 1
fi

# 실행 옵션 선택
echo -e "\n${BLUE}6. 실행 옵션 선택${NC}"
echo "다음 중 하나를 선택하세요:"
echo "1) 빠른 테스트 (1-2분)"
echo "2) 기본 실행 (5-10분) - 추천"
echo "3) PyTorch 버전 상세 분석 (10-15분)"
echo "4) Hugging Face 버전 실무 실행 (10-15분)"
echo "5) 분석 도구 데모만 실행 (1분)"
echo "6) 설치만 하고 종료"

read -p "선택하세요 (1-6): " choice

case $choice in
    1)
        echo -e "\n${YELLOW}빠른 테스트 실행 중...${NC}"
        python main.py --version test
        ;;
    2)
        echo -e "\n${YELLOW}기본 실행 중... (두 버전 모두 간단히 실행)${NC}"
        python main.py --version both --epochs 2 --train_size 500 --eval_size 100
        ;;
    3)
        echo -e "\n${YELLOW}PyTorch 버전 상세 분석 실행 중...${NC}"
        python main.py --version pytorch --epochs 3 --debug --visualize --batch_size 8 --train_size 1000 --eval_size 200
        ;;
    4)
        echo -e "\n${YELLOW}Hugging Face 버전 실무 실행 중...${NC}"
        python main.py --version huggingface --epochs 5 --track_internals --batch_size 16 --train_size 2000 --eval_size 500
        ;;
    5)
        echo -e "\n${YELLOW}분석 도구 데모 실행 중...${NC}"
        python main.py --version analysis
        ;;
    6)
        echo -e "\n${GREEN}✅ 설치 완료! 수동으로 실행하세요:${NC}"
        echo "  source transformer_env/bin/activate"
        echo "  python main.py --version both --epochs 2 --train_size 500"
        exit 0
        ;;
    *)
        echo -e "${RED}잘못된 선택입니다. 기본 실행을 진행합니다.${NC}"
        python main.py --version both --epochs 2 --train_size 500 --eval_size 100
        ;;
esac

# 실행 완료 메시지
echo -e "\n${GREEN}🎉 실행 완료!${NC}"
echo -e "\n${BLUE}생성된 결과 파일들:${NC}"
ls -la results_* 2>/dev/null || echo "결과 폴더가 아직 생성되지 않았습니다."

echo -e "\n${BLUE}다음 단계:${NC}"
echo "1. 생성된 시각화 파일들을 확인하세요:"
echo "   open results_*/visualizations/*.png"
echo ""
echo "2. 코드를 자세히 읽어보세요:"
echo "   - pytorch_version/: 순정 구현"
echo "   - huggingface_version/: 실무 구현"
echo ""
echo "3. 더 자세한 실험을 해보세요:"
echo "   python main.py --version both --epochs 10 --debug --track_internals"

echo -e "\n${YELLOW}가상환경을 계속 사용하려면:${NC}"
echo "  source transformer_env/bin/activate"
echo ""
echo -e "${YELLOW}가상환경을 종료하려면:${NC}"
echo "  deactivate"

echo -e "\n${GREEN}Happy Learning! 🚀${NC}"