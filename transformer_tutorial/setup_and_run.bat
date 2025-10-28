@echo off
REM Transformer Tutorial 자동 설치 및 실행 스크립트 (Windows)

echo 🚀 Transformer Tutorial 자동 설치 및 실행 스크립트
echo ==================================================

REM 현재 디렉토리 확인
echo 현재 디렉토리: %CD%

REM Python 버전 확인
echo.
echo 1. Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았거나 PATH에 등록되지 않았습니다.
    echo Python 3.8 이상을 설치하고 PATH에 등록해주세요.
    pause
    exit /b 1
)

python --version
echo ✅ Python 설치 확인

REM 가상환경 생성
echo.
echo 2. 가상환경 설정
if not exist "transformer_env" (
    echo 가상환경 생성 중...
    python -m venv transformer_env
    echo ✅ 가상환경 생성 완료
) else (
    echo ⚠️ 가상환경이 이미 존재합니다
)

REM 가상환경 활성화
echo 가상환경 활성화 중...
call transformer_env\Scripts\activate.bat

REM pip 업그레이드
echo.
echo 3. pip 업그레이드
python -m pip install --upgrade pip --quiet
echo ✅ pip 업그레이드 완료

REM 패키지 설치
echo.
echo 4. 패키지 설치
echo 필수 패키지 설치 중...
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    echo ✅ 패키지 설치 완료
) else (
    echo ⚠️ requirements.txt 파일이 없습니다. 개별 패키지 설치 중...
    pip install torch>=2.0.0 torchvision>=0.15.0 --quiet
    pip install transformers>=4.30.0 numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0 pandas>=2.0.0 tqdm>=4.65.0 --quiet
)

REM 설치 확인
echo.
echo 5. 설치 확인
python -c "import torch; import transformers; import numpy as np; import matplotlib.pyplot as plt; print('✅ 모든 패키지 정상 설치됨'); print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"

if errorlevel 1 (
    echo ❌ 패키지 설치 확인 실패
    pause
    exit /b 1
)

REM 실행 옵션 선택
echo.
echo 6. 실행 옵션 선택
echo 다음 중 하나를 선택하세요:
echo 1) 빠른 테스트 (1-2분)
echo 2) 기본 실행 (5-10분) - 추천
echo 3) PyTorch 버전 상세 분석 (10-15분)
echo 4) Hugging Face 버전 실무 실행 (10-15분)
echo 5) 분석 도구 데모만 실행 (1분)
echo 6) 설치만 하고 종료

set /p choice="선택하세요 (1-6): "

if "%choice%"=="1" (
    echo 빠른 테스트 실행 중...
    python main.py --version test
) else if "%choice%"=="2" (
    echo 기본 실행 중... (두 버전 모두 간단히 실행)
    python main.py --version both --epochs 2 --train_size 500 --eval_size 100
) else if "%choice%"=="3" (
    echo PyTorch 버전 상세 분석 실행 중...
    python main.py --version pytorch --epochs 3 --debug --visualize --batch_size 8 --train_size 1000 --eval_size 200
) else if "%choice%"=="4" (
    echo Hugging Face 버전 실무 실행 중...
    python main.py --version huggingface --epochs 5 --track_internals --batch_size 16 --train_size 2000 --eval_size 500
) else if "%choice%"=="5" (
    echo 분석 도구 데모 실행 중...
    python main.py --version analysis
) else if "%choice%"=="6" (
    echo ✅ 설치 완료! 수동으로 실행하세요:
    echo   transformer_env\Scripts\activate
    echo   python main.py --version both --epochs 2 --train_size 500
    goto end
) else (
    echo 잘못된 선택입니다. 기본 실행을 진행합니다.
    python main.py --version both --epochs 2 --train_size 500 --eval_size 100
)

REM 실행 완료 메시지
echo.
echo 🎉 실행 완료!
echo.
echo 생성된 결과 파일들:
dir results_* 2>nul || echo 결과 폴더가 아직 생성되지 않았습니다.

echo.
echo 다음 단계:
echo 1. 생성된 시각화 파일들을 확인하세요
echo 2. 코드를 자세히 읽어보세요:
echo    - pytorch_version\: 순정 구현
echo    - huggingface_version\: 실무 구현
echo 3. 더 자세한 실험을 해보세요:
echo    python main.py --version both --epochs 10 --debug --track_internals

echo.
echo 가상환경을 계속 사용하려면:
echo   transformer_env\Scripts\activate
echo.
echo 가상환경을 종료하려면:
echo   deactivate

:end
echo.
echo Happy Learning! 🚀
pause