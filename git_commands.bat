@echo off
echo Git 저장소 설정 및 푸시 스크립트
echo.

echo 1. Git 저장소 초기화...
git init

echo 2. 원격 저장소 추가...
git remote add origin https://github.com/MTM3MDM/-22.git

echo 3. 파일 추가...
git add .

echo 4. 커밋 생성...
git commit -m "Initial commit: 루시아 디스코드 봇 - 고급 자연어 AI 어시스턴트"

echo 5. GitHub에 푸시...
git push -u origin main

echo.
echo 완료! 저장소에 업로드되었습니다.
pause