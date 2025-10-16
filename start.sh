#!/bin/bash
# Project Generator Backend 서버 시작 스크립트

cd "$(dirname "$0")"

# 가상환경 확인
if [ ! -d "venv" ]; then
    echo "❌ 가상환경이 없습니다. 먼저 'python3 -m venv venv'를 실행하세요."
    exit 1
fi

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다. .env.example을 참고하여 .env를 생성하세요."
    exit 1
fi

echo "🚀 Project Generator 서버를 시작합니다..."
echo "📍 포트: 2024"
echo "📊 Health Check: http://localhost:2024/ok"
echo ""

# 가상환경 활성화 및 서버 시작
source venv/bin/activate
export PYTHONPATH="$(pwd)/src"
python -m project_generator.main

