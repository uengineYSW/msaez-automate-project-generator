FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY pyproject.toml ./
COPY uv.lock ./

# 애플리케이션 코드 복사 (uv sync 이전에 필요)
COPY src/ ./src/

# uv 설치 및 의존성 설치
RUN pip install uv
RUN uv sync --frozen

# Python 경로 설정
ENV PYTHONPATH=/app/src

# 포트 노출 (헬스체크용)
EXPOSE 2024

# 시간 맞추기
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 애플리케이션 실행
CMD ["uv", "run", "python", "-m", "project_generator.main"] 