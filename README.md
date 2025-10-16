# Project Generator Backend (LangGraph)

MSA-EZ Event Storming 자동화를 위한 LangGraph 기반 UserStory 생성 백엔드 서버

## 📋 개요

이 프로젝트는 Firebase Job Queue를 통해 프론트엔드와 통신하며, LangGraph 워크플로우를 사용하여 요구사항으로부터 User Story, Actor, Business Rule을 추출합니다.

### 주요 기능

- ✅ **UserStory Generator**: RAG 기반 User Story 자동 생성
- 🔥 **Firebase Integration**: Job Queue 방식의 비동기 처리
- 🚀 **Auto Scaling**: Kubernetes 환경에서 자동 스케일링
- 📊 **Health Check**: `/ok` 엔드포인트 제공

## 🏗️ 아키텍처

```
Frontend (Vue.js)
    ↓ (Firebase)
    ↓ jobs/user_story_generator/{jobId}
    ↓
Backend (Python/LangGraph)
    ├── DecentralizedJobManager (Job 감시)
    ├── UserStoryWorkflow (LangGraph)
    │   ├── RAG Retriever
    │   ├── LLM (GPT-4o)
    │   └── Output Parser
    └── Firebase System (결과 저장)
```

## 🛠️ 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가:

```bash
# Firebase
FIREBASE_DATABASE_URL=https://YOUR-PROJECT.firebaseio.com
FIREBASE_CREDENTIALS_PATH=./firebase-credentials.json

# OpenAI
OPENAI_API_KEY=sk-...

# Server
PORT=2024
IS_LOCAL_RUN=true

# Job Namespace
NAMESPACE=user_story_generator
```

### 3. Firebase 인증 설정

Firebase 콘솔에서 서비스 계정 키를 다운로드하여 `firebase-credentials.json`으로 저장

### 4. 서버 실행

```bash
# 간편 실행 (권장)
./start.sh

# 또는 수동 실행
source venv/bin/activate
export PYTHONPATH="$(pwd)/src"
python -m project_generator.main
```

서버가 `http://localhost:2024`에서 실행됩니다.

**Health Check:**
```bash
curl http://localhost:2024/ok
```

## 📁 프로젝트 구조

```
backend-generators/
└── src/project_generator/
    ├── main.py                 # 메인 서버 (Flask + Job Manager)
    ├── config.py               # 설정
    ├── workflows/
    │   ├── user_story/
    │   │   └── user_story_generator.py  # UserStory LangGraph 워크플로우
    │   └── common/
    │       └── rag_retriever.py         # RAG Knowledge Base
    ├── systems/
    │   └── firebase_system.py  # Firebase 연동
    ├── utils/
    │   ├── decentralized_job_manager.py  # Job Queue 관리
    │   ├── job_util.py          # Job 유틸리티
    │   ├── logging_util.py      # 로깅
    │   ├── json_util.py         # JSON 처리
    │   └── convert_case_util.py # CamelCase 변환
    └── models/
        └── (Legacy compatibility models)
├── .env                        # 환경 변수
├── .gitignore                  # Git 제외 파일
└── pyproject.toml              # Python 프로젝트 설정
```

## 🔧 주요 컴포넌트

### UserStoryWorkflow

```python
from project_generator.workflows.user_story.user_story_generator import UserStoryWorkflow

workflow = UserStoryWorkflow()
result = workflow.run({
    "jobId": "usgen-...",
    "requirements": "사용자는 ...",
    "bounded_contexts": []
})

# result:
# {
#     "userStories": [...],
#     "actors": [...],
#     "businessRules": [...],
#     "boundedContexts": [...],
#     "isCompleted": True
# }
```

### Firebase Job Queue

**Job 생성 (Frontend)**:
```javascript
// jobs/user_story_generator/{jobId}
await storage.setObject(`jobs/user_story_generator/${jobId}`, {
    state: {
        inputs: {
            jobId,
            requirements,
            bounded_contexts: []
        }
    }
});

// requestedJobs/user_story_generator/{jobId}
await storage.setObject(`requestedJobs/user_story_generator/${jobId}`, {
    createdAt: firebase.database.ServerValue.TIMESTAMP
});
```

**Job 처리 (Backend)**:
```python
# 1. DecentralizedJobManager가 requestedJobs 감시
# 2. 새로운 Job 발견 시 claim
# 3. process_user_story_job() 호출
# 4. UserStoryWorkflow 실행
# 5. 결과를 jobs/.../state/outputs에 저장
# 6. requestedJobs에서 삭제
```

## 🚀 배포

### Docker

```bash
# 이미지 빌드
docker build -t user-story-generator:latest .

# 실행
docker run -p 2024:2024 \
  -e FIREBASE_DATABASE_URL=... \
  -e OPENAI_API_KEY=... \
  user-story-generator:latest
```

### Kubernetes

```bash
# 배포
kubectl apply -f k8s/deployment.yaml

# 스케일링
kubectl scale deployment user-story-generator --replicas=3
```

## 📊 모니터링

### Health Check

```bash
curl http://localhost:2024/ok
```

### 로그

```bash
# 로그 확인
tail -f server.log

# 또는 Docker
docker logs -f <container-id>
```

## 🔍 트러블슈팅

### 1. Firebase 연결 오류

```bash
# Firebase 인증 확인
cat firebase-credentials.json

# 환경 변수 확인
echo $FIREBASE_DATABASE_URL
```

### 2. Job이 처리되지 않음

```bash
# Backend 로그 확인
grep "Job 시작" server.log

# Firebase에서 직접 확인
# https://console.firebase.google.com/
```

### 3. LLM API 오류

```bash
# OpenAI API 키 확인
echo $OPENAI_API_KEY

# Rate Limit 확인
# 로그에서 "rate_limit_exceeded" 검색
```

## 📝 개발 가이드

### 새로운 워크플로우 추가

1. `workflows/` 폴더에 새 디렉토리 생성
2. LangGraph `StateGraph` 정의
3. `main.py`에 처리 함수 추가
4. `decentralized_job_manager.py`에 네임스페이스 추가

### 테스트

```bash
# 단위 테스트
pytest tests/

# 통합 테스트
python -m src.eventstorming_generator.runs.run_user_story_generator
```

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

This project is licensed under the MIT License.

## 🔗 관련 문서

- [LangGraph 아키텍처 문서](../LANGGRAPH_ARCHITECTURE.md)
- [Frontend Integration Guide](../docs/frontend-integration.md)
- [Firebase Setup Guide](../docs/firebase-setup.md)

## 📧 문의

프로젝트에 대한 문의사항은 이슈로 등록해주세요.
