# 표준 변환 시스템 가이드

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [주요 컴포넌트](#주요-컴포넌트)
4. [데이터 흐름](#데이터-흐름)
5. [주요 파일 및 코드](#주요-파일-및-코드)
6. [스크립트 사용법](#스크립트-사용법)
7. [설정 및 환경 변수](#설정-및-환경-변수)
8. [트러블슈팅](#트러블슈팅)

---

## 시스템 개요

표준 변환 시스템은 생성된 Aggregate 초안을 회사 표준 문서에 맞게 자동으로 변환하는 RAG(Retrieval-Augmented Generation) 기반 시스템입니다.

### 주요 기능

- **표준 문서 인덱싱**: PPT, 엑셀, 텍스트 파일을 구조화하여 Vector Store에 저장
- **유사도 검색**: 생성된 초안의 이름들을 기반으로 관련 표준 문서 검색
- **자동 변환**: 검색된 표준을 기반으로 LLM이 초안을 표준에 맞게 변환

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vue.js)                        │
│  - Aggregate 초안 생성                                        │
│  - "표준 적용" 버튼 클릭                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ Firebase Job Queue
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (Python/LangGraph)                     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  StandardTransformer                                │   │
│  │  1. 이름 추출 (Aggregate, Enum, ValueObject)        │   │
│  │  2. 쿼리 생성 (182개 쿼리)                          │   │
│  │  3. RAG 검색 (Vector Store)                         │   │
│  │  4. LLM 변환                                        │   │
│  └────────────────────────────────────────────────────┘   │
│                       │                                     │
│                       ▼                                     │
│  ┌────────────────────────────────────────────────────┐   │
│  │  RAGRetriever                                       │   │
│  │  - Vector Store 검색                                │   │
│  │  - 유사도 기반 표준 문서 검색                        │   │
│  └────────────────────────────────────────────────────┘   │
│                       │                                     │
│                       ▼                                     │
│  ┌────────────────────────────────────────────────────┐   │
│  │  ChromaDB (Vector Store)                            │   │
│  │  - 임베딩 벡터 저장                                  │   │
│  │  - 유사도 검색                                      │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                       ▲
                       │
┌─────────────────────────────────────────────────────────────┐
│              인덱싱 시스템                                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  StandardLoader                                     │   │
│  │  - 엑셀/PPT/텍스트 파싱                             │   │
│  │  - 구조화된 텍스트 변환                             │   │
│  └────────────────────────────────────────────────────┘   │
│                       │                                     │
│                       ▼                                     │
│  ┌────────────────────────────────────────────────────┐   │
│  │  StandardIndexer                                    │   │
│  │  - OpenAI Embeddings 생성                           │   │
│  │  - ChromaDB에 저장                                  │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 주요 컴포넌트

### 1. StandardLoader
**위치**: `backend-generators/src/project_generator/workflows/common/standard_loader.py`

표준 문서를 파싱하고 구조화된 텍스트로 변환합니다.

**주요 메서드**:
- `load_standards()`: 표준 문서 디렉토리에서 모든 파일 로드
- `_load_excel()`: 엑셀 파일 파싱 (✅ 구조화된 데이터 지원)
- `_load_ppt()`: PPT 파일 파싱 (⚠️ 구조화된 데이터 미지원, 개선 예정)
- `_load_text()`: 텍스트 파일 파싱
- `_format_excel_row_as_standard_text()`: 엑셀 행을 구조화된 텍스트로 변환
- `_find_column_by_keywords()`: 컬럼명 패턴 매칭으로 값 추출

**컬럼명 패턴 매칭 방식**:
시트별로 컬럼명이 다를 수 있으므로, 키워드 패턴 매칭을 사용하여 유연하게 처리합니다.

- **첫 행을 헤더로 사용**: pandas의 기본 동작으로 첫 행이 컬럼명이 됩니다
- **하드코딩 없음**: 특정 컬럼명을 하드코딩하지 않고, 키워드 패턴 매칭 사용
- **키워드 기반 인식**:
  - **테이블명**: `['table', '테이블']` → "테이블명", "table_name", "테이블", "테이블_이름" 모두 인식
  - **컬럼명**: `['column', '컬럼', 'field', '필드']` → "컬럼명", "column_name", "필드" 모두 인식
  - **설명**: `['description', '설명', 'desc', '내용']` → "설명", "description", "내용" 모두 인식
  - **엔티티명**: `['entity', '엔티티', 'aggregate', '어그리거트', 'name', '이름', '논리명']` → 다양한 형태 인식

**모든 컬럼 포함**:
- 패턴 매칭으로 찾은 컬럼은 표준 키로 저장 (예: `table_name`, `column_name`)
- 나머지 모든 컬럼도 원본 컬럼명 그대로 `structured_data`에 포함
- 예: `타입`, `길이/정밀도`, `키구분`, `Nullable` 등 모든 컬럼 정보 저장

**변환 예시**:

**예시 1: DB 컬럼 표준 시트**
```python
# 원본 엑셀 행 (컬럼명: ['테이블명', '컬럼명', '타입', '길이/정밀도', '키구분', 'Nullable', '설명', '예시값'])
{
  "테이블명": "T_ODR_M",
  "컬럼명": "ODR_ID",
  "타입": "VARCHAR2",
  "길이/정밀도": 50,
  "키구분": "PK",
  "Nullable": "N",
  "설명": "주문 식별자",
  "예시값": "ODR202511250001"
}

# 변환된 텍스트 (임베딩용)
"T_ODR_M table ODR_ID column standard: 주문 식별자."

# 메타데이터 (JSON, 파싱용) - 모든 컬럼 포함
{
  "table_name": "T_ODR_M",        // 패턴 매칭으로 찾은 컬럼 (표준 키)
  "column_name": "ODR_ID",        // 패턴 매칭으로 찾은 컬럼 (표준 키)
  "description": "주문 식별자",    // 패턴 매칭으로 찾은 컬럼 (표준 키)
  "예시값": "ODR202511250001",     // 원본 컬럼명 그대로
  "타입": "VARCHAR2",             // 원본 컬럼명 그대로
  "길이/정밀도": 50,               // 원본 컬럼명 그대로
  "키구분": "PK",                  // 원본 컬럼명 그대로
  "Nullable": "N"                 // 원본 컬럼명 그대로
}
```

**예시 2: DB 테이블 표준 시트**
```python
# 원본 엑셀 행 (컬럼명: ['도메인', '테이블명', '논리명', '설명'])
{
  "도메인": "Transaction",
  "테이블명": "T_ODR_M",
  "논리명": "Order",
  "설명": "주문 단위 기본 정보를 보관"
}

# 변환된 텍스트 (임베딩용)
"Order entity standard: 주문 단위 기본 정보를 보관. T_ODR_M table standard: 주문 단위 기본 정보를 보관."

# 메타데이터 (JSON, 파싱용)
{
  "entity_name": "Order",
  "table_name": "T_ODR_M",
  "domain": "Transaction",
  "description": "주문 단위 기본 정보를 보관"
}
```

### 2. StandardIndexer
**위치**: `backend-generators/src/project_generator/workflows/common/standard_indexer.py`

표준 문서를 Vector Store에 인덱싱합니다.

**주요 메서드**:
- `index_standards()`: 표준 문서를 Vector Store에 인덱싱
- `get_indexed_count()`: 인덱싱된 문서 수 반환

**인덱싱 과정**:
1. `StandardLoader`로 문서 로드
2. 구조화된 텍스트로 변환
3. OpenAI Embeddings API로 벡터 생성
4. ChromaDB에 저장

### 3. RAGRetriever
**위치**: `backend-generators/src/project_generator/workflows/common/rag_retriever.py`

Vector Store에서 유사도 기반 검색을 수행합니다.

**주요 메서드**:
- `search_company_standards()`: 모든 회사 표준 검색
- `search_api_standards()`: API 표준 검색
- `search_terminology_standards()`: 용어 표준 검색

**검색 과정**:
1. 쿼리 텍스트를 임베딩으로 변환
2. ChromaDB에서 유사도 검색
3. 관련 문서 반환

### 4. StandardRAGService
**위치**: `backend-generators/src/project_generator/workflows/common/standard_rag_service.py`

카테고리별 표준 검색을 제공하는 서비스입니다.

**주요 메서드**:
- `search_table_name_standards()`: 테이블명 표준 검색 (type: database_standard, category: table_name)
- `search_column_name_standards()`: 컬럼명 표준 검색 (type: database_standard, category: column_name)
- `search_api_path_standards()`: API 경로 표준 검색 (type: api_standard, category: api_path)
- `search_terminology_standards()`: 용어 표준 검색 (type: terminology_standard, category: terminology)

**특징**:
- 메타데이터 필터링으로 검색 영역을 먼저 좁힌 후 유사도 검색
- ChromaDB의 `$and` 필터 형식 사용
- 유사도 임계값 기본값: 0.3

### 5. AggregateDraftStandardTransformer
**위치**: `backend-generators/src/project_generator/workflows/aggregate_draft/standard_transformer.py`

표준 변환의 핵심 로직을 담당합니다.

**주요 메서드**:
- `transform()`: 표준 변환 메인 함수
- `_extract_names_from_draft()`: Aggregate 초안에서 이름 추출
- `_build_standard_queries()`: 카테고리별 StandardQuery 생성 (짧은 키워드만 사용)
- `_retrieve_relevant_standards_with_categories()`: 카테고리별 RAG 검색 수행
- `_build_global_standard_mapping_context()`: StandardMappingContext 생성 (Terminology/Standard Mapping 레이어)
- `_apply_standard_mappings()`: Deterministic 룰 적용 (선행 치환)
- `_transform_with_llm()`: LLM을 사용한 변환
- `_transform_structure_with_chunking()`: 대용량 데이터 청킹 처리
- `_transform_fields_only_with_llm()`: 필드 전용 변환 (청킹)
- `_transform_enums_vos_only_with_llm()`: Enum/VO 전용 변환 (청킹)

**변환 과정**:
1. **이름 추출**: Aggregate, Enum, ValueObject 이름 추출
2. **쿼리 생성**: 각 이름에 대해 카테고리별 StandardQuery 생성
   - 짧은 키워드만 사용 (예: "Order", "Coupon")
   - category는 메타데이터 필터로 사용 (쿼리 문자열에 포함하지 않음)
   - table_name, terminology, api_path (조건부) 카테고리 검색
3. **RAG 검색**: 카테고리별로 관련 표준 검색
4. **StandardMappingContext 생성**: 검색된 표준 JSON에서 매핑 사전 추출
   - Vector Store 인덱싱 (사용자별 세션 관리)
   - 검색 결과가 없을 경우 LLM에 명시적 지시
5. **Deterministic 룰 적용**: 명확한 매핑은 코드 레벨에서 선행 치환
   - 예: "주문 마스터" → "T_ODR_M"
6. **LLM 변환**: 나머지는 LLM이 처리
   - **청킹 메커니즘**: 대용량 데이터(필드, Enum, VO)를 청크로 분할하여 처리
   - **refs 보존**: 변환 시 refs는 제거했다가 완료 후 원본에서 복원 (alias 기반 매칭)
   - **상세 진행 상황**: BC > Agg > Property Type > Chunk 단위로 진행 상황 업데이트

---

## 데이터 흐름

### 1. 인덱싱 흐름

```
표준 문서 (엑셀/PPT/텍스트)
    ↓
StandardLoader.load_standards()
    ↓
구조화된 텍스트 변환
    ↓
OpenAI Embeddings API
    ↓
ChromaDB 저장
```

### 2. 변환 흐름

```
Aggregate 초안
    ↓
이름 추출 (30개)
    ↓
카테고리별 쿼리 생성 (StandardQuery)
    - 짧은 키워드만 사용 (예: "Order", "Coupon")
    - category는 메타데이터 필터로 사용
    ↓
카테고리별 RAG 검색 (StandardRAGService)
    - type, category로 검색 영역 먼저 좁힘
    - 그 안에서만 유사도 검색
    ↓
관련 표준 문서 검색 (relevant_standards)
    ↓
StandardMappingContext 생성
    - 검색된 표준 JSON 파싱
    - 매핑 사전 추출 (entity_to_table, name_to_domain 등)
    ↓
Deterministic 룰 적용 (선행 치환)
    - "주문 마스터" → "T_ODR_M" (코드 레벨)
    - "Order" → "T_ODR_M" (대소문자 변형 포함)
    ↓
LLM 변환 프롬프트 구성
    - 나머지 변환은 LLM이 처리
    ↓
OpenAI API 호출
    ↓
변환된 초안 반환
```

---

## 주요 파일 및 코드

### Backend 파일

#### 1. 표준 로더
**파일**: `backend-generators/src/project_generator/workflows/common/standard_loader.py`

```python
class StandardLoader:
    def load_standards(self, standards_path: Optional[Path] = None) -> List[Document]:
        """표준 문서 디렉토리에서 모든 파일을 로드하고 청킹"""
        
    def _format_excel_row_as_standard_text(self, row: pd.Series, context: str = "") -> tuple[str, Dict]:
        """엑셀 행을 구조화된 표준 텍스트와 JSON으로 변환"""
        # 반환: (텍스트, 구조화된_데이터)
```

#### 2. 표준 인덱서
**파일**: `backend-generators/src/project_generator/workflows/common/standard_indexer.py`

```python
class StandardIndexer:
    def index_standards(self, standards_path: Optional[Path] = None, 
                       force_reindex: bool = False) -> bool:
        """표준 문서를 Vector Store에 인덱싱"""
```

#### 3. RAG 검색기
**파일**: `backend-generators/src/project_generator/workflows/common/rag_retriever.py`

```python
class RAGRetriever:
    def search_company_standards(self, query: str, k: int = 5) -> List[Dict]:
        """회사 표준 검색 (데이터베이스, API, 용어 등 모든 표준)"""
        
    def search_api_standards(self, query: str, k: int = 5) -> List[Dict]:
        """API 표준 검색"""
        
    def search_terminology_standards(self, query: str, k: int = 5) -> List[Dict]:
        """용어 표준 검색"""
```

#### 4. 표준 RAG 서비스
**파일**: `backend-generators/src/project_generator/workflows/common/standard_rag_service.py`

```python
@dataclass
class StandardRAGService:
    """카테고리별 표준 검색 서비스"""
    
    def search_table_name_standards(self, query: str, domain_hint: Optional[str] = None) -> List[StandardSearchResult]:
        """테이블명 표준 검색"""
        
    def search_column_name_standards(self, query: str, domain_hint: Optional[str] = None) -> List[StandardSearchResult]:
        """컬럼명 표준 검색"""
        
    def search_api_path_standards(self, query: str, domain_hint: Optional[str] = None) -> List[StandardSearchResult]:
        """API 경로 표준 검색"""
        
    def search_terminology_standards(self, query: str, domain_hint: Optional[str] = None) -> List[StandardSearchResult]:
        """용어 표준 검색"""
```

#### 5. 표준 변환기
**파일**: `backend-generators/src/project_generator/workflows/aggregate_draft/standard_transformer.py`

```python
class AggregateDraftStandardTransformer:
    def transform(self, draft_options: List[Dict], bounded_context: Dict, 
                  job_id: Optional[str] = None, 
                  firebase_update_callback: Optional[callable] = None,
                  transformation_session_id: Optional[str] = None) -> Dict:
        """Aggregate 초안을 표준에 맞게 변환"""
        
    def _extract_names_from_draft(self, draft_options: List[Dict]) -> List[str]:
        """Aggregate 초안에서 모든 이름 추출"""
        
    def _build_standard_queries(self, names: List[str], bounded_context: Dict) -> List[StandardQuery]:
        """카테고리별 StandardQuery 생성 (짧은 키워드만 사용)"""
        
    def _retrieve_relevant_standards_with_categories(self, standard_queries: List[StandardQuery]) -> List[Dict]:
        """카테고리별 RAG 검색 수행"""
        
    def _build_global_standard_mapping_context(self, relevant_standards: List[Dict],
                                              user_id: str,
                                              transformation_session_id: Optional[str] = None) -> StandardMappingContext:
        """검색된 표준 문서들로부터 StandardMappingContext 생성 (Vector Store 인덱싱 포함)"""
        
    def _apply_standard_mappings(self, draft_options: List[Dict], mapping: StandardMappingContext) -> List[Dict]:
        """Deterministic 룰 적용 (선행 치환)"""
        
    def _transform_with_llm(self, draft_options: List[Dict],
                           bounded_context: Dict,
                           relevant_standards: List[Dict],
                           query_search_results: Optional[List[Dict]] = None,
                           original_draft_options: Optional[List[Dict]] = None) -> List[Dict]:
        """LLM을 사용하여 표준에 맞게 변환 (청킹 지원)"""
        
    def _transform_structure_with_chunking(self, structure_item: Dict, ...) -> Dict:
        """대용량 데이터를 청크로 분할하여 변환"""
        
    def _transform_fields_only_with_llm(self, structure_item: Dict, ...) -> Dict:
        """필드 전용 변환 (청킹)"""
        
    def _transform_enums_vos_only_with_llm(self, structure_item: Dict, ...) -> Dict:
        """Enum/VO 전용 변환 (청킹)"""
        
    def _strip_unnecessary_fields_for_llm(self, draft_options: List[Dict]) -> List[Dict]:
        """LLM 요청 전 불필요한 필드 제거 (refs, description 등)"""
```

**주요 특징**:
- **refs 보존**: 변환 시 refs를 제거했다가 완료 후 원본에서 복원 (alias 기반 매칭)
- **청킹 메커니즘**: 대용량 데이터(필드, Enum, VO)를 청크로 분할하여 토큰 제한 회피
- **상세 진행 상황**: BC > Agg > Property Type > Chunk 단위로 Firebase를 통해 진행 상황 업데이트
- **Vector Store 세션 관리**: 사용자별 세션 ID로 인덱싱 상태 관리

#### 6. Traceability Generator
**파일**: `backend-generators/src/project_generator/workflows/aggregate_draft/traceability_generator.py`

도메인 객체(aggregates, enumerations, valueObjects)에 추적성 정보(refs)를 추가합니다.

**주요 메서드**:
- `generate()`: 추적성 생성 메인 함수
- `_extract_all_domain_objects()`: 모든 도메인 객체 추출 (중복 제거)
- `_filter_generated_draft_options()`: name, alias만 남기고 필터링
- `_add_line_numbers()`: 요구사항에 라인 번호 추가
- `_build_prompt()`: LLM 프롬프트 생성
- `_convert_refs_to_indexes()`: refs를 phrase → indexes로 변환
- `_sanitize_and_convert_refs()`: phrase를 column indexes로 변환
- `_convert_to_original_refs_using_trace_map()`: traceMap을 사용해 원본 라인으로 역변환

**변환 과정**:
1. **도메인 객체 추출**: 모든 aggregates, enumerations, valueObjects 추출
2. **필터링**: name, alias만 남기고 필터링 (LLM에 전달)
3. **라인 번호 추가**: 요구사항에 XML 형식으로 라인 번호 추가
4. **LLM 호출**: 도메인 객체와 요구사항을 매핑하여 refs 생성
5. **refs 변환**: 
   - phrase → column indexes 변환 (`_sanitize_and_convert_refs`)
   - traceMap을 사용해 원본 라인으로 역변환 (`_convert_to_original_refs_using_trace_map`)

**refs 형식**:
- LLM 출력: `[[[lineNumber, "phrase"], [lineNumber, "phrase"]]]`
- 변환 후: `[[[lineNumber, columnIndex], [lineNumber, columnIndex]]]`
- 최종: traceMap을 사용해 원본 요구사항 라인으로 역변환

#### 7. 백엔드 메인
**파일**: `backend-generators/src/project_generator/main.py`

```python
# 표준 변환 Job 처리
async def process_standard_transformation_job(job_id: str, complete_job_func: callable):
    """표준 변환 Job 처리"""
    transformer = AggregateDraftStandardTransformer()
    result = transformer.transform(
        draft_options, 
        bounded_context,
        job_id=job_id,
        firebase_update_callback=update_callback,
        transformation_session_id=transformation_session_id
    )

# 추적성 추가 Job 처리
async def process_traceability_job(job_id: str, complete_job_func: callable):
    """추적성 추가 Job 처리"""
    generator = TraceabilityGenerator()
    result = generator.generate(input_data)
```

#### 6. Job 관리
**파일**: `backend-generators/src/project_generator/utils/decentralized_job_manager.py`

```python
def _get_namespace_from_job_id(job_id: str) -> Optional[str]:
    """Job ID에서 namespace 추출"""
    # std-trans-{timestamp}-{random} → "standard_transformer"
```

**파일**: `backend-generators/src/project_generator/utils/job_util.py`

```python
def is_valid_job_id(job_id: str) -> bool:
    """Job ID 유효성 검사"""
    # std-trans- 패턴 검증
```

### Frontend 파일

#### 1. 프록시
**파일**: `src/components/designer/modeling/generators/proxies/StandardTransformerLangGraphProxy/StandardTransformerLangGraphProxy.js`

```javascript
class StandardTransformerLangGraphProxy {
    static generateJobId() {
        // std-trans-{timestamp}-{random}
    }
    
    static makeNewJob(draftOptions, boundedContext) {
        // Firebase Job 생성
    }
    
    static watchJob(jobId, onUpdate, onComplete, onError) {
        // Job 진행 상황 모니터링
    }
}
```

#### 2. 생성기
**파일**: `src/components/designer/modeling/generators/es-generators/StandardTransformer/StandardTransformerLangGraph.js`

```javascript
class StandardTransformerLangGraph {
    async generate(draftOptions, boundedContext) {
        // 표준 변환 Job 시작
    }
}
```

#### 3. UI 컴포넌트
**파일**: `src/components/designer/modeling/generators/ESDialoger.vue`

```javascript
transformWithStandards(boundedContextInfo, draftOptions, messageUniqueId) {
    // BC별로 순차 처리
    // 표준 변환 Job 시작
}
```

**파일**: `src/components/designer/modeling/generators/es-generators/components/AggregateDraftDialog/components/ESDialogerFooter.vue`

```vue
<v-btn @click="$emit('transformWithStandards', draftOptions[activeTab])">
    표준 적용
</v-btn>
```

---

## 스크립트 사용법

### 1. 표준 문서 인덱싱

**스크립트**: `backend-generators/scripts/index_standards.py`

```bash
# 기본 인덱싱
cd backend-generators
python scripts/index_standards.py

# 강제 재인덱싱 (기존 인덱스 삭제 후 재생성)
python scripts/index_standards.py --force

# 특정 경로 지정
python scripts/index_standards.py --path /path/to/standards
```

**출력 예시**:
```
🚀 Starting Standard Documents indexing...
📁 Standards path: /path/to/company_standards
🤖 Embedding model: text-embedding-3-small
📚 Loading standard documents...
✅ Loaded 5 chunks from db_api_naming_standards.xlsx
✅ Loaded 12 chunks from README.md
✅ Loaded 7 chunks from db_api_naming_standards.pptx
📊 Total documents to index: 24
📝 Indexing 24 documents...
✅ Indexing completed!
   Total documents indexed: 24
```

### 2. Vector Store 조회

**스크립트**: `backend-generators/scripts/query_vectorstore.py`

```bash
# 모든 문서 목록 조회
python scripts/query_vectorstore.py --list

# 검색
python scripts/query_vectorstore.py --search "Order aggregate table naming standard"

# 검색 (결과 수 지정)
python scripts/query_vectorstore.py --search "Order" --k 10
```

**출력 예시**:
```
📊 Vector Store에 저장된 문서 수: 24개

[1] ID: doc_001
    타입: database_standard
    출처: db_api_naming_standards.xlsx
    내용 미리보기: Order aggregate table naming standard: Use table prefix T_ with entity code ORD...
```

---

## 설정 및 환경 변수

### 환경 변수 (.env)

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Vector Store 경로 (선택사항, 기본값: ./knowledge_base/vectorstore)
VECTORSTORE_PATH=./knowledge_base/vectorstore

# Embedding 모델 (선택사항, 기본값: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small
```

### 설정 파일

**파일**: `backend-generators/src/project_generator/config.py`

```python
class Config:
    # Vector Store 경로
    VECTORSTORE_PATH = os.getenv('VECTORSTORE_PATH', './knowledge_base/vectorstore')
    
    # Embedding 모델
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # 표준 문서 경로
    COMPANY_STANDARDS_PATH = _project_root / 'knowledge_base' / 'company_standards'
```

### 표준 문서 디렉토리 구조

```
backend-generators/
└── knowledge_base/
    ├── company_standards/
    │   ├── db_api_naming_standards.xlsx
    │   ├── db_api_naming_standards.pptx
    │   └── README.md
    └── vectorstore/  # ChromaDB 저장소 (자동 생성)
```

---

## 트러블슈팅

### 1. Vector Store를 찾을 수 없음

**증상**: `⚠️  Vector Store not found at ./knowledge_base/vectorstore`

**해결**:
```bash
# 표준 문서 인덱싱 실행
python scripts/index_standards.py --force
```

### 2. 검색 결과가 0개

**증상**: `📚 검색된 표준 청크: 0개`

**원인**:
- Vector Store가 비어있음
- 쿼리와 문서 내용이 매칭되지 않음

**해결**:
```bash
# Vector Store 확인
python scripts/query_vectorstore.py --list

# 검색 테스트
python scripts/query_vectorstore.py --search "Order aggregate table naming standard"
```

### 3. LLM 스키마 오류

**증상**: `Invalid schema for response_format 'StandardTransformationResponse'`

**해결**: 스키마의 `required` 필드를 확인하고 수정 (이미 수정됨)

### 4. chromadb 모듈 없음

**증상**: `Could not import chromadb python package`

**해결**:
```bash
pip install chromadb
```

### 5. 백엔드가 Job을 감지하지 않음

**증상**: Job이 생성되었지만 백엔드에서 처리하지 않음

**확인 사항**:
1. `main.py`의 `monitored_namespaces`에 `'standard_transformer'` 포함 여부
2. `decentralized_job_manager.py`의 `_get_namespace_from_job_id`에 `std-trans-` 패턴 포함 여부
3. `job_util.py`의 `is_valid_job_id`에 `std-trans-` 패턴 포함 여부

---

## 로그 예시

### 성공적인 변환 로그

```
[StandardTransformer] [INFO] 🔄 표준 변환 시작
[StandardTransformer] [INFO] 📝 추출된 이름: 30개
[StandardTransformer] [INFO] 🔍 생성된 쿼리: 182개
[StandardTransformer] [INFO] 🔍 쿼리 [1/182]: 'Order aggregate table naming standard' → 3개 결과
[StandardTransformer] [INFO] 📊 검색 요약: 성공 150개, 실패 32개, 총 고유 결과 18개
[StandardTransformer] [INFO] 📚 최종 검색된 표준 청크: 18개
[StandardTransformer] [INFO] 📋 검색된 표준 상세:
   [1] 타입: database_standard, 출처: db_api_naming_standards.xlsx
      내용: Order aggregate table naming standard: Use table prefix T_ with entity code ORD...
[StandardTransformer] [INFO] 🔄 변환 결과:
   원본 옵션 수: 3개
   변환된 옵션 수: 3개
   첫 번째 옵션 구조 항목 수: 5개
   샘플 Aggregate: Order (alias: 주문)
[StandardTransformer] [INFO] ✅ 표준 변환 완료
```

---

## 주요 개념

### 1. 컬럼명 패턴 매칭

엑셀 파일의 시트별로 컬럼명이 다를 수 있으므로, 키워드 패턴 매칭 방식을 사용합니다.

**동작 원리**:
1. **첫 행을 헤더로 사용**: pandas의 기본 동작 (`header=0`)
   - 첫 행의 값들이 컬럼명(`df.columns`)이 됨
2. **키워드 패턴 매칭**: 컬럼명에 특정 키워드가 포함되어 있는지 확인
   - 하드코딩된 컬럼명이 아닌 키워드 리스트 사용
3. **값 추출**: 키워드가 포함된 컬럼의 값을 추출
4. **모든 컬럼 포함**: 패턴 매칭으로 찾은 컬럼뿐만 아니라 모든 컬럼을 `structured_data`에 저장

**장점**:
- 시트별로 컬럼명이 달라도 유연하게 처리
- 한글/영문 혼용 지원
- 컬럼명 변형에 대응 (예: "테이블명", "table_name", "테이블", "테이블_이름" 모두 인식)
- 하드코딩 없음: 컬럼명이 바뀌어도 키워드가 포함되어 있으면 자동 인식
- 모든 컬럼 정보 보존: LLM이 더 정확하게 파싱 가능

**예시**:
```python
# "테이블명" 컬럼 찾기 (하드코딩 없음)
table_name = self._find_column_by_keywords(
    row,
    ['table', '테이블']  # 키워드 리스트
)
# → "테이블명", "table_name", "테이블", "테이블_이름" 모두 매칭됨

# 모든 컬럼을 structured_data에 포함
for col, val in row.items():
    if col not in structured_data:  # 패턴 매칭으로 이미 저장된 컬럼 제외
        structured_data[col] = val  # 원본 컬럼명 그대로 저장
```

### 2. 하이브리드 임베딩 구조

표준 문서는 두 가지 형태로 저장됩니다:

- **텍스트 (page_content)**: 검색용 자연어 텍스트
  - 예: `"Order aggregate table naming standard: Use table prefix T_ with entity code ORD."`
  
- **JSON (metadata.structured_data)**: 파싱용 구조화된 데이터
  - 예: `{"entity_name": "Order", "table_prefix": "T_", "entity_code": "ORD"}`

이렇게 하면:
- **검색 효율성**: 자연어 텍스트로 의미적 유사성 검색
- **정확한 파싱**: JSON으로 정확한 값 추출

### 3. 카테고리별 검색 전략

각 이름에 대해 6가지 쿼리를 생성합니다:

1. `{name} aggregate table naming standard`
2. `{name} database naming convention`
3. `{name} API endpoint naming standard`
4. `{name} REST API naming convention`
5. `{name} terminology standard`
6. `{name} domain terminology`

이렇게 하면 다양한 관점에서 관련 표준을 찾을 수 있습니다.

### 4. 순차 처리

프론트엔드에서 여러 Bounded Context를 순차적으로 처리합니다:

```javascript
// ESDialoger.vue
const bcQueue = []; // BC별 큐
const processNextBC = function() {
    // 다음 BC 처리
};
```

### 5. refs 보존 메커니즘

표준 변환 시 refs는 다음과 같이 보존됩니다:

1. **변환 전**: `_strip_unnecessary_fields_for_llm()`에서 refs 제거 (토큰 절약)
2. **LLM 변환**: refs 없이 이름만 변환
3. **변환 후**: `copy.deepcopy(original_option)`으로 원본 복사 (refs 포함)
4. **이름 병합**: alias 기반 매칭으로 변환된 이름만 덮어쓰기
5. **refs 보존**: 원본의 refs가 그대로 유지됨

**코드 예시**:
```python
# 원본을 deep copy (refs 포함)
merged_option = copy.deepcopy(original_option)

# alias 기반 매칭으로 이름만 덮어쓰기
for orig_item in result_structure:
    orig_agg_alias = orig_item.get("aggregate", {}).get("alias")
    # ... alias로 매칭하여 이름만 업데이트
    # refs는 원본에 그대로 유지됨
```

### 6. 청킹 메커니즘

대용량 데이터 처리를 위해 청킹 메커니즘이 구현되어 있습니다:

**청킹 대상**:
- `enumerations`: Enum 배열을 청크로 분할
- `valueObjects`: VO 배열을 청크로 분할
- `previewAttributes`: 필드 배열을 청크로 분할
- `ddlFields`: DDL 필드 배열을 청크로 분할
- `query_search_results`: 검색 결과를 필터링하여 청크별로 전달

**청킹 전략**:
- Agg 정보는 최소화하여 전달 (name, alias만)
- 각 청크는 독립적으로 LLM에 전달
- 결과를 누적하여 최종 구조 생성

**동적 청크 크기 조정**:
- 예상 프롬프트 토큰 수에 따라 청크 크기 조정
- 기본 청크 크기: 10개 (필드/Enum/VO)
- 토큰 제한 초과 시 자동으로 청크 크기 감소

---

## 참고 자료

- **ChromaDB 문서**: https://www.trychroma.com/
- **LangChain 문서**: https://python.langchain.com/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

---

## 향후 개선 계획

### PPT 파일 구조화 처리 개선

**현재 상태:**
- **엑셀**: 구조화된 텍스트 + JSON 메타데이터 (하이브리드 임베딩) ✅
- **PPT**: 슬라이드 텍스트만 추출, 구조화된 데이터 없음 ❌

**문제점:**
1. PPT는 `structured_data` 메타데이터가 없어 LLM이 정확히 파싱하기 어려움
2. 엑셀과 PPT의 처리 방식이 일관되지 않음
3. PPT의 표나 구조화된 정보를 활용하기 어려움

**개선 계획:**

1. **PPT 구조화 파서 구현**
   - 슬라이드 텍스트를 파싱하여 규칙/예시 추출
   - 표(Table) 구조 파싱 (python-pptx의 Table 객체 활용)
   - 불릿 포인트, 번호 목록 구조 인식

2. **엑셀과 동일한 하이브리드 구조 적용**
   ```python
   # PPT도 엑셀처럼 구조화
   text, structured_data = self._format_ppt_slide_as_standard_text(slide)
   
   doc = Document(
       page_content=text,  # 자연어 텍스트 (임베딩용)
       metadata={
           "structured_data": json.dumps(structured_data)  # JSON (파싱용)
       }
   )
   ```

3. **구현 단계:**
   - [ ] 엑셀 인덱싱 검증 완료 후 진행
   - [ ] PPT 슬라이드 텍스트 파싱 로직 구현
   - [ ] PPT 표 구조 추출 로직 구현
   - [ ] `_format_ppt_slide_as_standard_text` 메서드 구현
   - [ ] 테스트 및 검증

**예상 효과:**
- PPT 표준 문서도 엑셀과 동일하게 정확한 파싱 가능
- LLM이 PPT 내용을 더 정확하게 활용
- 일관된 처리 방식으로 유지보수성 향상

---
