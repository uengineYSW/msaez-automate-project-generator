"""
Aggregate Draft Standard Transformer
생성된 Aggregate 초안을 회사 표준에 맞게 변환
RAG 기반으로 관련 표준만 동적으로 검색하여 적용
"""
from typing import Dict, List, Optional, TypedDict
from pathlib import Path
import json
import re
import os
from datetime import datetime
import pandas as pd
import tempfile
import shutil
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from project_generator.utils.logging_util import LoggingUtil
from src.project_generator.workflows.common.rag_retriever import RAGRetriever
from src.project_generator.workflows.common.standard_rag_service import (
    StandardRAGService, StandardQuery, StandardSearchResult
)
from src.project_generator.workflows.common.standard_loader import StandardLoader
from src.project_generator.config import Config


# ============================================================================
# Standard Mapping Context (Terminology/Standard Mapping 레이어)
# ============================================================================

class TableNameMapping(TypedDict):
    """테이블명 매핑"""
    entity_to_table: Dict[str, str]  # 한글/영문 도메인 명 -> 테이블명 (예: "주문 마스터" -> "T_ODR_M")
    table_standards: Dict[str, str]  # 모든 표준 매핑 (카테고리 구분 없음)
    column_standards: Dict[str, str]  # 모든 표준 매핑 (카테고리 구분 없음)
    domain_to_tables: Dict[str, List[str]]  # 도메인 코드 -> 테이블명 리스트 (예: "ODR" -> ["T_ODR_M", "T_ODR_D"])


class DomainCodeMapping(TypedDict):
    """도메인 코드 매핑"""
    name_to_domain: Dict[str, str]  # 도메인 명칭(한/영) -> 도메인 코드 (예: "주문"/"Order" -> "ODR")
    table_to_domain: Dict[str, str]  # 테이블명 -> 도메인 코드 (예: "T_ODR_M" -> "ODR")


class ColumnNameMapping(TypedDict):
    """컬럼명 매핑 (로깅용)"""
    column_desc_by_table: Dict[str, Dict[str, str]]  # 테이블명 -> {컬럼명 -> 설명} (로깅용, 실제 변환에는 사용 안 함)
    desc_to_columns: Dict[str, List[str]]  # 설명 -> 컬럼명 리스트 (로깅용, 실제 변환에는 사용 안 함)


class ApiPathMapping(TypedDict):
    """API 경로 매핑"""
    resource_abbrev: Dict[str, str]  # 개념명 -> 리소스 약어 (예: "Order" -> "odr")
    action_to_path: Dict[str, str]  # 행위/기능 -> API 경로 패턴 (예: "결제 요청" -> "/v1/pym/req")
    http_method_by_action: Dict[str, str]  # 행위 -> HTTP Method (예: "생성" -> "POST", "조회" -> "GET")


class StandardMappingContext(TypedDict):
    """표준 매핑 컨텍스트 (전체 매핑 사전)"""
    table: TableNameMapping
    domain: DomainCodeMapping
    column: ColumnNameMapping
    api: ApiPathMapping


class StandardTransformationState(TypedDict):
    """표준 변환 상태"""
    # Input
    draft_options: List[Dict]  # 생성된 Aggregate 초안 옵션들
    bounded_context: Dict  # Bounded Context 정보
    
    # Working state
    extracted_names: List[str]  # 추출된 이름들
    queries: List[str]  # 생성된 쿼리들
    relevant_standards: List[Dict]  # 검색된 관련 표준 청크들
    
    # Output
    transformed_options: List[Dict]  # 변환된 옵션들
    transformation_log: str  # 변환 로그
    is_completed: bool
    error: str


class AggregateDraftStandardTransformer:
    """
    Aggregate 초안 표준 변환기
    RAG를 사용하여 관련 표준만 검색하고 적용
    """
    
    # 클래스 레벨 변수: 세션별 인덱싱 상태 추적
    # 같은 transformation_session_id를 가진 BC들은 같은 벡터스토어를 공유
    _indexed_sessions = set()  # {transformation_session_id}
    _base_standards_indexed = set()  # {transformation_session_id} - 기본 표준 문서 인덱싱 완료된 세션
    _user_documents_downloaded = set()  # {user_id} - 사용자별 문서 다운로드 완료
    _user_vectorstores_indexed = set()  # {(user_id, transformation_session_id)} - 사용자별 Vector Store 인덱싱 완료
    _vectorstore_cleared_sessions = set()  # {(user_id, transformation_session_id)} - Vector Store 클리어 완료된 세션
    
    def __init__(self, enable_rag: bool = True, user_id: Optional[str] = None):
        """
        Args:
            enable_rag: RAG 기능 활성화 여부
            user_id: 사용자 ID (Firebase Storage에서 문서를 가져올 때 사용)
        """
        self.enable_rag = enable_rag
        self.user_id = user_id
        self.user_standards_path: Optional[Path] = None  # 사용자별 표준 문서 경로
        self.user_vectorstore_path: Optional[str] = None  # 사용자별 Vector Store 경로
        
        # 사용자 ID가 있으면 Firebase Storage에서 문서 다운로드
        if user_id:
            self.user_standards_path = self._download_user_standards_from_firebase(user_id)
            # 사용자별 Vector Store 경로 설정
            if self.user_standards_path:
                self.user_vectorstore_path = str(self.user_standards_path / 'vectorstore')
        
        # 사용자별 Vector Store 경로가 있으면 사용, 없으면 기본 경로 사용
        vectorstore_path = self.user_vectorstore_path if self.user_vectorstore_path else None
        self.rag_retriever = RAGRetriever(vectorstore_path=vectorstore_path) if enable_rag else None
        # StandardRAGService 초기화 (카테고리별 검색 지원)
        # 기본 임계값: 0.7 (RAG는 "참고용 컨텍스트 제공용"으로만 사용, 높은 유사도만 채택)
        # 필수 매핑은 전체 표준 원본을 직접 읽어서 global mapping 구성 (유사도 검색과 무관)
        # 환경 변수 STANDARD_TRANSFORMER_SCORE_THRESHOLD로 조정 가능
        score_threshold = Config.STANDARD_TRANSFORMER_SCORE_THRESHOLD
        self.standard_service = StandardRAGService(
            retriever=self.rag_retriever,
            score_threshold=score_threshold  # 기본값: 0.7
        ) if enable_rag and self.rag_retriever else None
        
        # StandardLoader 초기화 (전체 표준 원본 읽기용)
        # LLM 비활성화: 인덱싱 시 간단한 번역/키워드만 사용
        self.standard_loader = StandardLoader(enable_llm=False) if enable_rag else None
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=300,  # 5분 타임아웃 (초 단위) - LLM 응답 지연 대비
            max_tokens=32768
        )
        
        # aggregate_draft_generator와 동일한 패턴 사용
        self.llm_structured = self.llm.with_structured_output(
            self._get_response_schema(),
            strict=True
        )
        
        # 🔧 BC 간 참조를 위한 전역 aggregate 이름 매핑 (original_name -> transformed_name)
        # 모든 BC의 aggregate 변환 결과를 누적하여 저장
        self._global_aggregate_name_mapping: Dict[str, str] = {}
    
    def _download_user_standards_from_firebase(self, user_id: str) -> Optional[Path]:
        """
        Firebase Storage에서 사용자별 표준 문서를 다운로드하여 knowledge_base/company_standards/{user_id}/ 경로에 저장
        
        프로세스가 동작하는 동안 임시적으로 저장되며, 프로세스 종료 시 정리됩니다.
        같은 사용자 ID의 경우 한 번만 다운로드하여 재사용합니다.
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            다운로드한 문서가 저장된 경로 (Path 객체), 실패 시 None
        """
        # 이미 다운로드된 경우 스킵
        if user_id in AggregateDraftStandardTransformer._user_documents_downloaded:
            user_standards_dir = Config.COMPANY_STANDARDS_PATH / user_id
            if user_standards_dir.exists() and any(user_standards_dir.iterdir()):
                LoggingUtil.info("StandardTransformer", f"♻️  사용자({user_id}) 표준 문서 재사용 (이미 다운로드됨)")
                return user_standards_dir
        
        try:
            from firebase_admin import storage as firebase_storage
            
            LoggingUtil.info("StandardTransformer", f"📥 Firebase Storage에서 사용자({user_id}) 표준 문서 다운로드 시작")
            
            # knowledge_base/company_standards/{user_id}/ 경로에 저장
            user_standards_dir = Config.COMPANY_STANDARDS_PATH / user_id
            
            # 디렉토리 생성 및 권한 설정 (non-root 사용자를 위한 권한 설정)
            user_standards_dir.mkdir(parents=True, exist_ok=True)
            # 디렉토리와 부모 디렉토리들에 쓰기 권한 부여
            try:
                os.chmod(user_standards_dir, 0o777)
                # 부모 디렉토리들도 권한 설정
                parent = user_standards_dir.parent
                while parent.exists() and parent != Config.COMPANY_STANDARDS_PATH.parent:
                    os.chmod(parent, 0o777)
                    parent = parent.parent
                if Config.COMPANY_STANDARDS_PATH.exists():
                    os.chmod(Config.COMPANY_STANDARDS_PATH, 0o777)
            except (OSError, PermissionError) as perm_error:
                LoggingUtil.warning("StandardTransformer", f"⚠️  디렉토리 권한 설정 실패 (계속 진행): {perm_error}")
            
            # Firebase Storage에서 파일 목록 조회
            # bucket 이름을 환경 변수에서 가져오거나 명시적으로 지정
            storage_bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')
            if storage_bucket_name:
                # gs:// prefix 제거 (있는 경우)
                if storage_bucket_name.startswith('gs://'):
                    storage_bucket_name = storage_bucket_name[5:]
                bucket = firebase_storage.bucket(storage_bucket_name)
            else:
                bucket = firebase_storage.bucket()
            storage_path = f"standard-documents/{user_id}/"
            blobs = bucket.list_blobs(prefix=storage_path)
            
            downloaded_files = []
            for blob in blobs:
                # 디렉토리는 건너뛰기
                if blob.name.endswith('/'):
                    continue
                
                # 파일명 추출 (standard-documents/{user_id}/standard-document.xlsx -> standard-document.xlsx)
                file_name = blob.name.split('/')[-1]
                
                # 사용자별 디렉토리에 파일 다운로드
                local_file_path = user_standards_dir / file_name
                blob.download_to_filename(str(local_file_path))
                # 다운로드된 파일에 쓰기 권한 부여 (non-root 사용자를 위해)
                try:
                    os.chmod(local_file_path, 0o666)
                except (OSError, PermissionError):
                    pass  # 권한 설정 실패해도 계속 진행
                downloaded_files.append(file_name)
                LoggingUtil.info("StandardTransformer", f"✅ 다운로드 완료: {file_name}")
            
            if downloaded_files:
                LoggingUtil.info("StandardTransformer", f"📁 사용자 표준 문서 경로: {user_standards_dir}")
                # 다운로드 완료 표시
                AggregateDraftStandardTransformer._user_documents_downloaded.add(user_id)
                return user_standards_dir
            else:
                LoggingUtil.warning("StandardTransformer", f"⚠️  사용자({user_id})의 표준 문서를 찾을 수 없습니다.")
                # 빈 디렉토리 정리
                try:
                    user_standards_dir.rmdir()
                except (OSError, FileNotFoundError):
                    pass
                return None
                
        except ImportError:
            LoggingUtil.warning("StandardTransformer", "⚠️  firebase_admin.storage를 사용할 수 없습니다.")
            return None
        except Exception as e:
            LoggingUtil.error("StandardTransformer", f"❌ Firebase Storage에서 문서 다운로드 실패: {e}")
            return None
    
    def cleanup_user_standards(self):
        """
        사용자별 표준 문서 임시 디렉토리 정리
        프로세스 종료 시 호출하여 임시 파일들을 삭제
        """
        if self.user_standards_path and self.user_standards_path.exists():
            try:
                # Vector Store도 함께 삭제
                if self.user_vectorstore_path and Path(self.user_vectorstore_path).exists():
                    shutil.rmtree(self.user_vectorstore_path, ignore_errors=True)
                
                # 사용자별 문서 디렉토리 삭제
                shutil.rmtree(self.user_standards_path, ignore_errors=True)
                LoggingUtil.info("StandardTransformer", f"🧹 사용자 표준 문서 정리 완료: {self.user_standards_path}")
            except Exception as e:
                LoggingUtil.warning("StandardTransformer", f"⚠️  사용자 표준 문서 정리 중 오류: {e}")
    
    def reprocess_all_bcs_with_complete_mapping(self, all_bc_results: List[Dict]) -> List[Dict]:
        """
        모든 BC 처리 완료 후 전체적으로 다시 prefix를 처리
        BC 간 참조 문제를 해결하기 위해 모든 BC의 aggregate 매핑이 완료된 후 재처리
        
        Args:
            all_bc_results: 모든 BC의 변환 결과 리스트
                각 항목은 {
                    'boundedContext': str,
                    'transformedOptions': List[Dict],
                    ...
                } 형태
        
        Returns:
            재처리된 모든 BC 결과 리스트
        """
        LoggingUtil.info("StandardTransformer", "🔄 모든 BC 재처리 시작 (전체 aggregate 매핑 사용)")
        
        # 1. 전역 매핑 사용 (original_name -> transformed_name)
        complete_aggregate_mapping = self._global_aggregate_name_mapping.copy()
        LoggingUtil.info("StandardTransformer", 
                       f"   [전체매핑수집] complete_aggregate_mapping={complete_aggregate_mapping}")
        
        # 2. 각 BC의 결과를 다시 처리
        reprocessed_results = []
        for bc_result in all_bc_results:
            bounded_context = bc_result.get("boundedContext")
            transformed_options = bc_result.get("transformedOptions", [])
            
            LoggingUtil.info("StandardTransformer", 
                           f"   [재처리] BC: {bounded_context}, 옵션 수: {len(transformed_options)}")
            
            # 각 옵션의 Enum/VO를 전체 매핑으로 다시 처리
            reprocessed_options = []
            for option in transformed_options:
                structure = option.get("structure", [])
                
                for item in structure:
                    aggregate = item.get("aggregate", {})
                    agg_alias = aggregate.get("alias")
                    current_agg_name = aggregate.get("name")
                    
                    # Enum 처리
                    enumerations = item.get("enumerations", [])
                    for enum in enumerations:
                        enum_alias = enum.get("alias")
                        enum_name = enum.get("name")
                        
                        # 전역 매핑에서 참조하는 aggregate 찾기
                        # Enum 이름이 원본 aggregate 이름으로 시작하는지 확인
                        for original_agg_name, transformed_agg_name in complete_aggregate_mapping.items():
                            # 예: "OrderStatus"는 "Order"로 시작
                            if enum_name and original_agg_name and enum_name.startswith(original_agg_name):
                                # aggregate 이름을 포함하는 경우 prefix 적용
                                suffix = enum_name[len(original_agg_name):]
                                import re
                                suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                new_enum_name = transformed_agg_name + "_" + suffix_snake if suffix_snake else transformed_agg_name + "_enum"
                                # 변환이 필요한 경우에만 적용 (이미 올바르게 변환된 경우는 제외)
                                if enum_name != new_enum_name:
                                    enum["name"] = new_enum_name
                                    LoggingUtil.info("StandardTransformer", 
                                                   f"   [재처리Enum] '{enum_alias}': '{enum_name}' → '{new_enum_name}' (참조: {original_agg_name} → {transformed_agg_name})")
                                break
                    
                    # VO 처리
                    value_objects = item.get("valueObjects", [])
                    for vo in value_objects:
                        vo_alias = vo.get("alias")
                        vo_name = vo.get("name")
                        
                        # 전역 매핑에서 참조하는 aggregate 찾기
                        # VO 이름이 원본 aggregate 이름으로 시작하는지 확인
                        for original_agg_name, transformed_agg_name in complete_aggregate_mapping.items():
                            # 예: "CustomerReference"는 "Customer"로 시작
                            if vo_name and original_agg_name and vo_name.startswith(original_agg_name):
                                # aggregate 이름을 포함하는 경우 prefix 적용
                                suffix = vo_name[len(original_agg_name):]
                                import re
                                suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                new_vo_name = transformed_agg_name + "_" + suffix_snake if suffix_snake else transformed_agg_name + "_vo"
                                # 변환이 필요한 경우에만 적용 (이미 올바르게 변환된 경우는 제외)
                                if vo_name != new_vo_name:
                                    vo["name"] = new_vo_name
                                    LoggingUtil.info("StandardTransformer", 
                                                   f"   [재처리VO] '{vo_alias}': '{vo_name}' → '{new_vo_name}' (참조: {original_agg_name} → {transformed_agg_name})")
                                break
                
                reprocessed_options.append(option)
            
            reprocessed_results.append({
                **bc_result,
                "transformedOptions": reprocessed_options
            })
        
        LoggingUtil.info("StandardTransformer", "✅ 모든 BC 재처리 완료")
        return reprocessed_results
    
    def transform(self, draft_options: List[Dict], bounded_context: Dict, job_id: Optional[str] = None, 
                  firebase_update_callback: Optional[callable] = None, transformation_session_id: Optional[str] = None) -> Dict:
        """
        Aggregate 초안을 표준에 맞게 변환
        
        Args:
            draft_options: 생성된 Aggregate 초안 옵션들
            bounded_context: Bounded Context 정보
            job_id: Job ID (결과 저장용, 선택사항)
            firebase_update_callback: Firebase 업데이트 콜백 함수 (선택사항)
            
        Returns:
            변환된 옵션들과 변환 로그
        """
        def _update_job_progress(progress: int, stage: str, 
                                 bc_name: Optional[str] = None,
                                 agg_name: Optional[str] = None,
                                 property_type: Optional[str] = None,  # "enum", "vo", "field", "aggregate"
                                 chunk_info: Optional[str] = None,  # "청크 1/2" 등
                                 status: str = "processing",  # "processing", "completed", "error"
                                 error_message: Optional[str] = None):
            """Firebase에 상세 진행 상황 업데이트"""
            if firebase_update_callback:
                try:
                    # 상세 정보 구성
                    detail_info = []
                    if bc_name:
                        detail_info.append(f"BC: {bc_name}")
                    if agg_name:
                        detail_info.append(f"Agg: {agg_name}")
                    if property_type:
                        property_label = {
                            "aggregate": "Aggregate",
                            "enum": "Enum",
                            "vo": "ValueObject",
                            "field": "Field"
                        }.get(property_type, property_type)
                        detail_info.append(f"{property_label}")
                    if chunk_info:
                        detail_info.append(chunk_info)
                    
                    detail_text = " > ".join(detail_info) if detail_info else stage
                    
                    # 상태에 따른 메시지 포맷
                    if status == "error":
                        message = f"❌ 오류: {detail_text}"
                        if error_message:
                            message += f" ({error_message})"
                    elif status == "completed":
                        message = f"✅ 완료: {detail_text}"
                    else:
                        message = f"🔄 처리 중: {detail_text}"
                    
                    firebase_update_callback({
                        'progress': progress,
                        'transformationLog': message,
                        'isCompleted': False,
                        'currentBC': bc_name,
                        'currentAgg': agg_name,
                        'currentPropertyType': property_type,
                        'chunkInfo': chunk_info,
                        'status': status,
                        'error': error_message if status == "error" else None
                    })
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", f"Firebase 업데이트 실패: {e}")
        
        try:
            LoggingUtil.info("StandardTransformer", "🔄 표준 변환 시작")
            
            # 사용자별 Vector Store 인덱싱 상태 확인
            # 같은 (user_id, transformation_session_id) 조합에 대해 한 번만 인덱싱
            user_vectorstore_key = (self.user_id, transformation_session_id) if self.user_id and transformation_session_id else None
            should_index_vectorstore = False
            should_clear_vectorstore = False
            
            if user_vectorstore_key:
                # 사용자별 Vector Store가 아직 인덱싱되지 않은 경우
                if user_vectorstore_key not in AggregateDraftStandardTransformer._user_vectorstores_indexed:
                    should_index_vectorstore = True
                    should_clear_vectorstore = True
                    AggregateDraftStandardTransformer._user_vectorstores_indexed.add(user_vectorstore_key)
                    LoggingUtil.info("StandardTransformer", f"📝 사용자별 Vector Store 인덱싱 필요: user_id={self.user_id}, session={transformation_session_id}")
                else:
                    # 이미 인덱싱된 것으로 표시되어 있지만, Vector Store가 실제로 작동하는지 검증
                    if self.rag_retriever and (not self.rag_retriever._initialized or not self.rag_retriever.vectorstore):
                        # Vector Store 초기화 실패 감지: 인덱싱 상태 제거하고 재인덱싱
                        AggregateDraftStandardTransformer._user_vectorstores_indexed.discard(user_vectorstore_key)
                        should_index_vectorstore = True
                        should_clear_vectorstore = True
                        LoggingUtil.warning("StandardTransformer", 
                                          f"⚠️  Vector Store 초기화 실패 감지: 인덱싱 상태 초기화 및 재인덱싱 시도 (user_id={self.user_id}, session={transformation_session_id})")
                    else:
                        LoggingUtil.info("StandardTransformer", f"♻️  사용자별 Vector Store 재사용: user_id={self.user_id}, session={transformation_session_id}")
            elif transformation_session_id:
                # transformation_session_id만 있는 경우 (기존 로직)
                if transformation_session_id not in AggregateDraftStandardTransformer._indexed_sessions:
                    should_clear_vectorstore = True
                    AggregateDraftStandardTransformer._indexed_sessions.add(transformation_session_id)
            else:
                # transformation_session_id가 없으면 항상 클리어 (기존 동작 유지)
                should_clear_vectorstore = True
            
            # 같은 세션에서는 Vector Store를 한 번만 클리어 (중복 클리어 방지)
            vectorstore_clear_key = None
            if self.user_id and transformation_session_id:
                vectorstore_clear_key = (self.user_id, transformation_session_id)
            elif transformation_session_id:
                vectorstore_clear_key = (None, transformation_session_id)
            
            should_clear_now = should_clear_vectorstore
            if vectorstore_clear_key and vectorstore_clear_key in AggregateDraftStandardTransformer._vectorstore_cleared_sessions:
                should_clear_now = False  # 이미 클리어된 세션이면 스킵
                LoggingUtil.info("StandardTransformer", 
                               f"ℹ️  Vector Store는 이미 클리어됨 (세션: {transformation_session_id}). 스킵합니다.")
            
            if should_clear_now and self.rag_retriever:
                LoggingUtil.info("StandardTransformer", "🗑️  기존 Vector Store 클리어 중...")
                try:
                    clear_success = self.rag_retriever.clear_vectorstore()
                    if clear_success and vectorstore_clear_key:
                        # 클리어 성공 시 세션 기록
                        AggregateDraftStandardTransformer._vectorstore_cleared_sessions.add(vectorstore_clear_key)
                    elif not clear_success:
                        # Vector Store 클리어 실패 시 ChromaDB 데이터베이스 손상 가능성
                        # RAGRetriever의 복구 메서드 사용
                        LoggingUtil.warning("StandardTransformer", 
                                          f"⚠️  Vector Store 클리어 실패: 데이터베이스 손상 가능성. 복구 시도 중...")
                        try:
                            # RAGRetriever의 복구 메서드 호출
                            if hasattr(self.rag_retriever, '_repair_vectorstore'):
                                repair_success = self.rag_retriever._repair_vectorstore()
                                if repair_success:
                                    LoggingUtil.info("StandardTransformer", "✅ Vector Store 복구 및 재초기화 완료")
                                    if vectorstore_clear_key:
                                        AggregateDraftStandardTransformer._vectorstore_cleared_sessions.add(vectorstore_clear_key)
                                else:
                                    LoggingUtil.warning("StandardTransformer", 
                                                      f"⚠️  Vector Store 복구 실패: _initialized={self.rag_retriever._initialized}, vectorstore={self.rag_retriever.vectorstore is not None}")
                            else:
                                # 구버전 호환성: 수동 복구
                                import shutil
                                vectorstore_path = Path(self.rag_retriever.vectorstore_path) if hasattr(self.rag_retriever, 'vectorstore_path') else None
                                if vectorstore_path and vectorstore_path.exists():
                                    shutil.rmtree(vectorstore_path)
                                LoggingUtil.info("StandardTransformer", f"🗑️  손상된 Vector Store 디렉토리 삭제 완료: {vectorstore_path}")
                                # RAGRetriever 재초기화
                                from src.project_generator.workflows.common.rag_retriever import RAGRetriever
                                self.rag_retriever = RAGRetriever(vectorstore_path=str(vectorstore_path))
                                if not self.rag_retriever._initialized or not self.rag_retriever.vectorstore:
                                    LoggingUtil.warning("StandardTransformer", 
                                                      f"⚠️  Vector Store 재초기화 실패: _initialized={self.rag_retriever._initialized}, vectorstore={self.rag_retriever.vectorstore is not None}")
                                else:
                                    LoggingUtil.info("StandardTransformer", "✅ Vector Store 재초기화 완료")
                                if vectorstore_clear_key:
                                    AggregateDraftStandardTransformer._vectorstore_cleared_sessions.add(vectorstore_clear_key)
                        except Exception as cleanup_e:
                            LoggingUtil.warning("StandardTransformer", f"⚠️  Vector Store 복구 실패: {cleanup_e}")
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", f"⚠️  Vector Store 클리어 실패 (무시하고 계속): {e}")
            
            transformation_logs = []  # 진행 단계 로그 수집
            
            # 🔄 구조 변경: BC 단위가 아닌 structure(aggregate) 단위로 처리
            # 각 structure를 개별적으로 처리하여 프롬프트 복잡도 감소
            
            transformation_logs.append("표준 매핑 컨텍스트 생성 중...")
            _update_job_progress(0, "표준 매핑 컨텍스트 생성 중...")
            
            # 1. StandardMappingContext 생성 (선행 처리) - BC 전체 기준
            mapping_context: Optional[StandardMappingContext] = None
            mapping_context = self._build_global_standard_mapping_context(
                draft_options=draft_options,
                bounded_context=bounded_context,
                transformation_session_id=transformation_session_id,
                should_index_vectorstore=should_index_vectorstore
            )
            transformation_logs.append("선처리 매핑 적용 중...")
            _update_job_progress(20, "선처리 매핑 적용 중...")
            
            # 2. 선처리 전 원본 저장 (Enum/VO 후처리 시 매핑 정보 추출용)
            import copy
            original_draft_options = copy.deepcopy(draft_options)
            
            # 3. Deterministic 룰 적용 (선행 치환) - BC 전체에 적용
            mapped_draft_options = draft_options
            if mapping_context:
                mapped_draft_options = self._apply_standard_mappings(draft_options, mapping_context)
            
            transformation_logs.append("structure 단위 변환 시작...")
            _update_job_progress(30, "structure 단위 변환 시작...")
            
            # 4. 각 option의 structure를 개별적으로 처리
            transformed_options = []
            all_query_search_results = []  # 모든 structure의 쿼리별 검색 결과 수집
            all_relevant_standards = []  # 모든 structure의 검색된 표준 수집
            seen_standard_keys = set()  # 중복 제거용
            
            for opt_idx, option in enumerate(mapped_draft_options):
                structure = option.get("structure", [])
                transformed_structure = []
                
                LoggingUtil.info("StandardTransformer", 
                               f"📦 Option {opt_idx + 1}/{len(mapped_draft_options)} 처리 시작: {len(structure)}개 structure")
                
                # 각 structure(aggregate) 단위로 처리
                for struct_idx, struct_item in enumerate(structure):
                    agg_name = struct_item.get("aggregate", {}).get("name", "Unknown")
                    LoggingUtil.info("StandardTransformer", 
                                   f"   🔄 Structure {struct_idx + 1}/{len(structure)} 처리 시작: {agg_name}")
                    # 단일 structure로 구성된 임시 옵션 생성
                    single_structure_option = {
                        "structure": [struct_item]
                    }
                    single_structure_options = [single_structure_option]
                    
                    # 이 structure에서 이름 추출
                    structure_names = self._extract_names_from_structure(
                        single_structure_options, bounded_context
                    )
                    
                    # 선처리로 매핑된 이름 제외
                    mapped_names = set()
                    if mapping_context:
                        for name in structure_names:
                            if name in mapping_context["table"]["entity_to_table"]:
                                mapped_names.add(name)
                    
                    filtered_names = [n for n in structure_names if n not in mapped_names]
                    
                    # 쿼리 생성
                    standard_queries = self._build_standard_queries(
                        filtered_names,
                        bounded_context,
                        single_structure_options
                    )
                    
                    # RAG 검색 (top-k=3)
                    relevant_standards = []
                    query_search_results = []
                    has_rag_search_results = False
                    if self.enable_rag and self.standard_service and len(standard_queries) > 0:
                        search_result = self._retrieve_relevant_standards_with_categories(
                            standard_queries,
                            k_per_query=3,  # top-k 3개로 변경
                            transformation_session_id=transformation_session_id
                        )
                        if isinstance(search_result, tuple):
                            relevant_standards, query_search_results = search_result
                        else:
                            relevant_standards = search_result
                            query_search_results = getattr(self, '_last_query_search_results', [])
                        
                        # 검색 결과가 실제로 있는지 확인
                        has_rag_search_results = len(relevant_standards) > 0 or len(query_search_results) > 0
                        
                        if not has_rag_search_results:
                            LoggingUtil.warning("StandardTransformer", 
                                              f"⚠️  RAG 검색 결과 없음: {agg_name} - Vector Store 인덱싱 실패 또는 관련 표준 문서 없음. LLM이 표준 문서 없이 변환을 수행합니다.")
                            if firebase_update_callback:
                                try:
                                    bc_name_val = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
                                    _update_job_progress(0, f"⚠️ RAG 검색 결과 없음: {agg_name} (표준 문서 없이 변환 중)",
                                                        bc_name=bc_name_val,
                                                        agg_name=agg_name,
                                                        property_type="aggregate",
                                                        status="processing",
                                                        error_message="RAG 검색 결과 없음 - Vector Store 인덱싱 실패 가능")
                                except Exception as e:
                                    LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                        
                        all_query_search_results.extend(query_search_results)
                        
                        # relevant_standards 중복 제거하여 수집
                        for std in relevant_standards:
                            metadata = std.get("metadata", {})
                            source = metadata.get("source", "")
                            chunk_index = metadata.get("chunk_index", "")
                            std_key = f"{source}::{chunk_index}"
                            if std_key not in seen_standard_keys:
                                seen_standard_keys.add(std_key)
                                all_relevant_standards.append(std)
                    elif self.enable_rag and len(standard_queries) == 0:
                        LoggingUtil.warning("StandardTransformer", 
                                          f"⚠️  표준 쿼리 없음: {agg_name} - 변환할 항목이 없어 RAG 검색을 수행하지 않습니다.")
                    elif not self.enable_rag:
                        LoggingUtil.info("StandardTransformer", 
                                       f"ℹ️  RAG 비활성화: {agg_name} - RAG 검색 없이 LLM 변환 수행")
                    
                    # 단일 structure LLM 변환
                    try:
                        # original_structure_item 안전하게 가져오기
                        original_structure_item = None
                        try:
                            if (opt_idx < len(original_draft_options) and 
                                original_draft_options[opt_idx] and
                                "structure" in original_draft_options[opt_idx] and
                                struct_idx < len(original_draft_options[opt_idx]["structure"])):
                                original_structure_item = original_draft_options[opt_idx]["structure"][struct_idx]
                        except (IndexError, KeyError, TypeError) as e:
                            LoggingUtil.warning("StandardTransformer", 
                                              f"⚠️  원본 structure 항목 가져오기 실패 (opt_idx={opt_idx}, struct_idx={struct_idx}): {e}")
                        
                        LoggingUtil.info("StandardTransformer", 
                                       f"      🤖 LLM 변환 시작: {agg_name}")
                        
                        # BC, Agg 정보 추출 (클로저를 위해 로컬 변수로 저장)
                        bc_name_val = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
                        agg_name_val = agg_name
                        
                        # BC, Agg 정보를 포함한 콜백 래퍼 생성
                        def _update_progress_with_context(progress: int, stage: str, 
                                                          property_type: Optional[str] = None,
                                                          chunk_info: Optional[str] = None,
                                                          status: str = "processing",
                                                          error_message: Optional[str] = None,
                                                          bc_name: Optional[str] = None,
                                                          agg_name: Optional[str] = None):
                            _update_job_progress(
                                progress=progress,
                                stage=stage,
                                bc_name=bc_name or bc_name_val,
                                agg_name=agg_name or agg_name_val,
                                property_type=property_type,
                                chunk_info=chunk_info,
                                status=status,
                                error_message=error_message
                            )
                        
                        # 원본 option의 boundedContext 정보 가져오기 (청킹 처리용)
                        original_option_bounded_context = None
                        if opt_idx < len(original_draft_options) and original_draft_options[opt_idx]:
                            original_option_bounded_context = original_draft_options[opt_idx].get("boundedContext")
                        
                        transformed_single_structure = self._transform_single_structure_with_llm(
                            structure_item=struct_item,
                            bounded_context=bounded_context,
                            relevant_standards=relevant_standards,
                            query_search_results=query_search_results,
                            original_structure_item=original_structure_item,
                            mapping_context=mapping_context,
                            update_progress_callback=_update_progress_with_context,
                            original_option_bounded_context=original_option_bounded_context
                        )
                        
                        # 변환 결과가 None이거나 비어있으면 원본 사용
                        if not transformed_single_structure:
                            LoggingUtil.warning("StandardTransformer", 
                                              f"⚠️  structure 변환 결과가 비어있음, 원본 사용 (opt_idx={opt_idx}, struct_idx={struct_idx})")
                            transformed_single_structure = struct_item
                        
                        transformed_structure.append(transformed_single_structure)
                        LoggingUtil.info("StandardTransformer", 
                                       f"      ✅ Structure {struct_idx + 1}/{len(structure)} 변환 완료: {agg_name}")
                        
                        # 변환 완료 알림
                        if firebase_update_callback:
                            try:
                                bc_name_val = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
                                _update_job_progress(100, f"변환 완료: {agg_name}",
                                                    bc_name=bc_name_val,
                                                    agg_name=agg_name,
                                                    property_type="aggregate",
                                                    status="completed")
                            except Exception as update_e:
                                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                        
                    except Exception as e:
                        LoggingUtil.error("StandardTransformer", 
                                        f"❌ Structure {struct_idx + 1}/{len(structure)} 변환 중 오류 발생: {agg_name}, {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # 오류 알림
                        if firebase_update_callback:
                            try:
                                _update_job_progress(0, f"변환 실패: {agg_name}",
                                                    bc_name=bc_name,
                                                    agg_name=agg_name,
                                                    property_type="aggregate",
                                                    status="error",
                                                    error_message=str(e))
                            except Exception as update_e:
                                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                        
                        # 오류 발생 시 원본 사용
                        transformed_structure.append(struct_item)
                
                # 변환된 structure들을 원래 option에 합치기
                # 🔒 CRITICAL: original_draft_options에서 필터링된 필드 복원
                if opt_idx < len(original_draft_options):
                    # original_draft_options에서 원본 옵션 가져오기
                    original_option = original_draft_options[opt_idx]
                    transformed_option = copy.deepcopy(original_option)
                    # 변환된 structure만 교체
                    transformed_option["structure"] = transformed_structure
                else:
                    # original_draft_options가 없으면 현재 option 사용
                    transformed_option = copy.deepcopy(option)
                    transformed_option["structure"] = transformed_structure
                transformed_options.append(transformed_option)
            
            transformation_logs.append(f"LLM 변환 완료: {len(transformed_options)}개 옵션, 총 {sum(len(opt.get('structure', [])) for opt in transformed_options)}개 structure")
            _update_job_progress(80, "후처리 중...")
            
            LoggingUtil.info("StandardTransformer", f"✅ 표준 변환 완료: {len(transformed_options)}개 옵션 변환됨")
            transformation_logs.append(f"변환 완료: {len(transformed_options)}개 옵션")
            _update_job_progress(80, "후처리 중...")
            
            # 변환 결과 검증: 원본보다 옵션이 많이 줄어들면 경고
            if len(transformed_options) < len(draft_options) * 0.5:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  변환된 옵션이 원본의 50% 미만입니다. "
                                  f"원본: {len(draft_options)}개, 변환: {len(transformed_options)}개")
            
            # 변환 전후 결과를 JSON 파일로 저장
            if job_id:
                # 검색된 표준 정보와 전처리 매핑 정보 수집
                search_info = {
                    "relevant_standards": all_relevant_standards,  # 모든 structure의 검색된 표준 (중복 제거)
                    "query_search_results": all_query_search_results,  # 모든 structure의 쿼리별 검색 결과 (top-k=3)
                    "mapping_context": mapping_context,  # 전처리 매핑 컨텍스트
                    "mapped_draft_options": mapped_draft_options  # 전처리 매핑 후 옵션
                }
                self._save_transformation_results(
                    job_id, 
                    draft_options, 
                    transformed_options, 
                    bounded_context,
                    search_info=search_info
                )
            
            # 이름 개수 계산 (BC 전체 기준)
            all_names = self._extract_names_from_draft(draft_options, bounded_context)
            total_names = len(all_names)
            
            return {
                "transformed_options": transformed_options,  # snake_case (내부 사용)
                "transformedOptions": transformed_options,   # camelCase (프론트엔드 호환)
                "transformation_log": " → ".join(transformation_logs),
                "transformationLog": " → ".join(transformation_logs),
                "is_completed": True,
                "isCompleted": True
                # error가 None이면 필드 자체를 포함하지 않음
            }
        
        except Exception as e:
            LoggingUtil.error("StandardTransformer", f"❌ 표준 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "transformed_options": draft_options,  # 원본 반환
                "transformedOptions": draft_options,   # camelCase (프론트엔드 호환)
                "transformation_log": f"변환 실패: {str(e)}",
                "transformationLog": f"변환 실패: {str(e)}",
                "is_completed": False,
                "isCompleted": False,
                "error": str(e)
            }
    
    def _extract_names_from_draft(self, draft_options: List[Dict], bounded_context: Dict) -> List[str]:
        """
        Aggregate 초안에서 모든 이름 추출 (카테고리 구분 없음)
        name과 alias를 "name alias" 형태로 묶어서 처리
        
        Returns:
            이름 리스트: ["Customer 고객", "Order 주문", "customer_id 고객ID", ...]
        """
        all_names = set()  # 모든 이름 (카테고리 구분 없음, "name alias" 형태로 묶음)
        
        # BoundedContext 이름 추가 ("name alias" 형태)
        bc_name = bounded_context.get("name", "")
        bc_alias = bounded_context.get("alias", "")
        if bc_name and bc_alias:
            all_names.add(f"{bc_name} {bc_alias}")
        elif bc_name:
            all_names.add(bc_name)
        elif bc_alias:
            all_names.add(bc_alias)
        
        for option in draft_options:
            structure = option.get("structure", [])
            
            for item in structure:
                # Aggregate 이름 ("name alias" 형태)
                aggregate = item.get("aggregate", {})
                agg_name = aggregate.get("name", "")
                agg_alias = aggregate.get("alias", "")
                if agg_name and agg_alias:
                    all_names.add(f"{agg_name} {agg_alias}")
                elif agg_name:
                    all_names.add(agg_name)
                elif agg_alias:
                    all_names.add(agg_alias)
                
                # Enumeration 이름 ("name alias" 형태)
                enumerations = item.get("enumerations", [])
                for enum in enumerations:
                    enum_name = enum.get("name", "")
                    enum_alias = enum.get("alias", "")
                    if enum_name and enum_alias:
                        all_names.add(f"{enum_name} {enum_alias}")
                    elif enum_name:
                        all_names.add(enum_name)
                    elif enum_alias:
                        all_names.add(enum_alias)
                
                # ValueObject 이름 ("name alias" 형태)
                value_objects = item.get("valueObjects", [])
                for vo in value_objects:
                    vo_name = vo.get("name", "")
                    vo_alias = vo.get("alias", "")
                    if vo_name and vo_alias:
                        all_names.add(f"{vo_name} {vo_alias}")
                    elif vo_name:
                        all_names.add(vo_name)
                    elif vo_alias:
                        all_names.add(vo_alias)
                    # referencedAggregateName은 별도로 추가
                    if vo.get("referencedAggregateName"):
                        all_names.add(vo["referencedAggregateName"])
                
                # previewAttributes 필드 이름 ("fieldName fieldAlias" 형태)
                preview_attributes = item.get("previewAttributes", [])
                for attr in preview_attributes:
                    if isinstance(attr, dict):
                        field_name = attr.get("fieldName", "")
                        field_alias = attr.get("fieldAlias", "")
                        if field_name and field_alias:
                            all_names.add(f"{field_name} {field_alias}")
                        elif field_name:
                            all_names.add(field_name)
                        elif field_alias:
                            all_names.add(field_alias)
                    else:
                        # backward compatibility
                        if attr:
                            all_names.add(str(attr))
                
                # ddlFields 필드 이름 ("fieldName fieldAlias" 형태)
                ddl_fields = item.get("ddlFields", [])
                for field in ddl_fields:
                    if isinstance(field, dict):
                        field_name = field.get("fieldName", "")
                        field_alias = field.get("fieldAlias", "")
                        if field_name and field_alias:
                            all_names.add(f"{field_name} {field_alias}")
                        elif field_name:
                            all_names.add(field_name)
                        elif field_alias:
                            all_names.add(field_alias)
                    else:
                        # backward compatibility
                        if field:
                            all_names.add(str(field))
        
        return list(all_names)
    
    def _extract_names_from_structure(self, draft_options: List[Dict], bounded_context: Dict) -> List[str]:
        """
        단일 structure에서 이름 추출 (structure 단위 처리용)
        name과 alias를 "name alias" 형태로 묶어서 처리
        
        Returns:
            이름 리스트: ["Customer 고객", "customer_id 고객ID", ...]
        """
        all_names = set()
        
        # BoundedContext 이름 추가
        bc_name = bounded_context.get("name", "")
        bc_alias = bounded_context.get("alias", "")
        if bc_name and bc_alias:
            all_names.add(f"{bc_name} {bc_alias}")
        elif bc_name:
            all_names.add(bc_name)
        elif bc_alias:
            all_names.add(bc_alias)
        
        for option in draft_options:
            structure = option.get("structure", [])
            
            for item in structure:
                # Aggregate 이름
                aggregate = item.get("aggregate", {})
                agg_name = aggregate.get("name", "")
                agg_alias = aggregate.get("alias", "")
                if agg_name and agg_alias:
                    all_names.add(f"{agg_name} {agg_alias}")
                elif agg_name:
                    all_names.add(agg_name)
                elif agg_alias:
                    all_names.add(agg_alias)
                
                # Enumeration 이름
                enumerations = item.get("enumerations", [])
                for enum in enumerations:
                    enum_name = enum.get("name", "")
                    enum_alias = enum.get("alias", "")
                    if enum_name and enum_alias:
                        all_names.add(f"{enum_name} {enum_alias}")
                    elif enum_name:
                        all_names.add(enum_name)
                    elif enum_alias:
                        all_names.add(enum_alias)
                
                # ValueObject 이름
                value_objects = item.get("valueObjects", [])
                for vo in value_objects:
                    vo_name = vo.get("name", "")
                    vo_alias = vo.get("alias", "")
                    if vo_name and vo_alias:
                        all_names.add(f"{vo_name} {vo_alias}")
                    elif vo_name:
                        all_names.add(vo_name)
                    elif vo_alias:
                        all_names.add(vo_alias)
                    if vo.get("referencedAggregateName"):
                        all_names.add(vo["referencedAggregateName"])
                
                # previewAttributes 필드 이름
                preview_attributes = item.get("previewAttributes", [])
                for attr in preview_attributes:
                    if isinstance(attr, dict):
                        field_name = attr.get("fieldName", "")
                        field_alias = attr.get("fieldAlias", "")
                        if field_name and field_alias:
                            all_names.add(f"{field_name} {field_alias}")
                        elif field_name:
                            all_names.add(field_name)
                        elif field_alias:
                            all_names.add(field_alias)
                    else:
                        if attr:
                            all_names.add(str(attr))
                
                # ddlFields 필드 이름
                ddl_fields = item.get("ddlFields", [])
                for field in ddl_fields:
                    if isinstance(field, dict):
                        field_name = field.get("fieldName", "")
                        field_alias = field.get("fieldAlias", "")
                        if field_name and field_alias:
                            all_names.add(f"{field_name} {field_alias}")
                        elif field_name:
                            all_names.add(field_name)
                        elif field_alias:
                            all_names.add(field_alias)
                    else:
                        if field:
                            all_names.add(str(field))
        
        return list(all_names)
    
    def _build_standard_queries(
        self, 
        names: List[str], 
        bounded_context: Dict,
        draft_options: Optional[List[Dict]] = None
    ) -> List[StandardQuery]:
        """
        이름들을 기반으로 단순 검색 쿼리 생성 (카테고리 구분 없음)
        
        Args:
            names: 추출된 모든 이름 리스트
            bounded_context: Bounded Context 정보
            draft_options: Aggregate Draft 옵션들 (선택사항)
            
        Returns:
            StandardQuery 리스트
        """
        queries: List[StandardQuery] = []
        
        # 도메인 힌트 추출
        domain_hint = bounded_context.get("domain")
        if not domain_hint:
            domain_hint = bounded_context.get("name") or bounded_context.get("alias")
        
        # 모든 이름을 쿼리로 변환 (카테고리 구분 없음)
        for name in names:
            keyword = name.strip()
            if not keyword:
                continue
            
            queries.append(StandardQuery(
                query=keyword,
                domain_hint=domain_hint
            ))
        
        return queries
    
    def _build_queries(self, names: List[str], bounded_context: Dict) -> List[str]:
        """
        이름들을 기반으로 검색 쿼리 생성 (기존 호환성 유지)
        
        Args:
            names: 추출된 이름들
            bounded_context: Bounded Context 정보
            
        Returns:
            쿼리 리스트
        """
        # 기존 방식 유지 (하위 호환성)
        queries = []
        bc_name = bounded_context.get("name", "")
        bc_alias = bounded_context.get("alias", "")
        
        for name in names:
            # 데이터베이스 표준 쿼리
            queries.append(f"{name} aggregate table naming standard")
            queries.append(f"{name} database naming convention")
            
            # API 표준 쿼리
            queries.append(f"{name} API endpoint naming standard")
            queries.append(f"{name} REST API naming convention")
            
            # 용어 표준 쿼리
            queries.append(f"{name} terminology standard")
            queries.append(f"{name} domain terminology")
        
        # Bounded Context 관련 쿼리
        if bc_name:
            queries.append(f"{bc_name} bounded context naming standard")
        if bc_alias:
            queries.append(f"{bc_alias} 도메인 표준 용어")
        
        return queries
    
    def _retrieve_relevant_standards(self, queries: List[str], 
                                   k_per_query: int = 5) -> List[Dict]:
        """
        쿼리들을 사용하여 관련 표준 검색 (기존 방식 - 하위 호환성 유지)
        
        Args:
            queries: 검색 쿼리들
            k_per_query: 쿼리당 반환할 결과 수
            
        Returns:
            검색된 표준 청크들 (중복 제거)
        """
        if not self.rag_retriever:
            return []
        
        all_results = []
        seen_result_keys = set()  # (source + chunk_index)로 중복 제거
        total_results_found = 0  # 검색 결과 총 개수 (중복 포함)
        failed_queries = 0
        processed_queries = 0  # 처리된 쿼리 수
        
        # Vector Store 초기화 상태 확인 (검색 전)
        vectorstore_initialized = False
        if self.rag_retriever:
            vectorstore_initialized = self.rag_retriever._initialized and self.rag_retriever.vectorstore is not None
        
        # 각 쿼리로 검색
        for idx, query in enumerate(queries):
            try:
                # 데이터베이스 표준 검색 (점수 필터링 포함, 임계값 0.3)
                # 유사도 0.3 이상 = 거리 0.7 이하 (충분히 관련 있는 문서)
                # similarity = 1 - distance 변환 사용
                db_results = self.rag_retriever.search_company_standards(
                    query, k=k_per_query, score_threshold=0.3
                )
                
                # API 표준 검색 (점수 필터링 포함, 임계값 0.3)
                api_results = self.rag_retriever.search_api_standards(
                    query, k=k_per_query, score_threshold=0.3
                )
                
                # 용어 표준 검색 (점수 필터링 포함, 임계값 0.3)
                term_results = self.rag_retriever.search_terminology_standards(
                    query, k=k_per_query, score_threshold=0.3
                )
                
                # 첫 번째 쿼리에서 Vector Store 초기화 실패 감지
                if idx == 0 and vectorstore_initialized and not db_results and not api_results and not term_results:
                    # Vector Store가 초기화되었다고 표시되어 있지만 검색 결과가 없음
                    # 실제로는 작동하지 않을 수 있음 (ChromaDB 데이터베이스 손상 등)
                    LoggingUtil.warning("StandardTransformer", 
                                      "⚠️  Vector Store 검색 결과 없음: 초기화 상태는 있으나 실제 작동하지 않을 수 있습니다. 다음 실행 시 재인덱싱 시도됩니다")
                
                # 검색 결과 수 집계 (중복 포함)
                total_found = len(db_results) + len(api_results) + len(term_results)
                total_results_found += total_found
                processed_queries += 1
                
                # 중복 제거: (source + chunk_index)로 판단
                for result in db_results + api_results + term_results:
                    # README 파일 제외
                    metadata = result.get("metadata", {})
                    source = metadata.get("source", "")
                    if source:
                        source_name = Path(source).name.lower()
                        if source_name.startswith("readme"):
                            continue
                    
                    # 중복 제거: source + chunk_index 조합
                    chunk_index = metadata.get("chunk_index", "")
                    result_key = f"{source}::{chunk_index}"
                    
                    if result_key not in seen_result_keys:
                        seen_result_keys.add(result_key)
                        result_with_tracking = result.copy()
                        result_with_tracking["_matched_queries"] = [query]
                        all_results.append(result_with_tracking)
                    else:
                        # 중복된 결과: 쿼리 목록에만 추가
                        for r in all_results:
                            r_metadata = r.get("metadata", {})
                            r_source = r_metadata.get("source", "")
                            r_chunk_index = r_metadata.get("chunk_index", "")
                            if f"{r_source}::{r_chunk_index}" == result_key:
                                if query not in r.get("_matched_queries", []):
                                    r["_matched_queries"].append(query)
                                break
            
            except Exception as e:
                failed_queries += 1
                continue
        
        return all_results
    
    def _retrieve_relevant_standards_with_categories(
        self, 
        standard_queries: List[StandardQuery],
        k_per_query: int = 3,  # top-k 3개 사용 (structure 단위 처리)
        transformation_session_id: Optional[str] = None
    ) -> tuple:
        """
        StandardQuery 리스트를 사용하여 관련 표준 검색 (카테고리 구분 없음)
        
        Args:
            standard_queries: 표준 검색 쿼리들
            k_per_query: 쿼리당 반환할 결과 수
            
        Returns:
            (검색된 표준 청크들, 쿼리별 검색 결과) 튜플
        """
        if not self.standard_service:
            # Fallback: 기존 방식 사용
            queries = [sq["query"] for sq in standard_queries]
            results = self._retrieve_relevant_standards(queries, k_per_query)
            return (results, [])
        
        all_results = []
        seen_result_keys = set()  # (source + chunk_index)로 중복 제거
        total_results_found = 0
        failed_queries = 0
        processed_queries = 0
        query_search_results = []  # 각 쿼리별 검색 결과 (summary용)
        
        # 각 쿼리로 검색 (카테고리 구분 없음)
        for sq in standard_queries:
            try:
                query = sq["query"]
                domain_hint = sq.get("domain_hint")
                
                # 단일 검색 메서드 사용 (카테고리 구분 없음)
                found = self.standard_service.search_standards(
                    query, domain_hint=domain_hint, k=k_per_query, transformation_session_id=transformation_session_id
                )
                
                total_found = len(found)
                
                total_results_found += total_found
                processed_queries += 1
                
                # 쿼리별 검색 결과 추적 (summary용)
                # top-k 3개 모두 저장 (LLM이 참고할 후보군)
                if found:
                    # 유사도 내림차순 정렬 후 top-3 선택
                    sorted_found = sorted(found, key=lambda x: x.get("score", 0.0), reverse=True)
                    top_results = sorted_found[:3]  # top-3
                    
                    # 각 결과의 structured_data 추출
                    results_list = []
                    for result_item in top_results:
                        metadata = result_item.get("metadata", {})
                        structured_data_str = metadata.get("structured_data", "")
                        structured_data_json = None
                        if structured_data_str:
                            try:
                                structured_data_json = json.loads(structured_data_str)
                            except:
                                pass
                        
                        results_list.append({
                            "similarity_score": result_item.get("score", 0.0),
                            "result": structured_data_json
                        })
                    
                    query_search_results.append({
                        "query": query,
                        "results": results_list  # top-3 결과 리스트
                    })
                else:
                    # 검색 결과가 없는 경우는 summary에 포함하지 않음
                    pass
                
                # 중복 제거 및 결과 변환
                for item in found:
                    metadata = item.get("metadata", {})
                    source = metadata.get("source", "")
                    
                    # README 파일 제외
                    if source:
                        source_name = Path(source).name.lower()
                        if source_name.startswith("readme"):
                            continue
                    
                    # 중복 제거: source + chunk_index 조합
                    chunk_index = metadata.get("chunk_index", "")
                    result_key = f"{source}::{chunk_index}"
                    
                    if result_key not in seen_result_keys:
                        seen_result_keys.add(result_key)
                        # 기존 형식으로 변환 (하위 호환성)
                        result_dict = {
                            "content": item["content"],
                            "metadata": item["metadata"],
                            "score": item.get("score"),
                            "_matched_queries": [query]
                        }
                        all_results.append(result_dict)
                    else:
                        # 중복된 결과: 쿼리 목록에만 추가
                        for r in all_results:
                            r_metadata = r.get("metadata", {})
                            r_source = r_metadata.get("source", "")
                            r_chunk_index = r_metadata.get("chunk_index", "")
                            if f"{r_source}::{r_chunk_index}" == result_key:
                                if query not in r.get("_matched_queries", []):
                                    r["_matched_queries"].append(query)
                                break
            
            except Exception as e:
                failed_queries += 1
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  검색 실패 (query: {sq.get('query')}): {e}")
                continue
        
        # 검색 결과 요약 로그: 채택된 표준 정보 표시
        if processed_queries > 0:
            # 중복 제거된 고유 결과 수와 각 쿼리별 채택 수
            unique_results = len(all_results)
            total_adopted = processed_queries  # 각 쿼리마다 top-1 채택
            LoggingUtil.info("StandardTransformer", 
                           f"✅ 유사도 검색 완료: {processed_queries}개 쿼리 처리, {unique_results}개 고유 표준 문서 채택 (중복 제거), {len(query_search_results)}개 쿼리별 결과 저장")
            # 채택된 표준 정보 상세 로그 (최대 10개만 표시)
            for i, result in enumerate(all_results[:10], 1):
                content = result.get("content", "")[:80]  # 최대 80자
                score = result.get("score", 0.0)
                queries = result.get("_matched_queries", [])
                query_count = len(queries)
                query_preview = ", ".join(queries[:3]) if queries else "N/A"
                if query_count > 3:
                    query_preview += f" 외 {query_count - 3}개"
                metadata = result.get("metadata", {})
                source = Path(metadata.get("source", "")).name if metadata.get("source") else "unknown"
                LoggingUtil.info("StandardTransformer", 
                               f"   [{i}] {query_count}개 쿼리 매칭 → 유사도 {score:.3f} ({content}...) [출처: {source}]")
            if len(all_results) > 10:
                LoggingUtil.info("StandardTransformer", 
                               f"   ... 외 {len(all_results) - 10}개 결과 생략")
            if failed_queries > 0:
                LoggingUtil.warning("StandardTransformer", f"⚠️ {failed_queries}개 쿼리 실패")
        
        # 쿼리별 검색 결과를 인스턴스 변수에 저장 (summary 생성 시 사용, 하위 호환성)
        self._last_query_search_results = query_search_results
        
        # 튜플로 반환: (relevant_standards, query_search_results)
        return (all_results, query_search_results)
    
    # ============================================================================
    # Standard Mapping Builder (Terminology/Standard Mapping 레이어)
    # ============================================================================
    
    def _build_global_standard_mapping_context(
        self, 
        draft_options: Optional[List[Dict]] = None,
        bounded_context: Optional[Dict] = None,
        transformation_session_id: Optional[str] = None,
        should_index_vectorstore: bool = False
    ) -> StandardMappingContext:
        """
        전체 표준 원본을 직접 읽어서 StandardMappingContext 생성
        (유사도 검색과 무관하게 필수 매핑을 구성)
        
        핵심 원칙:
        - 필수 매핑은 RAG 유사도 검색 결과와 무관하게 전체 표준 원본에서 직접 구성
        - RAG는 "참고용 컨텍스트 제공용"으로만 사용
        - 이렇게 하면 threshold를 올리든 내리든 핵심 매핑은 절대 안 깨짐
        
        Returns:
            StandardMappingContext: 매핑 사전
        """
        # 초기화
        mapping: StandardMappingContext = {
            "table": {
                "entity_to_table": {},
                "table_standards": {},  # 모든 표준 매핑 (카테고리 구분 없음)
                "column_standards": {},  # 모든 표준 매핑 (카테고리 구분 없음)
                "domain_to_tables": {}
            },
            "domain": {
                "name_to_domain": {},
                "table_to_domain": {}
            },
            "column": {
                "column_desc_by_table": {},
                "desc_to_columns": {}
            },
            "api": {
                "resource_abbrev": {},
                "action_to_path": {},
                "http_method_by_action": {}
            }
        }
        
        # HTTP Method 기본 매핑 (표준 규칙)
        mapping["api"]["http_method_by_action"] = {
            "생성": "POST",
            "조회": "GET",
            "수정": "PATCH",
            "삭제": "DELETE",
            "create": "POST",
            "read": "GET",
            "update": "PATCH",
            "delete": "DELETE"
        }
        
        # StandardLoader가 없으면 빈 매핑 반환
        if not self.standard_loader:
            LoggingUtil.warning("StandardTransformer", 
                              "⚠️  StandardLoader가 초기화되지 않았습니다. 빈 매핑을 반환합니다.")
            return mapping
        
        try:
            # 전체 표준 문서 로드 (엑셀 파일 직접 읽기)
            # 사용자별 문서가 있으면 우선 사용, 없으면 기본 경로 사용
            standards_path = self.user_standards_path if self.user_standards_path and self.user_standards_path.exists() else Config.COMPANY_STANDARDS_PATH
            
            if not standards_path.exists():
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  표준 문서 경로를 찾을 수 없습니다: {standards_path}")
                return mapping
            
            # 엑셀 파일 직접 읽기
            try:
                import pandas as pd
            except ImportError:
                LoggingUtil.warning("StandardTransformer", 
                                  "⚠️  pandas가 설치되지 않았습니다. 엑셀 파일을 읽을 수 없습니다.")
                return mapping
            
            processed_files = 0
            processed_rows = 0
            
            # 세션 레벨에서 should_index 결정 (모든 시트에 동일하게 적용)
            # 사용자별 Vector Store 인덱싱 상태 확인
            should_index = False
            if self.user_standards_path and self.user_standards_path.exists():
                # 사용자별 문서가 있고, 아직 인덱싱되지 않은 경우에만 인덱싱
                if should_index_vectorstore:
                    should_index = True
                    LoggingUtil.info("StandardTransformer", 
                                  f"📝 사용자별 문서 감지: 인덱싱 수행 (경로: {self.user_standards_path})")
                else:
                    LoggingUtil.info("StandardTransformer", 
                                  f"♻️  사용자별 Vector Store 재사용 (이미 인덱싱됨)")
            elif transformation_session_id:
                # 이 세션에서 아직 인덱싱하지 않았으면 인덱싱
                if transformation_session_id not in AggregateDraftStandardTransformer._base_standards_indexed:
                    should_index = True
                    AggregateDraftStandardTransformer._base_standards_indexed.add(transformation_session_id)
            else:
                # transformation_session_id가 없으면 항상 인덱싱 (기존 동작 유지)
                should_index = True
            
            # 인덱싱할 문서 수집 (모든 파일/시트 처리 후 한번에 인덱싱)
            all_documents_to_index = []
            
            # Excel 파일 처리
            for file_path in standards_path.rglob('*.xlsx'):
                if file_path.name.lower().startswith('readme'):
                    continue
                
                try:
                    excel_file = pd.ExcelFile(file_path)
                    
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        
                        if df.empty:
                            continue
                        
                        # 시트 이름이나 컬럼명에 의존하지 않고, row의 모든 값을 그대로 처리
                        
                        # 모든 문서는 company_standards 내용과 초안 정보를 함께 사용하여 인덱싱
                        # "기본 표준문서"와 "초안정보 포함 문서"를 따로 구분하지 않음
                        # 효율성: 세션당 한 번만 인덱싱 (같은 세션의 후속 BC는 스킵)
                        # 하지만 모든 시트는 처리해야 함 (첫 번째 시트에서만 인덱싱하는 것이 아님)
                        
                        # 디버깅: 시트 처리 로그
                        if should_index:
                            print(f"📋 [StandardTransformer] 시트 '{sheet_name}' 인덱싱 시작 (행 수: {len(df)})")
                        else:
                            print(f"⏭️  [StandardTransformer] 시트 '{sheet_name}' 인덱싱 스킵 (이미 인덱싱됨)")
                        
                        # 초안 정보 구성 (draft_context)
                        # 모든 문서는 초안 정보를 포함하여 semantic_text 생성
                        draft_context = None
                        if draft_options and bounded_context:
                            # 초안 Aggregate 정보 추출
                            aggregates = []
                            for option in draft_options:
                                structure = option.get("structure", [])
                                for item in structure:
                                    aggregate = item.get("aggregate", {})
                                    preview_attrs = item.get("previewAttributes", [])
                                    value_objects = item.get("valueObjects", [])
                                    
                                    aggregates.append({
                                        "alias": aggregate.get("alias", ""),
                                        "name": aggregate.get("name", ""),
                                        "previewAttributes": preview_attrs,
                                        "valueObjects": value_objects
                                    })
                            
                            draft_context = {
                                "bounded_context": bounded_context,
                                "aggregates": aggregates
                            }
                        
                        # 매핑 파싱 및 인덱싱 문서 수집
                        # 각 row마다 개별 Document로 인덱싱
                        rows_list = [row for _, row in df.iterrows()]
                        
                        if not rows_list:
                            continue
                        
                        # 모든 row를 인덱싱 (첫 번째 row도 데이터로 처리)
                        for row_idx, row in enumerate(rows_list):
                            # 각 row의 값들을 공백으로 연결하여 page_content 생성 (한글 포함)
                            row_values = []
                            structured_data_by_column = {}  # 인덱싱용 structured_data
                            
                            # row의 모든 값을 순회 (컬럼명 무관)
                            for col_name, val in row.items():
                                if pd.notna(val) and str(val).strip():
                                    val_str = str(val).strip()
                                    row_values.append(val_str)
                                    
                                    # structured_data: 컬럼명을 키로 하고, 현재 row의 값을 단일 값으로 저장 (인덱싱용)
                                    structured_data_by_column[col_name] = val_str
                            
                            # page_content: 현재 row의 값들을 공백으로 연결 (한글 포함)
                            page_content = " ".join(row_values) if row_values else ""
                            
                            processed_rows += 1
                            
                            # 매핑 파싱 (기존 로직 유지)
                            # 간단한 키워드 텍스트 생성 (매핑 파싱용)
                            text, structured_data = self.standard_loader._format_excel_row_as_standard_text(
                                row,
                                sheet_name,
                                draft_context=draft_context
                            )
                            
                            if structured_data:
                                # 파싱 (카테고리 구분 없음)
                                self._parse_single_row_data(structured_data, mapping)
                            
                            # 각 row마다 개별 Document 생성
                            if should_index and page_content:
                                doc_metadata = {
                                    "source": str(file_path),
                                    "sheet": sheet_name,
                                    "format": "excel",
                                    "structured_data": json.dumps(structured_data_by_column, ensure_ascii=False)
                                }
                                
                                doc = Document(
                                    page_content=page_content,
                                    metadata=doc_metadata
                                )
                                all_documents_to_index.append(doc)
                    
                    processed_files += 1
                
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", 
                                      f"⚠️  표준 파일 읽기 실패 ({file_path.name}): {e}")
                    continue
            
            # PowerPoint 파일 처리 (StandardLoader 사용)
            for file_path in standards_path.rglob('*.pptx'):
                if file_path.name.lower().startswith('readme'):
                    continue
                
                try:
                    # StandardLoader를 사용하여 PPT 파일 로드
                    ppt_documents = self.standard_loader._load_ppt(file_path)
                    
                    if should_index and ppt_documents:
                        # PPT 문서를 인덱싱용 Document로 변환
                        for doc in ppt_documents:
                            # 기존 메타데이터 유지하면서 인덱싱
                            all_documents_to_index.append(doc)
                            processed_rows += 1
                        
                        LoggingUtil.info("StandardTransformer", 
                                      f"✅ PowerPoint 파일 처리 완료: {file_path.name} ({len(ppt_documents)}개 슬라이드)")
                    
                    processed_files += 1
                
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", 
                                      f"⚠️  PowerPoint 파일 읽기 실패 ({file_path.name}): {e}")
                    continue
            
            # 모든 파일/시트 처리 후 한번에 인덱싱 (세션당 한 번만)
            if all_documents_to_index and self.rag_retriever and should_index:
                try:
                    # Vector Store 초기화 상태 확인 및 재초기화 시도
                    if not self.rag_retriever._initialized or not self.rag_retriever.vectorstore:
                        LoggingUtil.warning("StandardTransformer", 
                                          f"⚠️  Vector Store가 초기화되지 않음. 재초기화 시도...")
                        # 재초기화 시도
                        if hasattr(self.rag_retriever, 'vectorstore_path'):
                            try:
                                from src.project_generator.workflows.common.rag_retriever import RAGRetriever
                                self.rag_retriever = RAGRetriever(vectorstore_path=self.rag_retriever.vectorstore_path)
                                if self.rag_retriever._initialized and self.rag_retriever.vectorstore:
                                    LoggingUtil.info("StandardTransformer", "✅ Vector Store 재초기화 성공")
                                else:
                                    LoggingUtil.warning("StandardTransformer", 
                                                      f"⚠️  Vector Store 재초기화 실패: _initialized={self.rag_retriever._initialized}")
                            except Exception as reinit_e:
                                LoggingUtil.warning("StandardTransformer", 
                                                  f"⚠️  Vector Store 재초기화 중 오류: {reinit_e}")
                    
                    # 중복 체크 활성화하여 인덱싱
                    add_success = self.rag_retriever.add_documents(all_documents_to_index, check_duplicates=True)
                    if add_success:
                        LoggingUtil.info("StandardTransformer", 
                                       f"📝 표준 문서 {len(all_documents_to_index)}개 Vector Store에 인덱싱 완료")
                        
                        # Vector Store가 실제로 작동하는지 검증 (인덱싱된 문서의 실제 키워드로 검색)
                        # 검증은 선택적이며, 실패해도 인덱싱은 성공한 것으로 간주
                        # 검증은 디버깅 목적이며, 실제 사용 시 RAG 검색으로 확인됨
                        try:
                            # 검증은 선택적으로만 수행 (디버그 모드나 특정 조건에서만)
                            # 실제 검색 기능은 RAG 검색 시 확인되므로 여기서는 스킵
                            # 인덱싱이 성공했으면 Vector Store는 정상적으로 작동할 것으로 가정
                            LoggingUtil.info("StandardTransformer", 
                                           "ℹ️  Vector Store 검증 스킵: 인덱싱 성공 (검색 기능은 RAG 검색 시 확인됨)")
                        except Exception as verify_e:
                            # 검증 실패해도 인덱싱은 성공했으므로 경고만 출력
                            LoggingUtil.warning("StandardTransformer", 
                                              f"⚠️  Vector Store 검증 중 오류: {verify_e}. 인덱싱은 성공했으므로 계속 진행")
                            # 검증 실패는 치명적이지 않으므로 인덱싱 상태는 유지
                    else:
                        LoggingUtil.warning("StandardTransformer", 
                                          "⚠️  Vector Store 인덱싱 실패: add_documents가 False 반환")
                        # 인덱싱 상태 제거
                        if self.user_id and transformation_session_id:
                            user_key = (self.user_id, transformation_session_id)
                            AggregateDraftStandardTransformer._user_vectorstores_indexed.discard(user_key)
                        elif transformation_session_id:
                            AggregateDraftStandardTransformer._base_standards_indexed.discard(transformation_session_id)
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", 
                                      f"⚠️  Vector Store 인덱싱 중 오류 발생: {e}")
                    # 인덱싱 상태 제거
                    if self.user_id and transformation_session_id:
                        user_key = (self.user_id, transformation_session_id)
                        AggregateDraftStandardTransformer._user_vectorstores_indexed.discard(user_key)
                    elif transformation_session_id:
                        AggregateDraftStandardTransformer._base_standards_indexed.discard(transformation_session_id)
        
        except Exception as e:
            LoggingUtil.warning("StandardTransformer", 
                              f"⚠️  전체 표준 매핑 구성 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 로깅
        table_count = len(mapping["table"]["entity_to_table"])
        column_count = sum(len(cols) for cols in mapping["column"]["column_desc_by_table"].values())
        
        LoggingUtil.info("StandardTransformer", 
                       f"📋 표준 매핑 컨텍스트 생성: 테이블 {table_count}개, 필드 {column_count}개")
        
        return mapping
    
    def _build_standard_mapping_context(self, relevant_standards: List[Dict]) -> StandardMappingContext:
        """
        검색된 표준 문서들로부터 StandardMappingContext 생성
        
        Args:
            relevant_standards: RAG로 검색된 표준 청크들
            
        Returns:
            StandardMappingContext: 매핑 사전
        """
        # 초기화
        mapping: StandardMappingContext = {
            "table": {
                "entity_to_table": {},
                "table_standards": {},  # 모든 표준 매핑 (카테고리 구분 없음)
                "column_standards": {},  # 모든 표준 매핑 (카테고리 구분 없음)
                "domain_to_tables": {}
            },
            "domain": {
                "name_to_domain": {},
                "table_to_domain": {}
            },
            "column": {
                "column_desc_by_table": {},
                "desc_to_columns": {}
            },
            "api": {
                "resource_abbrev": {},
                "action_to_path": {},
                "http_method_by_action": {}
            }
        }
        
        # HTTP Method 기본 매핑 (표준 규칙)
        mapping["api"]["http_method_by_action"] = {
            "생성": "POST",
            "조회": "GET",
            "수정": "PATCH",
            "삭제": "DELETE",
            "create": "POST",
            "read": "GET",
            "update": "PATCH",
            "delete": "DELETE"
        }
        
        # 각 표준 문서 파싱
        parsed_count = 0
        skipped_count = 0
        
        for std in relevant_standards:
            metadata = std.get("metadata", {})
            structured_data_str = metadata.get("structured_data", "")
            
            if not structured_data_str:
                skipped_count += 1
                continue
            
            try:
                # structured_data는 인덱싱 방식에 따라 키가 다를 수 있음
                # 단일 값 형식: {"컬럼명1": "값1", "컬럼명2": "값2"} (단일 값)
                # 배열 형식: {"컬럼명1": ["값1", "값2"], "컬럼명2": ["값3", "값4"]} (배열)
                structured_data = json.loads(structured_data_str)
                
                if not isinstance(structured_data, dict):
                    skipped_count += 1
                    continue
                
                # 단일 값 형식인지 배열 형식인지 확인
                is_array_format = any(isinstance(v, list) for v in structured_data.values() if v)
                
                if is_array_format:
                    # 배열 형식: 각 컬럼의 배열 길이 중 최대값을 구하여 row 개수 파악
                    max_rows = max((len(v) for v in structured_data.values() if isinstance(v, list)), default=0)
                    
                    # 각 row를 재구성하여 파싱
                    for i in range(max_rows):
                        row_data = {}
                        for key, values in structured_data.items():
                            if isinstance(values, list) and i < len(values):
                                row_data[key] = values[i]
                        
                        if not row_data:
                            continue
                        
                        # 파싱 (카테고리 구분 없음)
                        self._parse_single_row_data(row_data, mapping)
                        parsed_count += 1
                else:
                    # 단일 값 형식: 바로 파싱 (카테고리 구분 없음)
                    self._parse_single_row_data(structured_data, mapping)
                    parsed_count += 1
            
            except json.JSONDecodeError as e:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  표준 JSON 파싱 실패: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  표준 매핑 추출 중 오류: {e}")
                skipped_count += 1
                continue
        
        # 로깅
        table_count = len(mapping["table"]["entity_to_table"])
        domain_count = len(mapping["domain"]["name_to_domain"])
        column_count = sum(len(cols) for cols in mapping["column"]["column_desc_by_table"].values())
        column_standards_count = len(mapping["table"]["column_standards"])
        api_count = len(mapping["api"]["resource_abbrev"])
        
        LoggingUtil.info("StandardTransformer", 
                       f"📋 StandardMappingContext 생성 완료: "
                       f"테이블 {table_count}개, 도메인 {domain_count}개, 컬럼 {column_count}개, API {api_count}개")
        LoggingUtil.info("StandardTransformer", 
                       f"   파싱: {parsed_count}개 성공, {skipped_count}개 스킵 (총 {len(relevant_standards)}개 표준 문서)")
        
        return mapping
    
    def _parse_single_row_data(self, row_data: Dict, mapping: StandardMappingContext):
        """단일 row 데이터를 파싱하여 매핑에 추가 (RAG 검색 결과용)
        
        Args:
            row_data: structured_data 딕셔너리 (값 패턴만으로 추론)
            mapping: 매핑 컨텍스트
        """
        # 키 이름을 하드코딩하지 않고, 값 패턴만으로 추론
        korean_name = None
        english_name = None
        standard_name = None
        
        # 모든 값을 순회하며 패턴으로 추론 (키 이름 무관)
        for key, value in row_data.items():
            if not value:
                continue
            
            # 값이 리스트인 경우 첫 번째 값 사용
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = value[0]
            
            val_str = str(value).strip()
            if not val_str:
                continue
            
            # 한글명: 한글이 포함된 값
            if not korean_name and any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in val_str):
                if len(val_str) <= 50:  # 너무 긴 값은 제외
                    korean_name = val_str
            
            # 영문명 또는 표준명: 영문만 포함된 값
            if val_str.isascii() and len(val_str) > 1:
                # 영문명: CamelCase 또는 일반 영문 (표준명보다 짧거나 CamelCase)
                if not english_name:
                    if '_' not in val_str and (val_str[0].isupper() or len(val_str) <= 20):
                        english_name = val_str
                
                # 표준명: snake_case 또는 언더스코어 포함 (영문명보다 긴 경우도 있음)
                if not standard_name:
                    if '_' in val_str or (len(val_str) > 5 and val_str.islower()):
                        standard_name = val_str
        
        # 모든 표준을 테이블명과 컬럼명 매핑에 모두 추가 (카테고리 구분 없음)
        if korean_name and standard_name:
            parsed_row = {
                "korean_name": korean_name,
                "english_name": english_name,
                "table_name": standard_name
            }
            # 테이블명 표준 파싱 (카테고리 구분 없이 모두 추가)
            self._parse_table_name_standard(parsed_row, mapping)
        
        # 컬럼명 표준 파싱 (카테고리 구분 없이 모두 추가)
        if korean_name or english_name:
            self._parse_column_name_standard(row_data, mapping)
        
        # API 표준 파싱: "/v" 패턴이 포함된 값이 있으면 파싱
        has_api_pattern = any(
            '/v' in str(val) and re.search(r'/v\d+/', str(val))
            for val in row_data.values() if val
        )
        if has_api_pattern:
            self._parse_api_standard(row_data, mapping)
        
        # 용어 표준 파싱: 짧은 약어(2-5자)가 있으면 파싱
        has_terminology = any(
            str(val).strip().isascii() and 2 <= len(str(val).strip()) <= 5 and str(val).strip().isupper()
            for val in row_data.values() if val
        )
        if has_terminology:
            self._parse_terminology_standard(row_data, mapping)
    
    def _parse_table_name_standard(self, row: Dict, mapping: StandardMappingContext):
        """테이블명 표준 파싱 (카테고리 구분 없음)
        
        값 패턴만으로 추론하여 매핑에 추가
        
        Args:
            row: 표준 행 데이터 (structured_data 딕셔너리)
            mapping: 매핑 컨텍스트
        """
        # 키 이름을 하드코딩하지 않고, 값 패턴만으로 추론
        entity_name = None
        table_name = None
        english_name = None
        domain = None
        
        # 모든 값을 순회하며 패턴으로 추론 (키 이름 무관)
        for key, value in row.items():
            if not value:
                continue
            
            # 값이 리스트인 경우 첫 번째 값 사용
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = value[0]
            
            val_str = str(value).strip()
            if not val_str:
                continue
            
            # entity_name: 한글이 포함된 값
            if not entity_name and any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in val_str):
                if len(val_str) <= 50:
                    entity_name = val_str
            
            # table_name: 영문 snake_case 또는 CamelCase (prefix 무관)
            if not table_name and val_str.isascii() and len(val_str) > 1:
                if '_' in val_str or (val_str[0].isupper() and any(c.islower() for c in val_str)):
                    table_name = val_str
            
            # english_name: 영문 CamelCase 또는 일반 영문 (snake_case가 아닌 경우)
            if not english_name and val_str.isascii() and len(val_str) > 1:
                if '_' not in val_str and (val_str[0].isupper() or len(val_str) <= 20):
                    english_name = val_str
            
            # domain: 짧은 영문 약어 (2-5자) 또는 한글 약어
            if not domain:
                if val_str.isascii() and 2 <= len(val_str) <= 5 and val_str.isupper():
                    domain = val_str
                elif any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in val_str) and len(val_str) <= 10:
                    domain = val_str
        
        if entity_name and table_name:
            # 한글/영문 모두 매핑 (대소문자 무시)
            entity_clean = str(entity_name).strip()
            table_clean = str(table_name).strip()
            
            if entity_clean and table_clean:
                # 모든 매핑에 추가 (카테고리 구분 없이 모두 추가)
                mapping["table"]["entity_to_table"][entity_clean] = table_clean
                mapping["table"]["table_standards"][entity_clean] = table_clean
                mapping["table"]["column_standards"][entity_clean] = table_clean
                
                # 영문명 매핑
                if english_name:
                    english_clean = str(english_name).strip()
                    mapping["table"]["entity_to_table"][english_clean] = table_clean
                    mapping["table"]["table_standards"][english_clean] = table_clean
                    mapping["table"]["column_standards"][english_clean] = table_clean
                    
                    mapping["table"]["entity_to_table"][english_clean.lower()] = table_clean
                    mapping["table"]["entity_to_table"][english_clean.upper()] = table_clean
                    mapping["table"]["table_standards"][english_clean.lower()] = table_clean
                    mapping["table"]["table_standards"][english_clean.upper()] = table_clean
                    mapping["table"]["column_standards"][english_clean.lower()] = table_clean
                    mapping["table"]["column_standards"][english_clean.upper()] = table_clean
                # entity_name이 영문인 경우도 처리
                elif entity_clean.isascii():
                    mapping["table"]["entity_to_table"][entity_clean.lower()] = table_clean
                    mapping["table"]["entity_to_table"][entity_clean.upper()] = table_clean
                    mapping["table"]["table_standards"][entity_clean.lower()] = table_clean
                    mapping["table"]["table_standards"][entity_clean.upper()] = table_clean
                    mapping["table"]["column_standards"][entity_clean.lower()] = table_clean
                    mapping["table"]["column_standards"][entity_clean.upper()] = table_clean
        
        # domain -> table_name 그룹
        if domain and table_name:
            if domain not in mapping["table"]["domain_to_tables"]:
                mapping["table"]["domain_to_tables"][domain] = []
            if table_name not in mapping["table"]["domain_to_tables"][domain]:
                mapping["table"]["domain_to_tables"][domain].append(table_name)
            
            # table_name -> domain 역매핑
            mapping["domain"]["table_to_domain"][table_name] = domain
    
    def _parse_column_name_standard(self, row: Dict, mapping: StandardMappingContext):
        """컬럼명 표준 파싱
        
        structured_data의 키를 직접 사용하여 추출
        """
        # 키 이름을 하드코딩하지 않고, 값 패턴만으로 추론
        entity_name = None
        standard_name = None
        english_name = None
        
        # 모든 값을 순회하며 패턴으로 추론 (키 이름 무관)
        for key, value in row.items():
            if not value:
                continue
            
            # 값이 리스트인 경우 첫 번째 값 사용
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = value[0]
            
            val_str = str(value).strip()
            if not val_str:
                continue
            
            # entity_name: 한글이 포함된 값
            if not entity_name and any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in val_str):
                if len(val_str) <= 50:
                    entity_name = val_str
            
            # standard_name: 영문 snake_case (언더스코어 포함)
            if not standard_name and val_str.isascii() and len(val_str) > 1:
                if '_' in val_str:
                    standard_name = val_str
            
            # english_name: 영문 CamelCase 또는 일반 영문 (snake_case가 아닌 경우)
            if not english_name and val_str.isascii() and len(val_str) > 1:
                if '_' not in val_str and (val_str[0].isupper() or len(val_str) <= 20):
                    english_name = val_str
        
        # column_standards에 매핑 추가 (필드명 변환에 사용)
        if entity_name and standard_name:
            entity_clean = str(entity_name).strip()
            standard_clean = str(standard_name).strip()
            
            if entity_clean and standard_clean:
                # 한글명 -> 표준명 매핑
                mapping["table"]["column_standards"][entity_clean] = standard_clean
        
        if english_name and standard_name:
            english_clean = str(english_name).strip()
            standard_clean = str(standard_name).strip()
            
            if english_clean and standard_clean:
                # 영문명 -> 표준명 매핑 (대소문자 변형 포함)
                mapping["table"]["column_standards"][english_clean] = standard_clean
                mapping["table"]["column_standards"][english_clean.lower()] = standard_clean
                mapping["table"]["column_standards"][english_clean.upper()] = standard_clean
        
        # column_desc_by_table에 추가 (로깅용, 실제 변환에는 사용 안 함)
        if entity_name and standard_name:
            # table_name은 표준명 그대로 사용
            table_name = standard_name
            if table_name:
                if table_name not in mapping["column"]["column_desc_by_table"]:
                    mapping["column"]["column_desc_by_table"][table_name] = {}
                
                column_name = english_name or entity_name
                if column_name:
                    description = entity_name if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in str(entity_name)) else None
                    mapping["column"]["column_desc_by_table"][table_name][column_name] = description or ""
                    
                    # 역방향 매핑 (설명 -> 컬럼명)
                    if description:
                        if description not in mapping["column"]["desc_to_columns"]:
                            mapping["column"]["desc_to_columns"][description] = []
                        if column_name not in mapping["column"]["desc_to_columns"][description]:
                            mapping["column"]["desc_to_columns"][description].append(column_name)
    
    def _parse_api_standard(self, row: Dict, mapping: StandardMappingContext):
        """API 표준 파싱
        
        컬럼명에 전혀 의존하지 않고 값 패턴만으로 추출
        """
        api_format = None
        description = None
        
        # 모든 값을 순회하며 패턴 기반으로 추출 (컬럼명 무관)
        for col_name, col_value in row.items():
            if not col_value or not str(col_value).strip():
                continue
            
            col_value_str = str(col_value).strip()
            
            # api_format: "/v1/..." 패턴이 포함된 값
            if not api_format:
                if '/v' in col_value_str and re.search(r'/v\d+/', col_value_str):
                    api_format = col_value_str
            
            # description: 한글이 포함된 긴 설명 또는 "→" 패턴이 포함된 값
            if not description:
                if '→' in col_value_str or '->' in col_value_str:
                    description = col_value_str
                elif any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in col_value_str) and len(col_value_str) > 20:
                    description = col_value_str
        
        # API 경로 패턴 파싱 (예: "/v1/odr, /v1/dly, /v1/cst")
        if api_format:
            # "/v1/odr" -> "odr" 추출
            paths = re.findall(r'/v\d+/([^,\s]+)', api_format)
            for path in paths:
                # path는 "odr", "dly" 등
                # description에서 "Order→odr" 같은 패턴 찾기
                if description:
                    # "Order→odr" 또는 "Order -> odr" 패턴 찾기
                    patterns = re.findall(r'(\w+)\s*[→->]\s*(\w+)', description, re.IGNORECASE)
                    for concept, abbrev in patterns:
                        if abbrev.lower() == path.lower():
                            mapping["api"]["resource_abbrev"][concept] = abbrev
                            mapping["api"]["resource_abbrev"][concept.lower()] = abbrev
                            mapping["api"]["resource_abbrev"][concept.upper()] = abbrev
        
        # description에서 직접 매핑 추출
        if description:
            # "Order→odr, Delivery→dly" 같은 패턴
            patterns = re.findall(r'(\w+)\s*[→->]\s*(\w+)', description, re.IGNORECASE)
            for concept, abbrev in patterns:
                mapping["api"]["resource_abbrev"][concept] = abbrev
                mapping["api"]["resource_abbrev"][concept.lower()] = abbrev
                mapping["api"]["resource_abbrev"][concept.upper()] = abbrev
    
    def _parse_terminology_standard(self, row: Dict, mapping: StandardMappingContext):
        """용어 표준 파싱 (도메인 코드 매핑)
        
        컬럼명에 전혀 의존하지 않고 값 패턴만으로 추출
        """
        terminology = None
        entity_name = None
        table_name = None
        
        # 모든 값을 순회하며 패턴 기반으로 추출 (컬럼명 무관)
        for col_name, col_value in row.items():
            if not col_value or not str(col_value).strip():
                continue
            
            col_value_str = str(col_value).strip()
            
            # terminology: 짧은 영문 약어 (2-5자, 대문자) 또는 한글 약어
            if not terminology:
                if col_value_str.isascii() and 2 <= len(col_value_str) <= 5 and col_value_str.isupper():
                    terminology = col_value_str
                elif any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in col_value_str) and len(col_value_str) <= 10:
                    terminology = col_value_str
            
            # entity_name: 한글이 포함된 값
            if not entity_name:
                if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in col_value_str):
                    if len(col_value_str) <= 50:
                        entity_name = col_value_str
            
            # table_name: 영문 snake_case 또는 CamelCase (prefix 무관)
            if not table_name and col_value_str.isascii() and len(col_value_str) > 1:
                if '_' in col_value_str or (col_value_str[0].isupper() and any(c.islower() for c in col_value_str)):
                    table_name = col_value_str
        
        # 용어 표준에도 테이블명 정보가 있을 수 있음 (도메인표준 시트)
        if entity_name and table_name:
            entity_clean = str(entity_name).strip()
            table_clean = str(table_name).strip()
            
            if entity_clean and table_clean:
                # 테이블명 매핑도 추가
                mapping["table"]["entity_to_table"][entity_clean] = table_clean
                if entity_clean.isascii():
                    mapping["table"]["entity_to_table"][entity_clean.lower()] = table_clean
                    mapping["table"]["entity_to_table"][entity_clean.upper()] = table_clean
        
        if terminology and entity_name:
            # entity_name (한글/영문) -> terminology (도메인 코드)
            mapping["domain"]["name_to_domain"][entity_name] = terminology
            # 영문명도 매핑
            if entity_name.isascii():
                mapping["domain"]["name_to_domain"][entity_name.lower()] = terminology
                mapping["domain"]["name_to_domain"][entity_name.upper()] = terminology
    
    def _count_fields(self, draft_options: List[Dict]) -> List[str]:
        """필드명 목록 추출 (변환 전/후 비교용)"""
        field_names = []
        for option in draft_options:
            structure = option.get("structure", [])
            for item in structure:
                preview_attrs = item.get("previewAttributes", [])
                for attr in preview_attrs:
                    if isinstance(attr, dict):
                        field_name = attr.get("fieldName", "")
                        if field_name:
                            field_names.append(field_name)
                ddl_fields = item.get("ddlFields", [])
                for field in ddl_fields:
                    if isinstance(field, dict):
                        field_name = field.get("fieldName", "")
                        if field_name:
                            field_names.append(field_name)
        return field_names
    
    def _apply_standard_mappings(self, draft_options: List[Dict], 
                                 mapping: StandardMappingContext) -> List[Dict]:
        """
        StandardMappingContext를 사용하여 draft_options에 deterministic 룰 적용
        
        Args:
            draft_options: 원본 옵션들
            mapping: StandardMappingContext
            
        Returns:
            매핑이 적용된 옵션들 (새로운 복사본)
        """
        import copy
        mapped_options = copy.deepcopy(draft_options)
        
        applied_count = 0
        
        for option in mapped_options:
            structure = option.get("structure", [])
            
            for item in structure:
                aggregate = item.get("aggregate", {})
                alias = aggregate.get("alias", "")  # 한글 이름 (예: "주문 마스터", "쿠폰 마스터")
                name = aggregate.get("name", "")  # 영문 이름 (예: "Order", "Coupon")
                
                # ============================================================
                # 각 aggregate에 대해 모든 변환 작업을 순차적으로 수행
                # (테이블명, Enum, VO, 필드명은 독립적인 작업이므로 모두 실행)
                # ============================================================
                
                # 1. Aggregate 테이블명 치환: alias -> table_name (정확 매칭)
                # ✅ table_standards 사용 (m_ prefix만)
                if alias:
                    alias_clean = str(alias).strip()
                    if alias_clean in mapping["table"]["table_standards"]:
                        new_table_name = mapping["table"]["table_standards"][alias_clean]
                        if aggregate.get("name") != new_table_name:
                            aggregate["name"] = new_table_name
                            applied_count += 1
                
                # 2. 영문명으로도 테이블명 찾기 (name -> table_name)
                if name:
                    name_variants = [name, name.lower(), name.upper(), name.capitalize()]
                    for variant in name_variants:
                        if variant in mapping["table"]["table_standards"]:
                            new_table_name = mapping["table"]["table_standards"][variant]
                            if aggregate.get("name") != new_table_name:
                                aggregate["name"] = new_table_name
                                applied_count += 1
                            break
                
                # 3. Enumeration 선처리 (한글 alias 또는 영문 name 매칭)
                # ✅ table_standards 사용 (m_ prefix만, fld_ 자동 제외됨)
                enumerations = item.get("enumerations", [])
                for enum_item in enumerations:
                    enum_alias = enum_item.get("alias", "")
                    enum_name = enum_item.get("name", "")
                    
                    # alias (한글) 매칭 시도
                    if enum_alias:
                        enum_alias_clean = str(enum_alias).strip()
                        if enum_alias_clean in mapping["table"]["table_standards"]:
                            new_enum_name = mapping["table"]["table_standards"][enum_alias_clean]
                            if enum_item.get("name") != new_enum_name:
                                enum_item["name"] = new_enum_name
                                applied_count += 1
                                continue  # 이 continue는 enum_item 루프 내부이므로 다음 enum으로 넘어감 (정상)
                    
                    # name (영문) 매칭 시도
                    if enum_name:
                        name_variants = [enum_name, enum_name.lower(), enum_name.upper(), enum_name.capitalize()]
                        for variant in name_variants:
                            if variant in mapping["table"]["table_standards"]:
                                new_enum_name = mapping["table"]["table_standards"][variant]
                                if enum_item.get("name") != new_enum_name:
                                    enum_item["name"] = new_enum_name
                                    applied_count += 1
                                break
                
                # 4. ValueObject 선처리 (한글 alias 또는 영문 name 매칭)
                # ✅ table_standards 사용 (m_ prefix만, fld_ 자동 제외됨)
                value_objects = item.get("valueObjects", [])
                for vo_item in value_objects:
                    vo_alias = vo_item.get("alias", "")
                    vo_name = vo_item.get("name", "")
                    
                    # alias (한글) 매칭 시도
                    if vo_alias:
                        vo_alias_clean = str(vo_alias).strip()
                        if vo_alias_clean in mapping["table"]["table_standards"]:
                            new_vo_name = mapping["table"]["table_standards"][vo_alias_clean]
                            if vo_item.get("name") != new_vo_name:
                                vo_item["name"] = new_vo_name
                                applied_count += 1
                                continue  # 이 continue는 vo_item 루프 내부이므로 다음 vo로 넘어감 (정상)
                    
                    # name (영문) 매칭 시도
                    if vo_name:
                        name_variants = [vo_name, vo_name.lower(), vo_name.upper(), vo_name.capitalize()]
                        for variant in name_variants:
                            if variant in mapping["table"]["table_standards"]:
                                new_vo_name = mapping["table"]["table_standards"][variant]
                                if vo_item.get("name") != new_vo_name:
                                    vo_item["name"] = new_vo_name
                                    applied_count += 1
                                break
                
                # 5. 필드명 치환: previewAttributes, ddlFields
                # ✅ column_standards 사용
                # ⚠️ 중요: 테이블명 변환과 무관하게 항상 실행되어야 함
                preview_attrs = item.get("previewAttributes", [])
                
                for attr in preview_attrs:
                    if isinstance(attr, dict):
                        field_name = attr.get("fieldName", "")
                        if field_name:
                            field_name_clean = str(field_name).strip()
                            original_field_name = field_name_clean
                            
                            # 매핑 확인
                            if field_name_clean in mapping["table"]["column_standards"]:
                                new_field_name = mapping["table"]["column_standards"][field_name_clean]
                                attr["fieldName"] = new_field_name
                                applied_count += 1
                            else:
                                # 영문 필드명의 변형도 시도 (camelCase -> snake_case 변환 포함)
                                field_name_variants = [
                                    field_name_clean.lower(),
                                    field_name_clean.upper(),
                                    field_name_clean
                                ]
                                
                                # camelCase를 snake_case로 변환한 변형도 추가
                                import re
                                snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name_clean).lower()
                                if snake_case != field_name_clean.lower():
                                    field_name_variants.append(snake_case)
                                
                                matched = False
                                for variant in field_name_variants:
                                    if variant in mapping["table"]["column_standards"]:
                                        new_field_name = mapping["table"]["column_standards"][variant]
                                        attr["fieldName"] = new_field_name
                                        applied_count += 1
                                        matched = True
                                        break
                
                # 6. DDL 필드명 치환
                # ✅ column_standards 사용
                ddl_fields = item.get("ddlFields", [])
                for field in ddl_fields:
                    if isinstance(field, dict):
                        field_name = field.get("fieldName", "")
                        if field_name:
                            field_name_clean = str(field_name).strip()
                            
                            # 영문 필드명 매핑 시도
                            if field_name_clean in mapping["table"]["column_standards"]:
                                new_field_name = mapping["table"]["column_standards"][field_name_clean]
                                field["fieldName"] = new_field_name
                                applied_count += 1
                            else:
                                # 영문 필드명의 변형도 시도 (camelCase -> snake_case 변환 포함)
                                field_name_variants = [
                                    field_name_clean.lower(),
                                    field_name_clean.upper(),
                                    field_name_clean
                                ]
                                
                                # camelCase를 snake_case로 변환한 변형도 추가
                                import re
                                snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name_clean).lower()
                                if snake_case != field_name_clean.lower():
                                    field_name_variants.append(snake_case)
                                
                                matched = False
                                for variant in field_name_variants:
                                    if variant in mapping["table"]["column_standards"]:
                                        new_field_name = mapping["table"]["column_standards"][variant]
                                        field["fieldName"] = new_field_name
                                        applied_count += 1
                                        matched = True
                                        break
        
        if applied_count > 0:
            LoggingUtil.info("StandardTransformer", 
                           f"✅ 선처리 매핑 완료: {applied_count}개 항목 변환됨")
        else:
            LoggingUtil.info("StandardTransformer", 
                           "ℹ️  선처리 매핑: 매칭되는 룰 없음 (LLM 처리)")
        
        return mapped_options
    
    def _strip_unnecessary_fields_for_llm(self, draft_options: List[Dict]) -> List[Dict]:
        """
        LLM 요청 전에 네이밍 변환과 관련 없는 필드 제거 (토큰 절약)
        
        제거할 필드:
        - refs: 추적성 정보 (LLM 변환에 불필요)
        - description, pros, cons: 변환 대상이 아님
        - ddlFields: 전체 DDL 정보 (필요시 previewAttributes로 충분)
        
        유지할 필드:
        - name, alias: 변환 대상
        - structure, aggregate, enumerations, valueObjects: 구조 정보
        - previewAttributes: 필드명 변환용 (간소화 가능)
        """
        import copy
        stripped_options = []
        
        for option in draft_options:
            stripped = {}
            
            # boundedContext: aggregates만 유지 (description 제거)
            if "boundedContext" in option:
                bc = option["boundedContext"]
                stripped["boundedContext"] = {
                    "name": bc.get("name"),
                    "alias": bc.get("alias"),
                    "aggregates": []
                }
                for agg in bc.get("aggregates", []):
                    stripped["boundedContext"]["aggregates"].append({
                        "name": agg.get("name"),
                        "alias": agg.get("alias")
                        # refs 제거
                    })
            
            # structure: 네이밍 정보만 유지
            if "structure" in option:
                stripped["structure"] = []
                for item in option["structure"]:
                    stripped_item = {}
                    
                    # aggregate
                    if "aggregate" in item:
                        agg = item["aggregate"]
                        stripped_item["aggregate"] = {
                            "name": agg.get("name"),
                            "alias": agg.get("alias")
                            # refs 제거
                        }
                    
                    # enumerations
                    if "enumerations" in item:
                        stripped_item["enumerations"] = []
                        for enum in item["enumerations"]:
                            stripped_item["enumerations"].append({
                                "name": enum.get("name"),
                                "alias": enum.get("alias")
                                # refs 제거
                            })
                    
                    # valueObjects
                    if "valueObjects" in item:
                        stripped_item["valueObjects"] = []
                        for vo in item["valueObjects"]:
                            stripped_vo = {
                                "name": vo.get("name"),
                                "alias": vo.get("alias")
                                # refs 제거
                            }
                            if "referencedAggregateName" in vo:
                                stripped_vo["referencedAggregateName"] = vo["referencedAggregateName"]
                            stripped_item["valueObjects"].append(stripped_vo)
                    
                    # previewAttributes: fieldName과 fieldAlias 유지 (alias는 변환 대상이 아니므로 보존)
                    if "previewAttributes" in item:
                        stripped_item["previewAttributes"] = []
                        for attr in item["previewAttributes"]:
                            if isinstance(attr, dict):
                                stripped_attr = {
                                    "fieldName": attr.get("fieldName")
                                    # refs 제거
                                }
                                # fieldAlias는 변환 대상이 아니므로 보존
                                if "fieldAlias" in attr:
                                    stripped_attr["fieldAlias"] = attr.get("fieldAlias")
                                stripped_item["previewAttributes"].append(stripped_attr)
                            else:
                                stripped_item["previewAttributes"].append(attr)
                    
                    # ddlFields: fieldName과 fieldAlias 유지 (alias는 변환 대상이 아니므로 보존)
                    if "ddlFields" in item:
                        stripped_item["ddlFields"] = []
                        for field in item["ddlFields"]:
                            if isinstance(field, dict):
                                # 최소 정보만 유지
                                stripped_field = {
                                    "fieldName": field.get("fieldName"),
                                    "className": field.get("className")
                                }
                                if "type" in field:
                                    stripped_field["type"] = field["type"]
                                # fieldAlias는 변환 대상이 아니므로 보존
                                if "fieldAlias" in field:
                                    stripped_field["fieldAlias"] = field.get("fieldAlias")
                                stripped_item["ddlFields"].append(stripped_field)
                            else:
                                stripped_item["ddlFields"].append(field)
                    
                    stripped["structure"].append(stripped_item)
            
            # description, pros, cons 제거 (LLM이 변환할 필요 없음)
            
            stripped_options.append(stripped)
        
        return stripped_options
    
    def _transform_single_structure_with_llm(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        original_structure_item: Optional[Dict] = None,
        mapping_context: Optional[StandardMappingContext] = None,
        update_progress_callback: Optional[callable] = None,
        skip_chunking: bool = False,  # 청킹 내부 호출 시 무한 재귀 방지
        bc_name: Optional[str] = None,
        agg_name: Optional[str] = None,
        original_option_bounded_context: Optional[Dict] = None  # 원본 option의 boundedContext 정보 (청킹 처리용)
    ) -> Dict:
        """
        단일 structure(aggregate)를 LLM으로 변환
        
        Args:
            structure_item: 변환할 단일 structure 항목
            bounded_context: Bounded Context 정보
            relevant_standards: 검색된 표준 청크들
            query_search_results: 쿼리별 검색 결과 (top-k=3)
            original_structure_item: 원본 structure 항목 (선처리 전)
            mapping_context: 선처리 매핑 컨텍스트
            
        Returns:
            변환된 structure 항목
        """
        # bc_name과 agg_name이 없으면 추출
        if not bc_name:
            bc_name = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
        if not agg_name:
            agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
        
        # 단일 structure를 옵션 형식으로 래핑
        # boundedContext 정보도 포함 (청킹 처리 및 복원 시 참조용)
        single_structure_option = {
            "structure": [structure_item]
        }
        # 원본 option의 boundedContext 정보 우선 사용 (있는 경우)
        # 없으면 bounded_context 파라미터에서 기본 정보 구성
        if original_option_bounded_context:
            # 원본 option의 boundedContext 정보 사용 (청킹 처리를 위해 전체 정보 포함)
            # 단, _strip_unnecessary_fields_for_llm에서 필요한 필드만 유지됨
            single_structure_option["boundedContext"] = original_option_bounded_context
        elif bounded_context:
            # boundedContext의 name, alias, aggregates만 포함 (나머지는 복원 시 처리)
            bc_info = {
                "name": bounded_context.get("name"),
                "alias": bounded_context.get("alias"),
                "aggregates": bounded_context.get("aggregates", [])
            }
            single_structure_option["boundedContext"] = bc_info
        draft_options = [single_structure_option]
        
        # 불필요한 필드 제거
        stripped_draft_options = self._strip_unnecessary_fields_for_llm(draft_options)
        
        # 디버깅: structure 정보 확인
        if stripped_draft_options and len(stripped_draft_options) > 0:
            structure = stripped_draft_options[0].get("structure", [])
            if structure and len(structure) > 0:
                first_structure = structure[0]
                preview_attrs = first_structure.get("previewAttributes", [])
                ddl_fields = first_structure.get("ddlFields", [])
                LoggingUtil.info("StandardTransformer", 
                               f"      📋 Structure 정보: previewAttributes={len(preview_attrs)}개, ddlFields={len(ddl_fields)}개")
                if preview_attrs:
                    field_names = [attr.get("fieldName", "") for attr in preview_attrs[:5] if isinstance(attr, dict)]
                    LoggingUtil.info("StandardTransformer", 
                                   f"      📋 previewAttributes 필드 예시 (최대 5개): {', '.join(field_names)}")
                if ddl_fields:
                    ddl_field_names = [field.get("fieldName", "") for field in ddl_fields[:5] if isinstance(field, dict)]
                    LoggingUtil.info("StandardTransformer", 
                                   f"      📋 ddlFields 필드 예시 (최대 5개): {', '.join(ddl_field_names)}")
        
        # 디버깅: 쿼리 결과 확인 (필드 관련)
        if query_search_results:
            field_queries = []
            for qr in query_search_results:
                query = qr.get("query", "")
                # 필드 관련 쿼리인지 확인 (fieldName 패턴 또는 일반적인 필드명 패턴)
                if any(keyword in query.lower() for keyword in ["_id", "_fee", "_amount", "_status", "_at", "id", "fee", "amount", "status"]):
                    field_queries.append(query)
            if field_queries:
                LoggingUtil.info("StandardTransformer", 
                               f"      📋 필드 관련 쿼리: {len(field_queries)}개 (예: {', '.join(field_queries[:5])})")
        
        # 청킹 필요 여부 사전 판단 (프롬프트 생성 전)
        preview_attrs = structure_item.get("previewAttributes", [])
        ddl_fields = structure_item.get("ddlFields", [])
        enumerations = structure_item.get("enumerations", [])
        value_objects = structure_item.get("valueObjects", [])
        total_fields = len(preview_attrs) + len(ddl_fields)
        total_items = len(enumerations) + len(value_objects) + total_fields
        query_count = len(query_search_results) if query_search_results else 0
        
        # 전체 프롬프트 토큰 수 추정
        # 1. 검색 결과 토큰: 각 쿼리 결과당 평균 500 토큰 (이제 top-1만 사용하므로)
        estimated_query_tokens = query_count * 500  # 쿼리당 평균 500 토큰 추정 (top-1 기준)
        # 2. 아이템 정보 토큰: 각 아이템(enum/vo/field)당 평균 100 토큰
        estimated_items_tokens = total_items * 100
        # 3. 프롬프트 템플릿 및 컨텍스트: 약 2000 토큰
        estimated_template_tokens = 2000
        # 4. Aggregate 정보: 약 200 토큰
        estimated_agg_tokens = 200
        # 전체 예상 프롬프트 토큰
        estimated_prompt_tokens_total = estimated_query_tokens + estimated_items_tokens + estimated_template_tokens + estimated_agg_tokens
        
        # 청킹 필요 여부: 
        # 1. 전체 예상 프롬프트 토큰이 15000 이상이거나
        # 2. 필드+enum+vo가 10개 이상이거나
        # 3. 검색 결과 쿼리가 8개 이상인 경우
        # 단, skip_chunking이 True이면 청킹 건너뛰기 (무한 재귀 방지)
        should_chunk = not skip_chunking and (estimated_prompt_tokens_total > 15000 or total_items > 10 or query_count > 8)
        
        LoggingUtil.info("StandardTransformer", 
                       f"      📊 청킹 판단: 예상 프롬프트 토큰={estimated_prompt_tokens_total} (쿼리={estimated_query_tokens}, 아이템={estimated_items_tokens}, 템플릿={estimated_template_tokens}), 필드={total_fields}, enum={len(enumerations)}, vo={len(value_objects)}, 쿼리={query_count} → 청킹={'필요' if should_chunk else '불필요'}")
        
        if should_chunk:
            LoggingUtil.info("StandardTransformer", 
                           f"      📦 청킹 처리 필요: 예상 프롬프트 토큰={estimated_prompt_tokens_total}, 필드={total_fields}, enum={len(enumerations)}, vo={len(value_objects)}")
            # bc_name과 agg_name이 없으면 structure_item에서 추출
            if not bc_name:
                bc_name = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
            if not agg_name:
                agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
            return self._transform_structure_with_chunking(
                structure_item=structure_item,
                bounded_context=bounded_context,
                relevant_standards=relevant_standards,
                query_search_results=query_search_results,
                original_structure_item=original_structure_item,
                mapping_context=mapping_context,
                update_progress_callback=update_progress_callback,
                estimated_prompt_tokens=estimated_prompt_tokens_total,
                bc_name=bc_name,
                agg_name=agg_name,
                original_option_bounded_context=original_option_bounded_context
            )
        
        # 프롬프트 구성 (단일 structure용 - 청킹 불필요한 경우만)
        prompt = self._build_transformation_prompt(
            draft_options=stripped_draft_options,
            bounded_context=bounded_context,
            relevant_standards=relevant_standards,
            query_search_results=query_search_results
        )
        
        # LLM 호출 (기존 방식 - 청킹 불필요)
        try:
            agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
            LoggingUtil.info("StandardTransformer", 
                           f"      📤 LLM API 호출 시작...")
            
            # 진행 상황 업데이트: LLM 변환 시작
            if update_progress_callback:
                try:
                    update_progress_callback(0, f"LLM 변환 중: {agg_name}",
                                            bc_name=bc_name,
                                            agg_name=agg_name,
                                            property_type="aggregate",
                                            status="processing")
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
            
            # 타임아웃을 늘리고 재시도 로직 추가
            max_retries = 2
            retry_count = 0
            response = None
            
            while retry_count <= max_retries:
                try:
                    response = self.llm_structured.invoke(prompt)
                    break
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # 진행 상황 업데이트: 재시도 중
                    if update_progress_callback and retry_count <= max_retries:
                        try:
                            if "length limit" in error_msg or "completion_tokens=32768" in error_msg:
                                update_progress_callback(0, f"⚠️ LLM 응답 길이 초과 - 재시도 중 ({retry_count}/{max_retries}): {agg_name}",
                                                        bc_name=bc_name,
                                                        agg_name=agg_name,
                                                        property_type="aggregate",
                                                        status="processing",
                                                        error_message=f"응답 길이 초과 (재시도 {retry_count}/{max_retries})")
                            else:
                                update_progress_callback(0, f"⚠️ LLM 호출 실패 - 재시도 중 ({retry_count}/{max_retries}): {agg_name}",
                                                        bc_name=bc_name,
                                                        agg_name=agg_name,
                                                        property_type="aggregate",
                                                        status="processing",
                                                        error_message=f"LLM 호출 실패 (재시도 {retry_count}/{max_retries})")
                        except Exception as update_e:
                            LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                    
                    LoggingUtil.warning("StandardTransformer", 
                                      f"      ⚠️  LLM API 호출 실패 (재시도 {retry_count}/{max_retries}): {e}")
                    
                    if retry_count > max_retries:
                        # 진행 상황 업데이트: 최종 실패
                        if update_progress_callback:
                            try:
                                update_progress_callback(0, f"❌ LLM 변환 실패: {agg_name} (최대 재시도 횟수 초과)",
                                                        bc_name=bc_name,
                                                        agg_name=agg_name,
                                                        property_type="aggregate",
                                                        status="error",
                                                        error_message="최대 재시도 횟수 초과")
                            except Exception as update_e:
                                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                        raise
                    
                    import time
                    time.sleep(2)  # 2초 대기 후 재시도
            
            LoggingUtil.info("StandardTransformer", 
                           f"      📥 LLM API 응답 수신 완료")
            result = response.get("result", {})
            transformed_options = result.get("transformedOptions", [])
            
            if not transformed_options or len(transformed_options) == 0:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  LLM 응답에 transformedOptions가 없음, 원본 반환")
                return structure_item
            
            transformed_option = transformed_options[0]
            if not isinstance(transformed_option, dict):
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  transformedOption이 dict가 아님: {type(transformed_option)}, 원본 반환")
                return structure_item
            
            transformed_structure = transformed_option.get("structure", [])
            
            if not transformed_structure or len(transformed_structure) == 0:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  transformedStructure가 비어있음, 원본 반환")
                return structure_item
            
            if not isinstance(transformed_structure, list):
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  transformedStructure가 list가 아님: {type(transformed_structure)}, 원본 반환")
                return structure_item
            
            transformed_item = transformed_structure[0]
            if not isinstance(transformed_item, dict):
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  transformedItem이 dict가 아님: {type(transformed_item)}, 원본 반환")
                return structure_item
            
            # 🔒 CRITICAL: original_structure_item에서 원본 구조 가져오기 (refs 포함)
            import copy
            if original_structure_item:
                # 원본 structure_item 사용 (refs 포함)
                merged_item = copy.deepcopy(original_structure_item)
            else:
                # original_structure_item이 없으면 현재 structure_item 사용
                merged_item = copy.deepcopy(structure_item)
            
            # Aggregate 이름 변환
            if "aggregate" in transformed_item and "aggregate" in merged_item:
                trans_agg = transformed_item["aggregate"]
                merged_agg = merged_item["aggregate"]
                # 🔒 CRITICAL: refs 보존
                preserved_agg_refs = merged_agg.get("refs", []) if isinstance(merged_agg, dict) else []
                if "name" in trans_agg:
                    merged_agg["name"] = trans_agg["name"]
                if "alias" in trans_agg:
                    merged_agg["alias"] = trans_agg["alias"]
                # refs 복원 (혹시 모를 경우를 대비)
                if isinstance(merged_agg, dict) and "refs" not in merged_agg:
                    merged_agg["refs"] = preserved_agg_refs
            
            # Enumeration 이름 변환
            if "enumerations" in transformed_item and "enumerations" in merged_item:
                trans_enums = transformed_item["enumerations"]
                merged_enums = merged_item["enumerations"]
                for i, trans_enum in enumerate(trans_enums):
                    if i < len(merged_enums):
                        # 🔒 CRITICAL: refs 보존 (original_structure_item에서 이미 복원됨)
                        preserved_refs = merged_enums[i].get("refs", []) if isinstance(merged_enums[i], dict) else []
                        if "name" in trans_enum:
                            merged_enums[i]["name"] = trans_enum["name"]
                        if "alias" in trans_enum:
                            merged_enums[i]["alias"] = trans_enum["alias"]
                        # refs 복원 (혹시 모를 경우를 대비)
                        if isinstance(merged_enums[i], dict) and "refs" not in merged_enums[i]:
                            merged_enums[i]["refs"] = preserved_refs
            
            # ValueObject 이름 변환
            if "valueObjects" in transformed_item and "valueObjects" in merged_item:
                trans_vos = transformed_item["valueObjects"]
                merged_vos = merged_item["valueObjects"]
                for i, trans_vo in enumerate(trans_vos):
                    if i < len(merged_vos):
                        # 🔒 CRITICAL: refs 보존 (original_structure_item에서 이미 복원됨)
                        preserved_refs = merged_vos[i].get("refs", []) if isinstance(merged_vos[i], dict) else []
                        if "name" in trans_vo:
                            merged_vos[i]["name"] = trans_vo["name"]
                        if "alias" in trans_vo:
                            merged_vos[i]["alias"] = trans_vo["alias"]
                        if "referencedAggregateName" in trans_vo:
                            merged_vos[i]["referencedAggregateName"] = trans_vo["referencedAggregateName"]
                        # refs 복원 (혹시 모를 경우를 대비)
                        if isinstance(merged_vos[i], dict) and "refs" not in merged_vos[i]:
                            merged_vos[i]["refs"] = preserved_refs
            
            # previewAttributes 필드명 변환 (fieldAlias는 원본 보존)
            if "previewAttributes" in transformed_item and "previewAttributes" in merged_item:
                trans_attrs = transformed_item["previewAttributes"]
                merged_attrs = merged_item["previewAttributes"]
                for i, trans_attr in enumerate(trans_attrs):
                    if i < len(merged_attrs) and isinstance(trans_attr, dict) and isinstance(merged_attrs[i], dict):
                        # 🔒 CRITICAL: refs 보존 (original_structure_item에서 이미 복원됨)
                        # original_structure_item에서 온 merged_item은 이미 refs를 포함하고 있음
                        # 하지만 혹시 모를 경우를 대비해 명시적으로 보존
                        preserved_refs = merged_attrs[i].get("refs", [])
                        # fieldName만 변환 (LLM 결과 사용)
                        if "fieldName" in trans_attr:
                            merged_attrs[i]["fieldName"] = trans_attr["fieldName"]
                        # fieldAlias는 원본 보존 (LLM이 변환하지 않으므로 원본 유지)
                        # LLM 결과에 fieldAlias가 있어도 원본을 우선 (alias는 변환 대상이 아님)
                        # 🔒 CRITICAL: refs 명시적으로 보존 (빈 배열도 보존)
                        merged_attrs[i]["refs"] = preserved_refs
            
            # ddlFields 필드명 변환 (fieldAlias는 원본 보존)
            if "ddlFields" in transformed_item and "ddlFields" in merged_item:
                trans_ddl = transformed_item["ddlFields"]
                merged_ddl = merged_item["ddlFields"]
                for i, trans_field in enumerate(trans_ddl):
                    if i < len(merged_ddl) and isinstance(trans_field, dict) and isinstance(merged_ddl[i], dict):
                        # 🔒 CRITICAL: refs 보존 (original_structure_item에서 이미 복원됨)
                        # original_structure_item에서 온 merged_item은 이미 refs를 포함하고 있음
                        # 하지만 혹시 모를 경우를 대비해 명시적으로 보존
                        preserved_refs = merged_ddl[i].get("refs", [])
                        # fieldName만 변환 (LLM 결과 사용)
                        if "fieldName" in trans_field:
                            merged_ddl[i]["fieldName"] = trans_field["fieldName"]
                        # fieldAlias는 원본 보존 (LLM이 변환하지 않으므로 원본 유지)
                        # LLM 결과에 fieldAlias가 있어도 원본을 우선 (alias는 변환 대상이 아님)
                        # 🔒 CRITICAL: refs 명시적으로 보존 (빈 배열도 보존)
                        merged_ddl[i]["refs"] = preserved_refs
            
            return merged_item
            
        except Exception as e:
            import time
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            LoggingUtil.error("StandardTransformer", 
                            f"❌ 단일 structure LLM 변환 실패 (소요 시간: {elapsed_time:.2f}초): {e}")
            import traceback
            traceback.print_exc()
            return structure_item
    
    def _transform_with_llm(self, draft_options: List[Dict], 
                           bounded_context: Dict,
                           relevant_standards: List[Dict],
                           query_search_results: Optional[List[Dict]] = None,
                           original_draft_options: Optional[List[Dict]] = None) -> List[Dict]:
        """
        LLM을 사용하여 표준에 맞게 변환
        
        Args:
            draft_options: 원본 옵션들
            bounded_context: Bounded Context 정보
            relevant_standards: 검색된 표준 청크들
            
        Returns:
            변환된 옵션들
        """
        # LLM 요청 전: 불필요한 필드 제거 (토큰 절약)
        stripped_draft_options = self._strip_unnecessary_fields_for_llm(draft_options)
        
        LoggingUtil.info("StandardTransformer", "✂️  불필요한 필드 제거 완료 (refs, description, pros, cons, ddlFields)")
        
        # 프롬프트 구성 (정리된 데이터 사용)
        prompt = self._build_transformation_prompt(
            draft_options=stripped_draft_options,  # 정리된 데이터 사용
            bounded_context=bounded_context,
            relevant_standards=relevant_standards,
            query_search_results=query_search_results or []  # 쿼리별 검색 결과 추가
        )
        
        # LLM 호출
        try:
            response = self.llm_structured.invoke(prompt)
            
            result = response.get("result", {})
            transformed_options = result.get("transformedOptions", [])
            
            if not transformed_options:
                LoggingUtil.warning("StandardTransformer", 
                                  "⚠️  transformedOptions가 비어있습니다!")
                return draft_options
            
            # 옵션 수 검증 제거: 각 structure별로 처리하므로 옵션 수가 달라질 수 있음
            
            # 🔒 구조 보존 전략: 원본 구조를 유지하고 변환된 이름만 덮어쓰기
            # LLM 출력을 참고하되, 원본 구조를 절대 변경하지 않음
            merged_options = []
            for i, transformed_option in enumerate(transformed_options):
                if i < len(draft_options):
                    import copy
                    # 원본을 deep copy (구조 100% 보존)
                    original_option = draft_options[i]
                    merged_option = copy.deepcopy(original_option)
                    
                    original_structure = original_option.get("structure", [])
                    transformed_structure = transformed_option.get("structure", [])
                    
                    # 🔒 구조 보존: LLM 결과에서 변환된 이름만 추출하여 원본에 덮어쓰기
                    # merged_option은 이미 원본의 deep copy
                    result_structure = merged_option["structure"]
                    
                    # Aggregate 이름 매핑 생성 (변환 전 → 변환 후)
                    # ✅ 선처리 전 원본과 선처리 후(현재)를 비교해서 매핑 정보 추출
                    aggregate_name_mapping = {}
                    preprocessing_mapping = {}  # 선처리 매핑만 추적
                    llm_aggregate_mapping = {}  # LLM이 변환한 aggregate 이름 매핑
                    
                    # 선처리 매핑 정보 추출 (original_draft_options가 있으면)
                    if original_draft_options and i < len(original_draft_options):
                        original_opt_structure = original_draft_options[i].get("structure", [])
                        current_opt_structure = draft_options[i].get("structure", [])  # 선처리 후
                        
                        # alias 기반 매칭으로 선처리 매핑 추출
                        for orig_item in original_opt_structure:
                            orig_alias = orig_item.get("aggregate", {}).get("alias")
                            orig_name = orig_item.get("aggregate", {}).get("name")
                            
                            for curr_item in current_opt_structure:
                                curr_alias = curr_item.get("aggregate", {}).get("alias")
                                curr_name = curr_item.get("aggregate", {}).get("name")
                                
                                if orig_alias == curr_alias and orig_name and curr_name and orig_name != curr_name:
                                    # 선처리에서 변환된 매핑 (예: "Customer" → "m_cst")
                                    preprocessing_mapping[orig_name] = curr_name
                                    aggregate_name_mapping[orig_name] = curr_name
                                    # 로그 간소화: 선처리 매핑 로그 제거
                                    break
                    
                    # 🔧 CRITICAL FIX: 먼저 모든 aggregate의 LLM 변환 결과를 수집
                    # 그래야 나중에 Enum/VO 변환 시 모든 aggregate 매핑을 사용할 수 있음
                    # ⚠️ 중요: original_draft_options에서 원본 이름을 가져와야 함!
                    if original_draft_options and i < len(original_draft_options):
                        original_opt_structure = original_draft_options[i].get("structure", [])
                        for trans_item in transformed_structure:
                            trans_agg_alias = trans_item.get("aggregate", {}).get("alias")
                            if not trans_agg_alias:
                                continue
                            
                            # original_draft_options에서 원본 이름 찾기 (선처리 전!)
                            for orig_item in original_opt_structure:
                                if orig_item.get("aggregate", {}).get("alias") == trans_agg_alias:
                                    orig_agg_name = orig_item.get("aggregate", {}).get("name")  # 원본 이름 (예: "Customer")
                                    trans_agg_name = trans_item.get("aggregate", {}).get("name")  # LLM 결과 (예: "m_cst")
                                    
                                    # LLM이 변환했는지 확인 (선처리 후 이름과 비교)
                                    # result_structure에서 선처리 후 이름 가져오기
                                    current_agg_name = None
                                    for curr_item in result_structure:
                                        if curr_item.get("aggregate", {}).get("alias") == trans_agg_alias:
                                            current_agg_name = curr_item.get("aggregate", {}).get("name")  # 선처리 후 (예: "m_cst")
                                            break
                                    
                                    # LLM이 변환했으면 (trans_agg_name != current_agg_name) 또는
                                    # 선처리 매핑이 이미 있으면 aggregate_name_mapping에 추가
                                    if orig_agg_name:
                                        if trans_agg_name and trans_agg_name != current_agg_name:
                                            # LLM이 추가 변환함 (선처리 매핑이 없거나 다름)
                                            llm_aggregate_mapping[orig_agg_name] = trans_agg_name
                                        elif orig_agg_name in preprocessing_mapping:
                                            # 선처리 매핑이 이미 있음 (그대로 사용)
                                            pass
                                        else:
                                            # LLM이 변환하지 않았고 선처리 매핑도 없으면 원본 유지
                                            llm_aggregate_mapping[orig_agg_name] = orig_agg_name
                                    break
                    
                    # LLM 매핑을 aggregate_name_mapping에 병합 (선처리 매핑보다 우선)
                    for orig_name, new_name in llm_aggregate_mapping.items():
                        aggregate_name_mapping[orig_name] = new_name
                    
                    # 매핑 정보를 merged_option에 저장 (summary 생성 시 사용)
                    if "mapping_info" not in merged_option:
                        merged_option["mapping_info"] = {}
                    merged_option["mapping_info"][f"option_{i}"] = {
                        "preprocessing_mapping": preprocessing_mapping,
                        "llm_mapping": llm_aggregate_mapping
                    }
                    
                    # 🔧 BC 간 참조를 위한 전역 매핑에 현재 BC의 매핑 추가
                    self._global_aggregate_name_mapping.update(aggregate_name_mapping)
                    
                    # 🔧 enum/VO 변환 시 전역 매핑 사용 (다른 BC의 aggregate 참조 가능)
                    # 전역 매핑과 현재 BC 매핑을 병합 (전역 매핑이 우선, 현재 BC 매핑으로 보완)
                    combined_aggregate_mapping = {**aggregate_name_mapping, **self._global_aggregate_name_mapping}
                    
                    # 로그 간소화: 매핑 수집 로그 제거
                    
                    # 🔧 필드명 매핑 추적: 선처리에서 변환된 필드명 추적
                    # original_draft_options (선처리 전) vs draft_options (선처리 후) 비교
                    field_name_mapping = {}  # {aggregate_alias: {original_field: transformed_field}}
                    if original_draft_options and i < len(original_draft_options):
                        original_opt_structure = original_draft_options[i].get("structure", [])
                        current_opt_structure = draft_options[i].get("structure", [])  # 선처리 후
                        
                        for orig_item in original_opt_structure:
                            orig_alias = orig_item.get("aggregate", {}).get("alias")
                            if not orig_alias:
                                continue
                            
                            # 같은 alias를 가진 현재 항목 찾기
                            for curr_item in current_opt_structure:
                                curr_alias = curr_item.get("aggregate", {}).get("alias")
                                if orig_alias != curr_alias:
                                    continue
                                
                                # previewAttributes 필드명 매핑 추적
                                if orig_alias not in field_name_mapping:
                                    field_name_mapping[orig_alias] = {}
                                
                                orig_attrs = orig_item.get("previewAttributes", [])
                                curr_attrs = curr_item.get("previewAttributes", [])
                                for attr_idx in range(min(len(orig_attrs), len(curr_attrs))):
                                    if isinstance(orig_attrs[attr_idx], dict) and isinstance(curr_attrs[attr_idx], dict):
                                        orig_field = orig_attrs[attr_idx].get("fieldName")
                                        curr_field = curr_attrs[attr_idx].get("fieldName")
                                        if orig_field and curr_field and orig_field != curr_field:
                                            field_name_mapping[orig_alias][orig_field] = curr_field
                                
                                # ddlFields 필드명 매핑 추적
                                orig_ddl_fields = orig_item.get("ddlFields", [])
                                curr_ddl_fields = curr_item.get("ddlFields", [])
                                for ddl_idx in range(min(len(orig_ddl_fields), len(curr_ddl_fields))):
                                    if isinstance(orig_ddl_fields[ddl_idx], dict) and isinstance(curr_ddl_fields[ddl_idx], dict):
                                        orig_field = orig_ddl_fields[ddl_idx].get("fieldName")
                                        curr_field = curr_ddl_fields[ddl_idx].get("fieldName")
                                        if orig_field and curr_field and orig_field != curr_field:
                                            field_name_mapping[orig_alias][orig_field] = curr_field
                                
                                break
                    
                    # 로그 간소화: 필드 매핑 수집 로그 제거
                    
                    # 🔒 CRITICAL: alias 기반 매칭으로 refs 보존 보장
                    # LLM이 순서를 바꾸더라도 alias로 올바른 항목을 찾아서 매칭
                    # 이렇게 해야 refs가 올바른 Aggregate에 유지됨!
                    
                    # 2단계: Aggregate 이름 덮어쓰기 및 Enum/VO 처리
                    # ⚠️ CRITICAL FIX: result_structure를 기준으로 루프를 돌아야 모든 aggregate를 처리할 수 있음!
                    # LLM이 일부만 반환해도 모든 aggregate의 Enum/VO를 처리해야 함
                    
                    # result_structure를 기준으로 루프 (모든 aggregate 처리 보장)
                    for orig_item in result_structure:
                        orig_agg_alias = orig_item.get("aggregate", {}).get("alias")
                        
                        if not orig_agg_alias:
                            continue
                        
                        # transformed_structure에서 같은 alias를 가진 항목 찾기
                        trans_item = None
                        for item in transformed_structure:
                            if item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                trans_item = item
                                break
                        
                        # trans_item이 없어도 Enum/VO 처리는 진행 (aggregate_name_mapping 사용)
                        
                        # 1. Aggregate 이름 덮어쓰기 (alias로 매칭했으므로 refs는 올바른 위치에 유지!)
                        if "aggregate" in orig_item and trans_item and "aggregate" in trans_item:
                            orig_agg = orig_item["aggregate"]
                            trans_agg = trans_item["aggregate"]
                            
                            # 🔒 CRITICAL: alias 검증 및 보호 (LLM이 실수로 바꿀 수 있음)
                            orig_agg_alias = orig_agg.get("alias")
                            trans_agg_alias = trans_agg.get("alias")
                            if trans_agg_alias and trans_agg_alias != orig_agg_alias:
                                LoggingUtil.warning("StandardTransformer", 
                                                  f"   [경고] LLM이 alias를 변경 시도: '{orig_agg_alias}' → '{trans_agg_alias}' (원본 유지)")
                                # 원본 alias로 복구
                                trans_agg["alias"] = orig_agg_alias
                            
                            # 이름만 덮어쓰기
                            orig_agg_name = orig_agg.get("name")
                            trans_agg_name = trans_agg.get("name")
                            if trans_agg_name:
                                orig_agg["name"] = trans_agg_name
                                # aggregate_name_mapping은 이미 위에서 수집했으므로 여기서는 로그만 출력
                                # 로그 간소화: 이름변환 로그 제거 (summary에서 확인 가능)
                                pass
                        
                        # 🔧 Deterministic VO/Enum 변환: Aggregate 이름이 변환되면 자동으로 prefix 처리
                        # LLM에만 의존하지 않고 백엔드에서 강제 처리
                        if "aggregate" not in orig_item:
                            LoggingUtil.warning("StandardTransformer", 
                                              f"   [경고] orig_item에 aggregate가 없음 - Enum/VO 처리 건너뜀")
                            continue
                        
                        orig_agg = orig_item["aggregate"]
                        current_agg_name = orig_agg.get("name")  # 변환된 이름 (이미 덮어써진 후)
                        
                        # 원본 aggregate 이름 복원 (aggregate_name_mapping에서 역매핑)
                        # ⚠️ original_draft_options에서 직접 가져오기 (더 확실함)
                        original_agg_name = None
                        if original_draft_options and i < len(original_draft_options):
                            original_opt_structure = original_draft_options[i].get("structure", [])
                            for orig_opt_item in original_opt_structure:
                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                    original_agg_name = orig_opt_item.get("aggregate", {}).get("name")
                                    break
                        
                        # 역매핑으로도 시도 (fallback)
                        if not original_agg_name:
                            for orig_name, new_name in aggregate_name_mapping.items():
                                if new_name == current_agg_name:
                                    original_agg_name = orig_name
                                    break
                        
                        # 로그 간소화: Enum/VO 처리 상세 로그 제거
                        pass
                        
                        # 2. Enumeration 이름 덮어쓰기 (alias 매칭으로 refs 보존)
                        orig_enums = orig_item.get("enumerations", [])
                        trans_enums = trans_item.get("enumerations", []) if trans_item else []
                        
                        # 🔧 원본 Enum 이름 및 refs 가져오기 (original_draft_options에서)
                        original_enum_names = {}  # {alias: original_name}
                        original_enum_refs = {}  # {alias: refs}
                        if original_draft_options and i < len(original_draft_options):
                            original_opt_structure = original_draft_options[i].get("structure", [])
                            for orig_opt_item in original_opt_structure:
                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                    for orig_opt_enum in orig_opt_item.get("enumerations", []):
                                        enum_alias = orig_opt_enum.get("alias")
                                        enum_name = orig_opt_enum.get("name")
                                        enum_refs = orig_opt_enum.get("refs", [])
                                        if enum_alias and enum_name:
                                            original_enum_names[enum_alias] = enum_name
                                            original_enum_refs[enum_alias] = enum_refs
                                    break
                        
                        # 🔧 Deterministic Enum 변환: Aggregate prefix 자동 적용
                        # ✅ 모든 BC의 aggregate 이름 매핑 확인 (다른 BC 참조도 처리)
                        for orig_enum in orig_enums:
                            orig_enum_alias = orig_enum.get("alias")
                            orig_enum_name = orig_enum.get("name")
                            # 원본 이름 사용 (선처리 전)
                            original_enum_name = original_enum_names.get(orig_enum_alias, orig_enum_name)
                            
                            # LLM 결과에서 매칭되는 것 찾기
                            trans_enum_name = None
                            for trans_enum in trans_enums:
                                if trans_enum.get("alias") == orig_enum_alias:
                                    trans_enum_name = trans_enum.get("name")
                                    # 로그 간소화: LLM결과 로그 제거
                                    pass
                                    break
                            
                            # LLM 결과가 있으면 사용, 없거나 잘못되었으면 자동 생성
                            # ✅ LLM 결과 검증: aggregate 이름과 관련이 있는지 확인 (하드코딩된 prefix 체크 제거)
                            new_name = None
                            
                            if trans_enum_name:
                                # 🔧 CRITICAL FIX: LLM 결과가 aggregate 이름과 관련이 있는지 검증
                                # Enum 이름이 aggregate 이름으로 시작하는지 확인
                                is_valid_llm_result = False
                                
                                # 1. 현재 BC의 aggregate와 관련이 있는지 확인
                                if original_agg_name and original_enum_name and original_enum_name.startswith(original_agg_name):
                                    is_valid_llm_result = True
                                else:
                                    # 2. 다른 BC의 aggregate와 관련이 있는지 확인
                                    for mapped_orig_name, mapped_new_name in combined_aggregate_mapping.items():
                                        if original_enum_name.startswith(mapped_orig_name) and mapped_orig_name != original_enum_name:
                                            is_valid_llm_result = True
                                            break
                                
                                if is_valid_llm_result:
                                    # LLM이 올바르게 변환한 경우 (aggregate와 관련 있음)
                                    new_name = trans_enum_name
                                    # 로그 간소화: LLM채택 로그 제거
                                    pass
                                else:
                                    # LLM이 잘못 변환한 경우 (aggregate와 무관함)
                                    # 로그 간소화: LLM무시 로그 제거 (경고는 유지하되 간소화)
                                    pass
                            
                            # LLM 결과가 없거나 무효한 경우 자동 생성 시도
                            if not new_name:
                                # 자동 생성: aggregate_name + "_" + suffix
                                # 예: OrderStatus → m_odr_status, CartStatus → m_bkt_status
                                
                                # 1. 현재 aggregate 이름 확인 (변환 전 원본 이름 사용!)
                                # ⚠️ original_enum_name 사용 (선처리 전 원본 이름)
                                if original_agg_name and original_enum_name and original_enum_name.startswith(original_agg_name):
                                    # 현재 BC의 aggregate가 변환된 경우
                                    # 예: "OrderStatus"는 "Order"로 시작 → "Status" 추출
                                    suffix = original_enum_name[len(original_agg_name):]
                                    import re
                                    suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                    new_name = current_agg_name + "_" + suffix_snake if suffix_snake else current_agg_name + "_enum"
                                    # 로그 간소화: 자동변환 로그 제거
                                    pass
                                    
                                    # 🔧 Enum 내부 필드 변환 (Enum이 변환되면 내부 필드도 변환)
                                    # Enum 내부에 필드가 있는 경우 (예: previewAttributes, ddlFields)
                                    if "previewAttributes" in orig_enum or "ddlFields" in orig_enum:
                                        # Enum 내부 필드도 aggregate 이름에 따라 변환
                                        enum_preview_attrs = orig_enum.get("previewAttributes", [])
                                        for enum_attr in enum_preview_attrs:
                                            if isinstance(enum_attr, dict):
                                                enum_field_name = enum_attr.get("fieldName", "")
                                                if enum_field_name:
                                                    # aggregate 이름으로 시작하는 필드명 변환
                                                    # 예: "customerStatus" → 변환된 aggregate 이름 + "_status"
                                                    if original_agg_name and enum_field_name.startswith(original_agg_name.lower()):
                                                        # aggregate 이름으로 시작하는 필드
                                                        suffix = enum_field_name[len(original_agg_name.lower()):]
                                                        # m_ prefix를 fld_로 변경 (표준명 형식에 맞춤)
                                                        new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix if suffix else current_agg_name.replace("m_", "fld_")
                                                        enum_attr["fieldName"] = new_field_name
                                                        LoggingUtil.info("StandardTransformer", 
                                                                       f"   [Enum필드변환] Enum '{orig_enum_alias}' 내부 필드: '{enum_field_name}' → '{new_field_name}'")
                                        
                                        enum_ddl_fields = orig_enum.get("ddlFields", [])
                                        for enum_ddl_field in enum_ddl_fields:
                                            if isinstance(enum_ddl_field, dict):
                                                enum_field_name = enum_ddl_field.get("fieldName", "")
                                                if enum_field_name:
                                                    if original_agg_name and enum_field_name.startswith(original_agg_name.lower()):
                                                        suffix = enum_field_name[len(original_agg_name.lower()):]
                                                        new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix if suffix else current_agg_name.replace("m_", "fld_")
                                                        enum_ddl_field["fieldName"] = new_field_name
                                                        LoggingUtil.info("StandardTransformer", 
                                                                       f"   [Enum필드변환] Enum '{orig_enum_alias}' 내부 DDL필드: '{enum_field_name}' → '{new_field_name}'")
                                
                                # 2. 다른 BC의 aggregate 이름 확인 (참조 관계) - original_agg_name이 없어도 실행
                                if (not new_name or new_name == orig_enum_name) and original_enum_name:
                                    # 🔧 combined_aggregate_mapping 사용 (전역 매핑 포함, 다른 BC 참조 가능)
                                    for mapped_orig_name, mapped_new_name in combined_aggregate_mapping.items():
                                        # Enum 이름이 aggregate 이름으로 시작하는지 확인
                                        # 예: "OrderStatus"는 "Order"로 시작, "CartStatus"는 "Cart"로 시작
                                        # ⚠️ original_enum_name 사용 (선처리 전 원본 이름)
                                        if original_enum_name.startswith(mapped_orig_name) and mapped_orig_name != original_enum_name:
                                            # 포함된 aggregate 이름을 변환된 이름으로 교체
                                            suffix = original_enum_name[len(mapped_orig_name):]  # 나머지 부분
                                            import re
                                            # snake_case 변환
                                            suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                            new_name = mapped_new_name + "_" + suffix_snake if suffix_snake else mapped_new_name + "_enum"
                                            # 로그 간소화: 참조변환 로그 제거
                                            break
                                
                                # 3. 매핑이 없으면 원본 유지 (aggregate와 무관한 독립적인 Enum)
                                if not new_name:
                                    new_name = original_enum_name  # 원본 이름 유지
                                    # 로그 간소화: 원본유지 로그 제거
                            
                            if new_name and orig_enum_name != new_name:
                                # 🔒 CRITICAL: refs 보존 (빈 배열도 보존)
                                preserved_refs = orig_enum.get("refs", [])
                                # original_draft_options에서 원본 refs 가져오기
                                if orig_enum_alias in original_enum_refs:
                                    preserved_refs = original_enum_refs[orig_enum_alias]
                                
                                orig_enum["name"] = new_name
                                # refs 복구 (빈 배열도 보존)
                                if "refs" not in orig_enum or not orig_enum.get("refs"):
                                    orig_enum["refs"] = preserved_refs
                                # 로그 간소화: 이름변환 로그 제거
                        
                        # 🔧 Aggregate 필드명 변환: aggregate 이름이 변환되면 관련 필드명도 변환
                        # 예: Customer → m_cst이면, customerStatus → fld_cst_status, customerId → fld_cst_id
                        if original_agg_name and current_agg_name and original_agg_name != current_agg_name:
                            import re
                            
                            # PreviewAttributes 필드명 변환
                            orig_attrs = orig_item.get("previewAttributes", [])
                            for attr in orig_attrs:
                                if isinstance(attr, dict):
                                    field_name = attr.get("fieldName", "")
                                    if field_name:
                                        # aggregate 이름으로 시작하는 필드명 변환
                                        # 예: customerStatus → fld_cst_status, customerId → fld_cst_id
                                        field_lower = field_name.lower()
                                        original_agg_lower = original_agg_name.lower()
                                        
                                        if field_lower.startswith(original_agg_lower):
                                            # aggregate 이름으로 시작하는 필드
                                            suffix = field_name[len(original_agg_name):]
                                            # camelCase를 snake_case로 변환
                                            suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                            new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix_snake if suffix_snake else current_agg_name.replace("m_", "fld_")
                                            attr["fieldName"] = new_field_name
                                            # 로그 간소화: Agg필드변환 로그 제거
                            
                            # DDLFields 필드명 변환
                            orig_ddl_fields = orig_item.get("ddlFields", [])
                            for ddl_field in orig_ddl_fields:
                                if isinstance(ddl_field, dict):
                                    field_name = ddl_field.get("fieldName", "")
                                    if field_name:
                                        # aggregate 이름으로 시작하는 필드명 변환
                                        field_lower = field_name.lower()
                                        original_agg_lower = original_agg_name.lower()
                                        
                                        if field_lower.startswith(original_agg_lower):
                                            # aggregate 이름으로 시작하는 필드
                                            suffix = field_name[len(original_agg_name):]
                                            # camelCase를 snake_case로 변환
                                            suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                            new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix_snake if suffix_snake else current_agg_name.replace("m_", "fld_")
                                            ddl_field["fieldName"] = new_field_name
                                            # 로그 간소화: Agg필드변환 로그 제거
                        
                        # 3. ValueObject 이름 덮어쓰기 (alias 매칭으로 refs 보존)
                        orig_vos = orig_item.get("valueObjects", [])
                        trans_vos = trans_item.get("valueObjects", []) if trans_item else []
                        
                        # 🔧 원본 VO 이름 및 refs 가져오기 (original_draft_options에서)
                        original_vo_names = {}  # {alias: original_name}
                        original_vo_refs = {}  # {alias: refs}
                        if original_draft_options and i < len(original_draft_options):
                            original_opt_structure = original_draft_options[i].get("structure", [])
                            for orig_opt_item in original_opt_structure:
                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                    for orig_opt_vo in orig_opt_item.get("valueObjects", []):
                                        vo_alias = orig_opt_vo.get("alias")
                                        vo_name = orig_opt_vo.get("name")
                                        vo_refs = orig_opt_vo.get("refs", [])
                                        if vo_alias and vo_name:
                                            original_vo_names[vo_alias] = vo_name
                                            original_vo_refs[vo_alias] = vo_refs
                                    break
                        
                        # 🔧 Deterministic VO 변환: Aggregate prefix 자동 적용
                        # ✅ 모든 BC의 aggregate 이름 매핑 확인 (다른 BC 참조도 처리)
                        import re
                        for orig_vo in orig_vos:
                            orig_vo_alias = orig_vo.get("alias")
                            orig_vo_name = orig_vo.get("name")
                            # 원본 이름 사용 (선처리 전)
                            original_vo_name = original_vo_names.get(orig_vo_alias, orig_vo_name)
                            
                            # LLM 결과에서 매칭되는 것 찾기
                            trans_vo_name = None
                            for trans_vo in trans_vos:
                                if trans_vo.get("alias") == orig_vo_alias:
                                    trans_vo_name = trans_vo.get("name")
                                    # 로그 간소화: LLM결과 로그 제거
                                    break
                            
                            # LLM 결과가 있으면 사용, 없거나 잘못되었으면 자동 생성
                            # ✅ LLM 결과 검증: aggregate 이름과 관련이 있는지 확인 (하드코딩된 prefix 체크 제거)
                            new_name = None
                            
                            if trans_vo_name:
                                # 🔧 CRITICAL FIX: LLM 결과가 aggregate 이름과 관련이 있는지 검증
                                # VO 이름이 aggregate 이름으로 시작하는지 확인
                                is_valid_llm_result = False
                                
                                # 1. 현재 BC의 aggregate와 관련이 있는지 확인
                                if original_agg_name and original_vo_name and original_vo_name.startswith(original_agg_name):
                                    is_valid_llm_result = True
                                else:
                                    # 2. 다른 BC의 aggregate와 관련이 있는지 확인
                                    for mapped_orig_name, mapped_new_name in combined_aggregate_mapping.items():
                                        if original_vo_name.startswith(mapped_orig_name) and mapped_orig_name != original_vo_name:
                                            is_valid_llm_result = True
                                            break
                                
                                if is_valid_llm_result:
                                    # LLM이 올바르게 변환한 경우 (aggregate와 관련 있음)
                                    new_name = trans_vo_name
                                    # 로그 간소화: LLM채택 로그 제거
                                else:
                                    # LLM이 잘못 변환한 경우 (aggregate와 무관함)
                                    # 로그 간소화: LLM무시 로그 제거
                                    pass  # new_name은 None으로 유지되어 자동 생성 로직으로 진행
                            
                            # LLM 결과가 없거나 무효한 경우 자동 생성 시도
                            if not new_name:
                                # 자동 생성: aggregate_name + "_" + suffix
                                # 예: OrderItem → m_odr_item, CartItem → m_bkt_item, CustomerReference → m_cst_reference
                                
                                # 1. 현재 aggregate 이름 확인 (변환 전 원본 이름 사용!)
                                # ⚠️ original_vo_name 사용 (선처리 전 원본 이름)
                                if original_agg_name and original_vo_name and original_vo_name.startswith(original_agg_name):
                                    # 현재 BC의 aggregate가 변환된 경우
                                    # 예: "OrderItem"은 "Order"로 시작 → "Item" 추출
                                    suffix = original_vo_name[len(original_agg_name):]
                                    suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                    new_name = current_agg_name + "_" + suffix_snake if suffix_snake else current_agg_name + "_vo"
                                    # 로그 간소화: 자동변환 로그 제거
                                    
                                    # 🔧 VO 내부 필드 변환 (VO가 변환되면 내부 필드도 변환)
                                    # VO 내부에 필드가 있는 경우 (예: previewAttributes, ddlFields)
                                    if "previewAttributes" in orig_vo or "ddlFields" in orig_vo:
                                        # VO 내부 필드도 aggregate 이름에 따라 변환
                                        vo_preview_attrs = orig_vo.get("previewAttributes", [])
                                        for vo_attr in vo_preview_attrs:
                                            if isinstance(vo_attr, dict):
                                                vo_field_name = vo_attr.get("fieldName", "")
                                                if vo_field_name:
                                                    # aggregate 이름으로 시작하는 필드명 변환
                                                    if original_agg_name and vo_field_name.startswith(original_agg_name.lower()):
                                                        # aggregate 이름으로 시작하는 필드
                                                        suffix = vo_field_name[len(original_agg_name.lower()):]
                                                        new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix if suffix else current_agg_name.replace("m_", "fld_")
                                                        vo_attr["fieldName"] = new_field_name
                                                        # 로그 간소화: VO필드변환 로그 제거
                                        
                                        vo_ddl_fields = orig_vo.get("ddlFields", [])
                                        for vo_ddl_field in vo_ddl_fields:
                                            if isinstance(vo_ddl_field, dict):
                                                vo_field_name = vo_ddl_field.get("fieldName", "")
                                                if vo_field_name:
                                                    if original_agg_name and vo_field_name.startswith(original_agg_name.lower()):
                                                        suffix = vo_field_name[len(original_agg_name.lower()):]
                                                        new_field_name = current_agg_name.replace("m_", "fld_") + "_" + suffix if suffix else current_agg_name.replace("m_", "fld_")
                                                        vo_ddl_field["fieldName"] = new_field_name
                                                        # 로그 간소화: VO필드변환 로그 제거
                                
                                # 2. 다른 BC의 aggregate 이름 확인 (참조 관계) - original_agg_name이 없어도 실행
                                if (not new_name or new_name == orig_vo_name) and original_vo_name:
                                    # 🔧 combined_aggregate_mapping 사용 (전역 매핑 포함, 다른 BC 참조 가능)
                                    for mapped_orig_name, mapped_new_name in combined_aggregate_mapping.items():
                                        # VO 이름이 aggregate 이름으로 시작하는지 확인
                                        # 예: "OrderItem"는 "Order"로 시작, "CartItem"는 "Cart"로 시작, "CustomerReference"는 "Customer"로 시작
                                        # ⚠️ original_vo_name 사용 (선처리 전 원본 이름)
                                        if original_vo_name.startswith(mapped_orig_name) and mapped_orig_name != original_vo_name:
                                            # 포함된 aggregate 이름을 변환된 이름으로 교체
                                            suffix = original_vo_name[len(mapped_orig_name):]  # 나머지 부분
                                            # snake_case 변환
                                            suffix_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', suffix).lower()
                                            new_name = mapped_new_name + "_" + suffix_snake if suffix_snake else mapped_new_name + "_vo"
                                            # 로그 간소화: 참조변환 로그 제거
                                            break
                                
                                # 3. 매핑이 없으면 원본 유지 (aggregate와 무관한 독립적인 VO/Enum)
                                if not new_name:
                                    new_name = original_vo_name  # 원본 이름 유지
                                    # 로그 간소화: 원본유지 로그 제거
                                    pass
                            
                            if new_name and orig_vo_name != new_name:
                                # 🔒 CRITICAL: refs 보존 (빈 배열도 보존)
                                preserved_refs = orig_vo.get("refs", [])
                                # original_draft_options에서 원본 refs 가져오기
                                if orig_vo_alias in original_vo_refs:
                                    preserved_refs = original_vo_refs[orig_vo_alias]
                                
                                orig_vo["name"] = new_name
                                # refs 복구 (빈 배열도 보존)
                                if "refs" not in orig_vo or not orig_vo.get("refs"):
                                    orig_vo["refs"] = preserved_refs
                                # 로그 간소화: 이름변환 로그 제거
                        
                        # 4. PreviewAttributes fieldName 덮어쓰기 (인덱스 기반 - fieldName은 고유 식별자 없음)
                        orig_attrs = orig_item.get("previewAttributes", [])
                        trans_attrs = trans_item.get("previewAttributes", []) if trans_item else []
                        trans_agg_alias = orig_agg_alias  # orig_item의 alias 사용
                        
                        # 🔒 CRITICAL: original_draft_options에서 원본 refs 복원
                        # fieldAlias 또는 원본 fieldName을 키로 사용 (인덱스 기반은 LLM이 순서를 바꿀 수 있어 불안정)
                        original_attrs_refs = {}  # {fieldAlias or fieldName: refs}
                        if original_draft_options and i < len(original_draft_options):
                            original_opt_structure = original_draft_options[i].get("structure", [])
                            for orig_opt_item in original_opt_structure:
                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                    orig_opt_attrs = orig_opt_item.get("previewAttributes", [])
                                    for orig_opt_attr in orig_opt_attrs:
                                        if isinstance(orig_opt_attr, dict):
                                            # fieldAlias 우선, 없으면 원본 fieldName 사용
                                            key = orig_opt_attr.get("fieldAlias") or orig_opt_attr.get("fieldName")
                                            if key:
                                                original_attrs_refs[key] = orig_opt_attr.get("refs", [])
                                    break
                        
                        if trans_item and trans_attrs:
                            for attr_idx in range(min(len(orig_attrs), len(trans_attrs))):
                                if isinstance(trans_attrs[attr_idx], dict) and "fieldName" in trans_attrs[attr_idx]:
                                    orig_attr = orig_attrs[attr_idx] if isinstance(orig_attrs[attr_idx], dict) else None
                                    if not orig_attr:
                                        continue
                                    
                                    orig_field = orig_attr.get("fieldName")
                                    new_field = trans_attrs[attr_idx]["fieldName"]
                                    
                                    # 🔒 CRITICAL: fieldAlias 또는 원본 fieldName으로 refs 매칭 (인덱스 기반은 불안정)
                                    # fieldAlias 우선 (변환되지 않는 한글 이름), 없으면 원본 fieldName 사용
                                    match_key = orig_attr.get("fieldAlias") or orig_field
                                    if match_key and match_key in original_attrs_refs:
                                        orig_attr["refs"] = original_attrs_refs[match_key]
                                    elif "refs" not in orig_attr:
                                        orig_attr["refs"] = []
                                    
                                    if orig_field and orig_field != new_field:
                                        # 🔧 CRITICAL FIX: 선처리 결과를 우선 확인
                                        # 선처리 결과가 있으면 우선 적용 (deterministic mapping이 더 정확)
                                        preprocessed_field = None
                                        original_field_name = None
                                        if orig_agg_alias and orig_agg_alias in field_name_mapping:
                                            # 선처리 전 원본 필드명 찾기
                                            for orig_field_name, preprocessed_field_name in field_name_mapping[orig_agg_alias].items():
                                                if preprocessed_field_name == orig_field:
                                                    # orig_field가 선처리 결과임
                                                    preprocessed_field = orig_field
                                                    original_field_name = orig_field_name
                                                    break
                                        
                                        # 선처리 결과가 있으면 우선 적용 (LLM 결과 무시)
                                        if preprocessed_field and original_field_name:
                                            # 선처리 결과 유지 (이미 orig_field에 적용되어 있음)
                                            # 🔒 CRITICAL: refs 보존 (fieldAlias/fieldName 기반 매칭)
                                            orig_attr = orig_attrs[attr_idx]
                                            if isinstance(orig_attr, dict):
                                                match_key = orig_attr.get("fieldAlias") or original_field_name
                                                if match_key and match_key in original_attrs_refs:
                                                    orig_attr["refs"] = original_attrs_refs[match_key]
                                                elif "refs" not in orig_attr:
                                                    orig_attr["refs"] = []
                                            # 로그 간소화: 선처리우선 로그 제거
                                            pass
                                        elif orig_field != new_field:
                                            # 선처리 결과가 없을 때만 LLM 결과 사용
                                            # 🔒 CRITICAL: refs 보존 (fieldAlias/fieldName 기반 매칭)
                                            orig_attr = orig_attrs[attr_idx]
                                            if isinstance(orig_attr, dict):
                                                orig_attr["fieldName"] = new_field
                                                # fieldAlias 또는 원본 fieldName으로 매칭
                                                match_key = orig_attr.get("fieldAlias") or orig_field
                                                if match_key and match_key in original_attrs_refs:
                                                    orig_attr["refs"] = original_attrs_refs[match_key]
                                                elif "refs" not in orig_attr:
                                                    orig_attr["refs"] = []
                                            else:
                                                # 새로 생성하는 경우도 fieldAlias/fieldName 기반으로 refs 찾기
                                                match_key = orig_field
                                                refs = original_attrs_refs.get(match_key, []) if match_key else []
                                                orig_attrs[attr_idx] = {"fieldName": new_field, "refs": refs}
                                            # 로그 간소화: LLM변환 로그 제거
                        elif not trans_item or not trans_attrs:
                            # LLM 결과가 없거나 trans_attrs가 비어있으면 선처리 결과 확인
                            # current_opt_structure에서 선처리된 필드명 가져오기
                            if orig_agg_alias and i < len(draft_options):
                                current_opt_structure = draft_options[i].get("structure", [])
                                for curr_item in current_opt_structure:
                                    curr_alias = curr_item.get("aggregate", {}).get("alias")
                                    if curr_alias == orig_agg_alias:
                                        curr_attrs = curr_item.get("previewAttributes", [])
                                        # 🔒 CRITICAL: original_draft_options에서 원본 refs 복원 (fieldAlias/fieldName 기반)
                                        original_attrs_refs = {}  # {fieldAlias or fieldName: refs}
                                        if original_draft_options and i < len(original_draft_options):
                                            original_opt_structure = original_draft_options[i].get("structure", [])
                                            for orig_opt_item in original_opt_structure:
                                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                                    orig_opt_attrs = orig_opt_item.get("previewAttributes", [])
                                                    for orig_opt_attr in orig_opt_attrs:
                                                        if isinstance(orig_opt_attr, dict):
                                                            key = orig_opt_attr.get("fieldAlias") or orig_opt_attr.get("fieldName")
                                                            if key:
                                                                original_attrs_refs[key] = orig_opt_attr.get("refs", [])
                                                    break
                                        # 선처리된 필드명 적용
                                        for attr_idx in range(min(len(orig_attrs), len(curr_attrs))):
                                            if isinstance(orig_attrs[attr_idx], dict) and isinstance(curr_attrs[attr_idx], dict):
                                                orig_attr = orig_attrs[attr_idx]
                                                orig_field = orig_attr.get("fieldName")
                                                curr_field = curr_attrs[attr_idx].get("fieldName")
                                                
                                                # 🔒 CRITICAL: fieldAlias 또는 원본 fieldName으로 refs 매칭
                                                match_key = orig_attr.get("fieldAlias") or orig_field
                                                if match_key and match_key in original_attrs_refs:
                                                    orig_attr["refs"] = original_attrs_refs[match_key]
                                                elif "refs" not in orig_attr:
                                                    orig_attr["refs"] = []
                                                
                                                if orig_field and curr_field and orig_field != curr_field:
                                                    orig_attr["fieldName"] = curr_field
                                                    # 로그 간소화: 선처리적용 로그 제거
                                        break
                            # 로그 간소화: 정보 로그 제거
                        
                        # 5. DDLFields fieldName 덮어쓰기 (인덱스 기반 - fieldName은 고유 식별자 없음)
                        orig_ddl_fields = orig_item.get("ddlFields", [])
                        trans_ddl_fields = trans_item.get("ddlFields", []) if trans_item else []
                        
                        # 🔒 CRITICAL: original_draft_options에서 원본 ddlFields refs 복원
                        # fieldAlias 또는 원본 fieldName을 키로 사용 (인덱스 기반은 LLM이 순서를 바꿀 수 있어 불안정)
                        original_ddl_fields_refs = {}  # {fieldAlias or fieldName: refs}
                        if original_draft_options and i < len(original_draft_options):
                            original_opt_structure = original_draft_options[i].get("structure", [])
                            for orig_opt_item in original_opt_structure:
                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                    orig_opt_ddl_fields = orig_opt_item.get("ddlFields", [])
                                    for orig_opt_ddl_field in orig_opt_ddl_fields:
                                        if isinstance(orig_opt_ddl_field, dict):
                                            # fieldAlias 우선, 없으면 원본 fieldName 사용
                                            key = orig_opt_ddl_field.get("fieldAlias") or orig_opt_ddl_field.get("fieldName")
                                            if key:
                                                original_ddl_fields_refs[key] = orig_opt_ddl_field.get("refs", [])
                                    break
                        
                        if trans_item and trans_ddl_fields:
                            for ddl_idx in range(min(len(orig_ddl_fields), len(trans_ddl_fields))):
                                if isinstance(trans_ddl_fields[ddl_idx], dict) and "fieldName" in trans_ddl_fields[ddl_idx]:
                                    orig_ddl_attr = orig_ddl_fields[ddl_idx] if isinstance(orig_ddl_fields[ddl_idx], dict) else None
                                    if not orig_ddl_attr:
                                        continue
                                    
                                    orig_ddl_field = orig_ddl_attr.get("fieldName")
                                    new_ddl_field = trans_ddl_fields[ddl_idx]["fieldName"]
                                    
                                    # 🔒 CRITICAL: fieldAlias 또는 원본 fieldName으로 refs 매칭 (인덱스 기반은 불안정)
                                    # fieldAlias 우선 (변환되지 않는 한글 이름), 없으면 원본 fieldName 사용
                                    match_key = orig_ddl_attr.get("fieldAlias") or orig_ddl_field
                                    if match_key and match_key in original_ddl_fields_refs:
                                        orig_ddl_attr["refs"] = original_ddl_fields_refs[match_key]
                                    elif "refs" not in orig_ddl_attr:
                                        orig_ddl_attr["refs"] = []
                                    
                                    if orig_ddl_field and orig_ddl_field != new_ddl_field:
                                        # 🔧 CRITICAL FIX: 선처리 결과를 우선 확인
                                        # 선처리 결과가 있으면 우선 적용 (deterministic mapping이 더 정확)
                                        preprocessed_field = None
                                        original_field_name = None
                                        if orig_agg_alias and orig_agg_alias in field_name_mapping:
                                            # 선처리 전 원본 필드명 찾기
                                            for orig_field_name, preprocessed_field_name in field_name_mapping[orig_agg_alias].items():
                                                if preprocessed_field_name == orig_ddl_field:
                                                    # orig_ddl_field가 선처리 결과임
                                                    preprocessed_field = orig_ddl_field
                                                    original_field_name = orig_field_name
                                                    break
                                        
                                        # 선처리 결과가 있으면 우선 적용 (LLM 결과 무시)
                                        if preprocessed_field and original_field_name:
                                            # 선처리 결과 유지 (이미 orig_ddl_field에 적용되어 있음)
                                            # 🔒 CRITICAL: refs 보존 (fieldAlias/fieldName 기반 매칭)
                                            orig_ddl_attr = orig_ddl_fields[ddl_idx]
                                            if isinstance(orig_ddl_attr, dict):
                                                match_key = orig_ddl_attr.get("fieldAlias") or original_field_name
                                                if match_key and match_key in original_ddl_fields_refs:
                                                    orig_ddl_attr["refs"] = original_ddl_fields_refs[match_key]
                                                elif "refs" not in orig_ddl_attr:
                                                    orig_ddl_attr["refs"] = []
                                            # 로그 간소화: 선처리우선 로그 제거
                                            pass
                                        elif orig_ddl_field != new_ddl_field:
                                            # 선처리 결과가 없을 때만 LLM 결과 사용
                                            # 🔒 CRITICAL: refs 보존 (fieldAlias/fieldName 기반 매칭)
                                            orig_ddl_attr = orig_ddl_fields[ddl_idx]
                                            if isinstance(orig_ddl_attr, dict):
                                                orig_ddl_attr["fieldName"] = new_ddl_field
                                                # fieldAlias 또는 원본 fieldName으로 매칭
                                                match_key = orig_ddl_attr.get("fieldAlias") or orig_ddl_field
                                                if match_key and match_key in original_ddl_fields_refs:
                                                    orig_ddl_attr["refs"] = original_ddl_fields_refs[match_key]
                                                elif "refs" not in orig_ddl_attr:
                                                    orig_ddl_attr["refs"] = []
                                            else:
                                                # 새로 생성하는 경우도 fieldAlias/fieldName 기반으로 refs 찾기
                                                match_key = orig_ddl_field
                                                refs = original_ddl_fields_refs.get(match_key, []) if match_key else []
                                                orig_ddl_fields[ddl_idx] = {"fieldName": new_ddl_field, "refs": refs}
                                            # 로그 간소화: LLM변환 로그 제거
                        elif not trans_item or not trans_ddl_fields:
                            # LLM 결과가 없거나 trans_ddl_fields가 비어있으면 선처리 결과 확인
                            # current_opt_structure에서 선처리된 필드명 가져오기
                            if orig_agg_alias and i < len(draft_options):
                                current_opt_structure = draft_options[i].get("structure", [])
                                for curr_item in current_opt_structure:
                                    curr_alias = curr_item.get("aggregate", {}).get("alias")
                                    if curr_alias == orig_agg_alias:
                                        curr_ddl_fields = curr_item.get("ddlFields", [])
                                        # 🔒 CRITICAL: original_draft_options에서 원본 ddlFields refs 복원 (fieldAlias/fieldName 기반)
                                        original_ddl_fields_refs = {}  # {fieldAlias or fieldName: refs}
                                        if original_draft_options and i < len(original_draft_options):
                                            original_opt_structure = original_draft_options[i].get("structure", [])
                                            for orig_opt_item in original_opt_structure:
                                                if orig_opt_item.get("aggregate", {}).get("alias") == orig_agg_alias:
                                                    orig_opt_ddl_fields = orig_opt_item.get("ddlFields", [])
                                                    for orig_opt_ddl_field in orig_opt_ddl_fields:
                                                        if isinstance(orig_opt_ddl_field, dict):
                                                            key = orig_opt_ddl_field.get("fieldAlias") or orig_opt_ddl_field.get("fieldName")
                                                            if key:
                                                                original_ddl_fields_refs[key] = orig_opt_ddl_field.get("refs", [])
                                                    break
                                        # 선처리된 필드명 적용
                                        for ddl_idx in range(min(len(orig_ddl_fields), len(curr_ddl_fields))):
                                            if isinstance(orig_ddl_fields[ddl_idx], dict) and isinstance(curr_ddl_fields[ddl_idx], dict):
                                                orig_ddl_attr = orig_ddl_fields[ddl_idx]
                                                orig_field = orig_ddl_attr.get("fieldName")
                                                curr_field = curr_ddl_fields[ddl_idx].get("fieldName")
                                                
                                                # 🔒 CRITICAL: fieldAlias 또는 원본 fieldName으로 refs 매칭
                                                match_key = orig_ddl_attr.get("fieldAlias") or orig_field
                                                if match_key and match_key in original_ddl_fields_refs:
                                                    orig_ddl_attr["refs"] = original_ddl_fields_refs[match_key]
                                                elif "refs" not in orig_ddl_attr:
                                                    orig_ddl_attr["refs"] = []
                                                
                                                if orig_field and curr_field and orig_field != curr_field:
                                                    orig_ddl_attr["fieldName"] = curr_field
                                                    # 로그 간소화: 선처리적용 로그 제거
                                        break
                            # 로그 간소화: 정보 로그 제거
                            pass
                    
                    # 🔒 CRITICAL: original_draft_options에서 필터링된 필드 복원
                    if original_draft_options and i < len(original_draft_options):
                        original_option = original_draft_options[i]
                        
                        # boundedContext 복원 (aggregates는 제외, description도 제외 - 원래 없었음)
                        if "boundedContext" in original_option:
                            orig_bc = original_option["boundedContext"]
                            if "boundedContext" not in merged_option:
                                merged_option["boundedContext"] = {}
                            
                            # aggregates는 이미 업데이트된 것 사용, description은 제외
                            for key, value in orig_bc.items():
                                if key not in ["aggregates", "description"]:  # aggregates는 이미 업데이트됨, description은 원래 없었음
                                    merged_option["boundedContext"][key] = value
                        
                        # pros 복원
                        if "pros" in original_option:
                            merged_option["pros"] = original_option["pros"]
                        elif "pros" in transformed_option:
                            # LLM 결과에 있으면 사용 (fallback)
                            merged_option["pros"] = transformed_option["pros"]
                        
                        # cons 복원
                        if "cons" in original_option:
                            merged_option["cons"] = original_option["cons"]
                        elif "cons" in transformed_option:
                            # LLM 결과에 있으면 사용 (fallback)
                            merged_option["cons"] = transformed_option["cons"]
                        
                        # description 복원 (옵션 레벨)
                        if "description" in original_option:
                            merged_option["description"] = original_option["description"]
                        
                        # 기타 필드 복원 (original_option에 있지만 merged_option에 없는 모든 필드)
                        for key, value in original_option.items():
                            if key not in ["structure", "boundedContext", "pros", "cons", "description"]:
                                # structure는 이미 처리됨, 나머지는 복원
                                if key not in merged_option:
                                    merged_option[key] = value
                    else:
                        # original_draft_options가 없으면 LLM 결과 사용 (fallback)
                        if "pros" in transformed_option:
                            merged_option["pros"] = transformed_option["pros"]
                        if "cons" in transformed_option:
                            merged_option["cons"] = transformed_option["cons"]
                    
                    # boundedContext.aggregates[].name도 structure[].aggregate.name과 동기화
                    if "boundedContext" in merged_option and "aggregates" in merged_option["boundedContext"]:
                        bc_aggregates = merged_option["boundedContext"]["aggregates"]
                        result_structure = merged_option["structure"]
                        # structure의 aggregate.name으로 boundedContext.aggregates[].name 업데이트
                        for idx, structure_item in enumerate(result_structure):
                            if idx < len(bc_aggregates):
                                structure_agg_name = structure_item.get("aggregate", {}).get("name")
                                if structure_agg_name:
                                    bc_aggregates[idx]["name"] = structure_agg_name
                                    # 로그 간소화: 동기화 로그 제거
                                    pass
                    
                    # referencedAggregateName 업데이트 및 referencedAggregate 객체 생성
                    result_structure = merged_option["structure"]
                    
                    # aggregate 이름 매핑 생성 (원본과 결과 비교)
                    aggregate_name_mapping = {}
                    for idx in range(len(result_structure)):
                        # 결과 구조의 현재 이름 (이미 LLM 결과로 덮어씌워짐)
                        current_agg_name = result_structure[idx].get("aggregate", {}).get("name", "")
                        # 원본 이름 (변환 전)
                        if idx < len(original_structure):
                            original_agg_name = original_structure[idx].get("aggregate", {}).get("name", "")
                            if original_agg_name and current_agg_name and original_agg_name != current_agg_name:
                                aggregate_name_mapping[original_agg_name] = current_agg_name
                                LoggingUtil.info("StandardTransformer", 
                                               f"   [LLM 변환 결과] Aggregate '{original_agg_name}' → '{current_agg_name}'")
                    
                    # 모든 VO의 referencedAggregateName 업데이트
                    for structure_item in result_structure:
                        value_objects = structure_item.get("valueObjects", [])
                        for vo in value_objects:
                            ref_agg_name = vo.get("referencedAggregateName")
                            if ref_agg_name:
                                # referencedAggregateName이 변환된 aggregate를 참조하면 업데이트
                                if ref_agg_name in aggregate_name_mapping:
                                    new_ref_agg_name = aggregate_name_mapping[ref_agg_name]
                                    vo["referencedAggregateName"] = new_ref_agg_name
                                    LoggingUtil.info("StandardTransformer", 
                                                   f"   [참조업데이트] VO '{vo.get('alias')}' ref: '{ref_agg_name}' → '{new_ref_agg_name}'")
                    
                    merged_options.append(merged_option)
                else:
                    # 원본이 없는 경우 변환된 옵션 사용
                    merged_options.append(transformed_option)
            
            # 첫 번째 옵션 구조 확인
            if merged_options:
                first_option = merged_options[0]
                LoggingUtil.info("StandardTransformer", 
                               f"   첫 번째 옵션 키: {list(first_option.keys()) if isinstance(first_option, dict) else 'N/A'}")
                if isinstance(first_option, dict) and "structure" in first_option:
                    structure = first_option.get("structure", [])
                    LoggingUtil.info("StandardTransformer", 
                                   f"   첫 번째 옵션 structure 항목 수: {len(structure)}개")
                    if structure and isinstance(structure[0], dict):
                        LoggingUtil.info("StandardTransformer", 
                                       f"   첫 번째 structure 항목 키: {list(structure[0].keys())}")
            
            return merged_options
        except Exception as e:
            LoggingUtil.error("StandardTransformer", f"❌ LLM 호출 실패: {e}")
            import traceback
            LoggingUtil.error("StandardTransformer", traceback.format_exc())
            # 원본 반환
            return draft_options
    
    def _build_transformation_prompt(self, draft_options: List[Dict],
                                   bounded_context: Dict,
                                   relevant_standards: List[Dict],
                                   query_search_results: Optional[List[Dict]] = None) -> str:
        """
        변환 프롬프트 구성
        """
        # 표준 문서 포맷팅
        standards_text = ""
        
        # 쿼리별 검색 결과를 표준전환 대상 형식으로 변환 (top-k=3 결과 모두 포함)
        # 쿼리를 key로 하고, 해당 쿼리의 결과 리스트를 value로 하는 딕셔너리 구조
        transformed_query_results = {}
        if query_search_results:
            for qr in query_search_results:
                query = qr.get("query", "")
                if not query:
                    continue
                
                # top-k=3 결과가 "results" 리스트로 전달됨
                if "results" in qr:
                    results_list = qr["results"]
                    # 각 결과를 리스트로 변환 (result만 포함, similarity_score 제거)
                    query_results = []
                    for result_item in results_list:
                        query_results.append(result_item.get("result", {}))
                    transformed_query_results[query] = query_results
                else:
                    # 하위 호환성: 기존 형식 (단일 result)
                    transformed_query_results[query] = [qr.get("result", {})]
        
        # 검색 결과가 있는 경우에만 표준 변환 정보 추가
        if transformed_query_results:
            standards_text += "\n\n## Standard Transformation Reference:\n\n"
            standards_text += "The following JSON contains search results from the company standards database.\n"
            standards_text += "Each query (standard transformation target) has up to 3 candidate results.\n\n"
            standards_text += "**JSON Structure**:\n"
            standards_text += "- Each key is a search query (e.g., \"Order 주문\", \"customer_id 고객ID\")\n"
            standards_text += "- Each value is a list of search results for that query\n"
            standards_text += "- Each result contains standard information\n\n"
            standards_text += "**Matching and Selection Rules**:\n"
            standards_text += "- If input `name` or `fieldName` matches or is contained in a JSON key, consider transformation\n"
            standards_text += "- Example: Input `name: \"Order\"` matches key `\"Order 주문\"`, Input `fieldName: \"customer_id\"` matches key `\"customer_id 고객ID\"`\n"
            standards_text += "- If matched, select the most appropriate item from the value list based on context and meaning, then use its `표준명` (standard name)\n"
            standards_text += "- Standard name format: `\"m_cst\"`, `\"fld_cst_id\"`, `\"fld_odr_amt\"` (not common names)\n"
            standards_text += "- If no match or inappropriate, keep the original unchanged\n\n"
            
            # 변환된 검색 결과를 JSON 형식으로 전달 (딕셔너리 구조)
            standards_text += "```json\n"
            standards_text += json.dumps(transformed_query_results, ensure_ascii=False, indent=2)
            standards_text += "\n```\n\n"
        else:
            standards_text = "\n\n⚠️  **CRITICAL: No standard transformation information found.**\n\n"
            standards_text += "**STRICT REQUIREMENT**: Since no standard information is available:\n"
            standards_text += "- **DO NOT transform names** - keep original names as they are\n"
            standards_text += "- **DO NOT invent or guess standard names**\n"
            standards_text += "- **DO NOT apply general naming conventions (camelCase, etc.)**\n"
            standards_text += "- **Keep ALL names EXACTLY as they are in the input**\n"
            standards_text += "- **This means: aggregate.name, enum.name, vo.name, field.fieldName should ALL remain unchanged**\n\n"
            standards_text += "**REASON**: Vector Store indexing may have failed or no relevant standards were found.\n"
            standards_text += "Without company standards, transformation should NOT occur.\n\n"
            LoggingUtil.warning("StandardTransformer", 
                              f"⚠️  표준 전환 정보 없음: query_search_results가 비어있음 - LLM에게 원본 유지 지시")
        
        # Bounded Context 정보
        bc_name = bounded_context.get("name", "")
        bc_alias = bounded_context.get("alias", "")
        bc_desc = bounded_context.get("description", "")
        
        # 각 옵션의 aggregate 수 계산 및 명시
        option_aggregate_counts = []
        for i, option in enumerate(draft_options):
            structure = option.get("structure", [])
            aggregate_count = len(structure)
            option_aggregate_counts.append(aggregate_count)
        
        aggregate_counts_text = "\n".join([
            f"- Option {i}: {count} aggregates in structure array"
            for i, count in enumerate(option_aggregate_counts)
        ])
        
        # 추가 옵션들의 aggregate 수 요구사항 텍스트 생성
        additional_options_text = ""
        if len(option_aggregate_counts) > 1:
            additional_lines = []
            for i, count in enumerate(option_aggregate_counts[1:], start=1):
                additional_lines.append(f"- Option {i} MUST have EXACTLY {count} aggregates in its structure array.")
            additional_options_text = "\n".join(additional_lines)
        
        # 섹션 5의 추가 옵션 텍스트 생성
        section5_additional = ""
        if len(option_aggregate_counts) > 1:
            section5_lines = []
            for i, count in enumerate(option_aggregate_counts[1:], start=1):
                section5_lines.append(f"     * Option {i}: MUST have {count} aggregates (same as original)")
            section5_additional = "\n".join(section5_lines)
        
        # Output Format 섹션의 추가 옵션 텍스트 생성
        output_format_additional = ""
        if len(option_aggregate_counts) > 1:
            output_lines = []
            for i, count in enumerate(option_aggregate_counts[1:], start=1):
                output_lines.append(f"  * Option {i}: MUST have EXACTLY {count} aggregates")
            output_format_additional = "\n".join(output_lines)
        
        # Final Reminder 섹션의 추가 옵션 텍스트 생성
        reminder_additional = ""
        if len(option_aggregate_counts) > 1:
            reminder_lines = []
            for i, (j, count) in enumerate(enumerate(option_aggregate_counts[1:], start=1), start=3):
                reminder_lines.append(f"{i}. Option {j} MUST have EXACTLY {count} aggregates in its structure array.")
            reminder_additional = "\n".join(reminder_lines)
            final_reminder_num = len(option_aggregate_counts) + 2
        else:
            final_reminder_num = 3
        
        prompt = f"""You are a DDD Standard Compliance Specialist. Transform aggregate names to match company standards.

## Task: Transform ONLY the `name` and `fieldName` fields. Keep EVERYTHING else EXACTLY unchanged.

## Input:
{json.dumps(draft_options, ensure_ascii=False, indent=2)}

{standards_text}

## CRITICAL RULES (READ CAREFULLY):

**⚠️ STRUCTURE PRESERVATION (MOST IMPORTANT):**
- **MUST preserve EXACT structure for EACH option**: 
{aggregate_counts_text}
- **MUST preserve ALL options**: Input has {len(draft_options)} options → Output MUST have EXACTLY {len(draft_options)} options
- **⚠️ CRITICAL: Each option's `structure` array MUST contain ALL aggregates from the input**:
  * If input has 3 aggregates in structure array, output MUST have EXACTLY 3 aggregates in structure array
  * DO NOT split aggregates into separate options
  * DO NOT merge aggregates into one
  * ALL aggregates must remain in the SAME structure array within the SAME option
- **MUST preserve ALL arrays**: `previewAttributes`, `ddlFields`, `enumerations`, `valueObjects` - keep ALL items, only transform `name`/`fieldName` values
- **MUST preserve ALL fields**: Every field in input must exist in output (only `name` and `fieldName` values may change)

**WHAT TO TRANSFORM (ALL EQUALLY IMPORTANT):**
- `aggregate.name` - Transform aggregate names
- `enumerations[].name` - Transform enumeration names  
- `valueObjects[].name` - Transform value object names
- `previewAttributes[].fieldName` - Transform preview field names (check EVERY field in the array)
- `ddlFields[].fieldName` - Transform DDL field names (check EVERY field in the array)

**Transformation Process:**
- For each `name` or `fieldName`, check if it matches a key in "Standard Transformation Reference"
- If matched, transform using the most appropriate `표준명` from the candidate list
- If no match, keep original unchanged
- Apply this process to ALL transformation targets listed above - do not skip any

**WHAT TO KEEP UNCHANGED (CRITICAL - DO NOT MODIFY):**
- 🔒 **ALL `alias` fields (Korean text) - NEVER CHANGE THESE!** They are used for matching and traceability.
- **ALL array structures** - Keep ALL items in `previewAttributes`, `ddlFields`, `enumerations`, `valueObjects`
- **ALL aggregate count for EACH option** - 
{aggregate_counts_text}
- **ALL other fields** - `className`, `type`, `referencedAggregateName`, etc. - keep unchanged

**⚠️ CRITICAL WARNING:**
- **DO NOT change any `alias` field** - they must match the input exactly
- **DO NOT translate or modify Korean text in `alias` fields**
- **ONLY transform `name` and `fieldName` fields**

**NOTE:** Input has been pre-processed to remove tracking fields (refs, description, pros, cons). You don't need to include them in output - they will be restored automatically.

**TRANSFORMATION RULES:**

1. Matching: If input `name` or `fieldName` is contained in or matches a JSON key in "Standard Transformation Reference", it matches.
   - Example: `"Order"` matches key `"Order 주문"`, `"customer_id"` matches key `"customer_id 고객ID"`

2. Transformation: If matched, select the most appropriate item from the value list based on context and meaning, then use its `표준명` (standard name).
   - Standard name format: `"m_cst"`, `"fld_cst_id"`, `"fld_odr_amt"` (not common names)
   - Common names: `"Customer"`, `"customer_id"`, `"order_amount"` (not standard names)

3. Transformation Targets (ALL must be checked and transformed if matched):
   - `aggregate.name` - Transform aggregate names
   - `enumerations[].name` - Transform enumeration names
   - `valueObjects[].name` - Transform value object names
   - `previewAttributes[].fieldName` - Transform preview field names (CRITICAL: Check every field)
   - `ddlFields[].fieldName` - Transform DDL field names (CRITICAL: Check every field)

4. Parent-Child Relationship: If parent aggregate name is transformed, child names (enumerations, valueObjects) should use the transformed parent name as prefix.

5. No Match: Keep original unchanged.

## CRITICAL: What You MUST Preserve:

**DO NOT REMOVE:**
- ✅ ALL aggregates ({option_aggregate_counts[0] if option_aggregate_counts else 0} total)
- ✅ `previewAttributes` array (if exists, keep ALL items)
- ✅ ALL `alias` fields

**Where to Transform (ONLY these fields - ALL must be processed):**
- `structure[].aggregate.name` - Aggregate name
- `structure[].enumerations[].name` - Enumeration name
- `structure[].valueObjects[].name` - ValueObject name
- `structure[].previewAttributes[].fieldName` - Preview field name (process ALL fields in the array)
- `structure[].ddlFields[].fieldName` - DDL field name (process ALL fields in the array)

**IMPORTANT:** You must check and transform fields in `previewAttributes` and `ddlFields` arrays just like you do for aggregates, enumerations, and value objects. Do not skip field transformations.

## Output Format:

Return JSON with EXACT structure as input, ONLY changing `name`/`fieldName` values based on the "Standard Transformation Reference" section above.
Each option must preserve the EXACT structure from input.
Only transform names/fieldNames that match keys in the reference JSON above.

## FINAL CHECK (VERIFY BEFORE OUTPUT):
1. ✅ **Count aggregates in structure array**: Input has {option_aggregate_counts[0] if option_aggregate_counts else 0} aggregates in structure array → Output MUST have EXACTLY {option_aggregate_counts[0] if option_aggregate_counts else 0} aggregates in structure array
   * **CRITICAL**: ALL aggregates must be in the SAME structure array, NOT split across multiple options
   * **CRITICAL**: If input has 3 aggregates (Customer, Cart, Order), output MUST have 3 aggregates in structure array
2. ✅ `previewAttributes` array: Output MUST always include this field (empty array `[]` if input doesn't have it)
   * **CRITICAL**: If input has `previewAttributes`, output MUST have it with ALL items (only `fieldName` values may change)
   * **CRITICAL**: Check EVERY field in `previewAttributes` array and transform `fieldName` if it matches Standard Transformation Reference
3. ✅ `ddlFields` array: Output MUST always include this field (empty array `[]` if input doesn't have it)
   * **CRITICAL**: If input has `ddlFields`, output MUST have it with ALL items (only `fieldName` values may change)
   * **CRITICAL**: Check EVERY field in `ddlFields` array and transform `fieldName` if it matches Standard Transformation Reference
4. ✅ `enumerations` array: If input has it, output MUST have it with ALL items (only `name` values may change)
5. ✅ `valueObjects` array: If input has it, output MUST have it with ALL items (only `name` values may change)
6. ✅ `boundedContext.aggregates[].name` = `structure[].aggregate.name` (must match - same count and order)
7. ✅ ALL `alias` fields unchanged (Korean text preserved exactly)

**⚠️ CRITICAL**: If you cannot preserve the exact structure, return the input unchanged rather than losing data!

**FINAL INSTRUCTION**: 
Transform `name` and `fieldName` values based on the "Standard Transformation Reference" section above.
If no match or inappropriate match is found, keep the original unchanged.
"""
        return prompt
    
    def _save_transformation_results(self, job_id: str, 
                                     draft_options: List[Dict], 
                                     transformed_options: List[Dict],
                                     bounded_context: Dict,
                                     search_info: Optional[Dict] = None) -> None:
        """
        변환 전후 결과를 JSON 파일로 저장
        
        Args:
            job_id: Job ID 또는 transformationSessionId (디렉토리명으로 사용)
            draft_options: 변환 전 옵션들
            transformed_options: 변환 후 옵션들
            bounded_context: Bounded Context 정보
        """
        try:
            # result 디렉토리 생성
            result_dir = Config._project_root / 'result' / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # BC 이름을 파일명에 포함 (안전한 파일명으로 변환)
            bc_name = bounded_context.get("name", "unknown")
            bc_name_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in bc_name)
            
            LoggingUtil.info("StandardTransformer", 
                          f"💾 변환 결과 저장 중: {result_dir} (BC: {bc_name})")
            
            # 각 옵션별로 변환 전후 JSON 저장 (항상 옵션은 하나뿐이므로 option_0 제거)
            max_options = max(len(draft_options), len(transformed_options))
            
            for i in range(max_options):
                # 변환 전 옵션 저장 (BC 이름 포함)
                if i < len(draft_options):
                    before_file = result_dir / f'{bc_name_safe}_before.json'
                    with open(before_file, 'w', encoding='utf-8') as f:
                        json.dump(draft_options[i], f, ensure_ascii=False, indent=2)
                    LoggingUtil.info("StandardTransformer", 
                                  f"   저장됨: {before_file.name}")
                
                # 변환 후 옵션 저장 (BC 이름 포함)
                if i < len(transformed_options):
                    after_file = result_dir / f'{bc_name_safe}_after.json'
                    with open(after_file, 'w', encoding='utf-8') as f:
                        json.dump(transformed_options[i], f, ensure_ascii=False, indent=2)
                    LoggingUtil.info("StandardTransformer", 
                                  f"   저장됨: {after_file.name}")
            
            # BC별 요약 정보 저장 (BC 이름 포함)
            summary = {
                "job_id": job_id,
                "bounded_context_name": bc_name,
                "bounded_context": {
                    "name": bounded_context.get("name", ""),
                    "alias": bounded_context.get("alias", ""),
                    "description": bounded_context.get("description", "")
                },
                "transformation_timestamp": datetime.now().isoformat()
            }
            
            # 검색된 표준 정보 추가
            if search_info:
                # 쿼리별 검색 결과 (top-k=3) 저장
                if search_info.get("query_search_results"):
                    query_search_results = search_info["query_search_results"]
                    summary["search_queries"] = []
                    for qr in query_search_results:
                        query = qr.get("query", "")
                        # top-k=3 결과가 "results" 리스트로 전달됨
                        if "results" in qr:
                            results_list = qr["results"]
                            for result_item in results_list:
                                query_info = {
                                    "query": query,
                                    "similarity_score": result_item.get("similarity_score", 0.0),
                                    "result": result_item.get("result", {})
                                }
                                summary["search_queries"].append(query_info)
                        else:
                            # 하위 호환성: 기존 형식 (단일 result)
                            query_info = {
                                "query": query,
                                "similarity_score": qr.get("similarity_score", 0.0),
                                "result": qr.get("result", {})
                            }
                            summary["search_queries"].append(query_info)
                
                # 유사도 검색 채택 결과 (키워드, 쿼리, 유사도, 인덱싱 내용)
                summary["rag_search_results"] = []
                if search_info.get("query_search_results"):
                    query_search_results = search_info["query_search_results"]
                    for qr in query_search_results:
                        query = qr.get("query", "")
                        # top-k=3 결과가 "results" 리스트로 전달됨
                        if "results" in qr:
                            results_list = qr["results"]
                            for result_item in results_list:
                                if result_item.get("result"):
                                    summary["rag_search_results"].append({
                                        "query": query,
                                        "similarity_score": result_item.get("similarity_score", 0.0),
                                        "result": result_item.get("result", {})
                                    })
                        else:
                            # 하위 호환성: 기존 형식 (단일 result)
                            if qr.get("result"):
                                summary["rag_search_results"].append({
                                    "query": query,
                                    "similarity_score": qr.get("similarity_score", 0.0),
                                    "result": qr.get("result", {})
                                })
                
                # 전처리 매핑 정보 (실제로 변환된 것만 표시)
                if search_info.get("mapping_context"):
                    mapping_ctx = search_info["mapping_context"]
                    
                    # 실제로 선처리 매핑으로 변환된 항목만 추출
                    # ⚠️ 중요: mapped_draft_options는 선처리 매핑 후 결과이므로, 선처리 매핑만 추출
                    actual_table_mappings = {}
                    actual_column_mappings = {}
                    
                    # 선처리 매핑만 추출: original_draft_options와 mapped_draft_options 비교
                    if search_info.get("mapped_draft_options") and len(draft_options) > 0:
                        # draft_options는 original_draft_options와 동일 (선처리 전 원본)
                        # mapped_draft_options는 선처리 매핑 후 결과
                        for i, (original, mapped) in enumerate(zip(draft_options, search_info["mapped_draft_options"])):
                            original_structure = original.get("structure", [])
                            mapped_structure = mapped.get("structure", [])
                            
                            # Aggregate 이름 변환 수집 (선처리 매핑만)
                            for orig_item, mapped_item in zip(original_structure, mapped_structure):
                                orig_agg = orig_item.get("aggregate", {})
                                mapped_agg = mapped_item.get("aggregate", {})
                                orig_name = orig_agg.get("name", "")
                                mapped_name = mapped_agg.get("name", "")
                                
                                # 선처리 매핑으로 변환되었는지 확인
                                # mapping_context의 table_standards에 있는지 확인
                                if orig_name != mapped_name and mapping_ctx:
                                    # 선처리 매핑으로 변환되었는지 확인
                                    orig_alias = orig_agg.get("alias", "")
                                    is_preprocessing_mapping = False
                                    
                                    # alias로 매핑되었는지 확인
                                    if orig_alias and orig_alias in mapping_ctx["table"]["table_standards"]:
                                        if mapping_ctx["table"]["table_standards"][orig_alias] == mapped_name:
                                            is_preprocessing_mapping = True
                                    
                                    # name으로 매핑되었는지 확인 (대소문자 변형 포함)
                                    if not is_preprocessing_mapping and orig_name:
                                        name_variants = [orig_name, orig_name.lower(), orig_name.upper(), orig_name.capitalize()]
                                        for variant in name_variants:
                                            if variant in mapping_ctx["table"]["table_standards"]:
                                                if mapping_ctx["table"]["table_standards"][variant] == mapped_name:
                                                    is_preprocessing_mapping = True
                                                    break
                                    
                                    # 선처리 매핑으로 변환된 경우만 추가
                                    if is_preprocessing_mapping:
                                        actual_table_mappings[orig_name] = mapped_name
                                        if orig_alias:
                                            actual_table_mappings[orig_alias] = mapped_name
                                
                                # Enum/VO 이름 변환 수집 (선처리 매핑만)
                                orig_enums = orig_item.get("enumerations", [])
                                mapped_enums = mapped_item.get("enumerations", [])
                                for orig_enum, mapped_enum in zip(orig_enums, mapped_enums):
                                    orig_enum_name = orig_enum.get("name", "")
                                    mapped_enum_name = mapped_enum.get("name", "")
                                    if orig_enum_name != mapped_enum_name and mapping_ctx:
                                        # 선처리 매핑으로 변환되었는지 확인
                                        orig_enum_alias = orig_enum.get("alias", "")
                                        is_preprocessing_mapping = False
                                        
                                        if orig_enum_alias and orig_enum_alias in mapping_ctx["table"]["table_standards"]:
                                            if mapping_ctx["table"]["table_standards"][orig_enum_alias] == mapped_enum_name:
                                                is_preprocessing_mapping = True
                                        
                                        if not is_preprocessing_mapping and orig_enum_name:
                                            name_variants = [orig_enum_name, orig_enum_name.lower(), orig_enum_name.upper(), orig_enum_name.capitalize()]
                                            for variant in name_variants:
                                                if variant in mapping_ctx["table"]["table_standards"]:
                                                    if mapping_ctx["table"]["table_standards"][variant] == mapped_enum_name:
                                                        is_preprocessing_mapping = True
                                                        break
                                        
                                        if is_preprocessing_mapping:
                                            actual_table_mappings[orig_enum_name] = mapped_enum_name
                                            if orig_enum_alias:
                                                actual_table_mappings[orig_enum_alias] = mapped_enum_name
                                
                                orig_vos = orig_item.get("valueObjects", [])
                                mapped_vos = mapped_item.get("valueObjects", [])
                                for orig_vo, mapped_vo in zip(orig_vos, mapped_vos):
                                    orig_vo_name = orig_vo.get("name", "")
                                    mapped_vo_name = mapped_vo.get("name", "")
                                    if orig_vo_name != mapped_vo_name and mapping_ctx:
                                        # 선처리 매핑으로 변환되었는지 확인
                                        orig_vo_alias = orig_vo.get("alias", "")
                                        is_preprocessing_mapping = False
                                        
                                        if orig_vo_alias and orig_vo_alias in mapping_ctx["table"]["table_standards"]:
                                            if mapping_ctx["table"]["table_standards"][orig_vo_alias] == mapped_vo_name:
                                                is_preprocessing_mapping = True
                                        
                                        if not is_preprocessing_mapping and orig_vo_name:
                                            name_variants = [orig_vo_name, orig_vo_name.lower(), orig_vo_name.upper(), orig_vo_name.capitalize()]
                                            for variant in name_variants:
                                                if variant in mapping_ctx["table"]["table_standards"]:
                                                    if mapping_ctx["table"]["table_standards"][variant] == mapped_vo_name:
                                                        is_preprocessing_mapping = True
                                                        break
                                        
                                        if is_preprocessing_mapping:
                                            actual_table_mappings[orig_vo_name] = mapped_vo_name
                                            if orig_vo_alias:
                                                actual_table_mappings[orig_vo_alias] = mapped_vo_name
                                
                                # 필드명 변환 수집 (선처리 매핑만)
                                orig_attrs = orig_item.get("previewAttributes", [])
                                mapped_attrs = mapped_item.get("previewAttributes", [])
                                for orig_attr, mapped_attr in zip(orig_attrs, mapped_attrs):
                                    if isinstance(orig_attr, dict) and isinstance(mapped_attr, dict):
                                        orig_field = orig_attr.get("fieldName", "")
                                        mapped_field = mapped_attr.get("fieldName", "")
                                        if orig_field != mapped_field and mapping_ctx:
                                            # 선처리 매핑으로 변환되었는지 확인
                                            if orig_field in mapping_ctx["table"]["column_standards"]:
                                                if mapping_ctx["table"]["column_standards"][orig_field] == mapped_field:
                                                    actual_column_mappings[orig_field] = mapped_field
                                
                                # DDL 필드명 변환 수집 (선처리 매핑만)
                                orig_ddl_fields = orig_item.get("ddlFields", [])
                                mapped_ddl_fields = mapped_item.get("ddlFields", [])
                                for orig_ddl_field, mapped_ddl_field in zip(orig_ddl_fields, mapped_ddl_fields):
                                    if isinstance(orig_ddl_field, dict) and isinstance(mapped_ddl_field, dict):
                                        orig_ddl_field_name = orig_ddl_field.get("fieldName", "")
                                        mapped_ddl_field_name = mapped_ddl_field.get("fieldName", "")
                                        if orig_ddl_field_name != mapped_ddl_field_name and mapping_ctx:
                                            # 선처리 매핑으로 변환되었는지 확인
                                            if orig_ddl_field_name in mapping_ctx["table"]["column_standards"]:
                                                if mapping_ctx["table"]["column_standards"][orig_ddl_field_name] == mapped_ddl_field_name:
                                                    actual_column_mappings[orig_ddl_field_name] = mapped_ddl_field_name
                    
                    summary["preprocessing_mappings"] = {
                        "table_standards": actual_table_mappings,  # 실제 변환된 것만
                        "column_standards": actual_column_mappings,  # 실제 변환된 것만
                        "total_table_mappings": len(actual_table_mappings),
                        "total_column_mappings": len(actual_column_mappings)
                    }
                
                # 선처리 매핑과 유사도 검색(LLM) 결과를 구분해서 표시
                # 유사도 검색 결과가 실제로 사용되었는지 확인
                has_rag_results = False
                
                # query_search_results가 있고 실제 결과가 있는지 확인
                if search_info.get("query_search_results"):
                    query_search_results = search_info["query_search_results"]
                    for qr in query_search_results:
                        if qr.get("results") and len(qr.get("results", [])) > 0:
                            has_rag_results = True
                            break
                        # 하위 호환성: 단일 result 형식
                        if qr.get("result"):
                            has_rag_results = True
                            break
                
                # relevant_standards가 있는지 확인
                if not has_rag_results and search_info.get("relevant_standards"):
                    has_rag_results = len(search_info["relevant_standards"]) > 0
                
                # search_queries에서 실제로 채택된 결과가 있는지 확인
                if not has_rag_results and search_info.get("standard_queries"):
                    for query_info in search_info["standard_queries"]:
                        if query_info.get("total_found", 0) > 0:
                            has_rag_results = True
                            break
                
                # 검색 결과가 없으면 경고 로그
                if not has_rag_results:
                    LoggingUtil.warning("StandardTransformer", 
                                      f"⚠️  RAG 검색 결과 없음: BC={bounded_context.get('name', 'Unknown')} - LLM이 표준 문서 없이 변환을 수행했습니다.")
                
                rag_llm_mappings = {
                    "table_standards": {},
                    "column_standards": {},
                    "total_table_mappings": 0,
                    "total_column_mappings": 0,
                    "used_rag_search": has_rag_results  # 유사도 검색 결과가 실제로 사용되었는지
                }
                
                # transformed_options에서 매핑 정보 추출
                for i, transformed_option in enumerate(transformed_options):
                    if i < len(draft_options):
                        original_option = draft_options[i]
                        original_structure = original_option.get("structure", [])
                        transformed_structure = transformed_option.get("structure", [])
                        
                        # mapping_info에서 선처리 매핑과 LLM 매핑 추출
                        mapping_info = transformed_option.get("mapping_info", {}).get(f"option_{i}", {})
                        llm_mapping = mapping_info.get("llm_mapping", {})
                        
                        # LLM 매핑으로 변환된 항목 수집 (선처리 매핑이 아닌 것만)
                        for orig_item, trans_item in zip(original_structure, transformed_structure):
                            orig_agg = orig_item.get("aggregate", {})
                            trans_agg = trans_item.get("aggregate", {})
                            orig_name = orig_agg.get("name", "")
                            trans_name = trans_agg.get("name", "")
                            
                            # LLM 매핑에 있고 선처리 매핑이 아닌 경우만 추가
                            if orig_name in llm_mapping and llm_mapping[orig_name] == trans_name:
                                if orig_name != trans_name:
                                    rag_llm_mappings["table_standards"][orig_name] = trans_name
                                    orig_alias = orig_agg.get("alias", "")
                                    if orig_alias:
                                        rag_llm_mappings["table_standards"][orig_alias] = trans_name
                            
                            # Enum/VO LLM 매핑
                            orig_enums = orig_item.get("enumerations", [])
                            trans_enums = trans_item.get("enumerations", [])
                            for orig_enum, trans_enum in zip(orig_enums, trans_enums):
                                orig_enum_name = orig_enum.get("name", "")
                                trans_enum_name = trans_enum.get("name", "")
                                if orig_enum_name != trans_enum_name:
                                    # 선처리 매핑이 아닌 경우만 추가 (preprocessing_mappings에 없으면)
                                    if orig_enum_name not in actual_table_mappings:
                                        rag_llm_mappings["table_standards"][orig_enum_name] = trans_enum_name
                                        orig_enum_alias = orig_enum.get("alias", "")
                                        if orig_enum_alias:
                                            rag_llm_mappings["table_standards"][orig_enum_alias] = trans_enum_name
                            
                            orig_vos = orig_item.get("valueObjects", [])
                            trans_vos = trans_item.get("valueObjects", [])
                            for orig_vo, trans_vo in zip(orig_vos, trans_vos):
                                orig_vo_name = orig_vo.get("name", "")
                                trans_vo_name = trans_vo.get("name", "")
                                if orig_vo_name != trans_vo_name:
                                    # 선처리 매핑이 아닌 경우만 추가
                                    if orig_vo_name not in actual_table_mappings:
                                        rag_llm_mappings["table_standards"][orig_vo_name] = trans_vo_name
                                        orig_vo_alias = orig_vo.get("alias", "")
                                        if orig_vo_alias:
                                            rag_llm_mappings["table_standards"][orig_vo_alias] = trans_vo_name
                            
                            # 필드명 LLM 매핑
                            orig_attrs = orig_item.get("previewAttributes", [])
                            trans_attrs = trans_item.get("previewAttributes", [])
                            for orig_attr, trans_attr in zip(orig_attrs, trans_attrs):
                                if isinstance(orig_attr, dict) and isinstance(trans_attr, dict):
                                    orig_field = orig_attr.get("fieldName", "")
                                    trans_field = trans_attr.get("fieldName", "")
                                    if orig_field != trans_field:
                                        # 선처리 매핑이 아닌 경우만 추가
                                        if orig_field not in actual_column_mappings:
                                            rag_llm_mappings["column_standards"][orig_field] = trans_field
                            
                            # DDL 필드명 LLM 매핑
                            orig_ddl_fields = orig_item.get("ddlFields", [])
                            trans_ddl_fields = trans_item.get("ddlFields", [])
                            for orig_ddl_field, trans_ddl_field in zip(orig_ddl_fields, trans_ddl_fields):
                                if isinstance(orig_ddl_field, dict) and isinstance(trans_ddl_field, dict):
                                    orig_ddl_field_name = orig_ddl_field.get("fieldName", "")
                                    trans_ddl_field_name = trans_ddl_field.get("fieldName", "")
                                    if orig_ddl_field_name != trans_ddl_field_name:
                                        # 선처리 매핑이 아닌 경우만 추가
                                        if orig_ddl_field_name not in actual_column_mappings:
                                            rag_llm_mappings["column_standards"][orig_ddl_field_name] = trans_ddl_field_name
                
                rag_llm_mappings["total_table_mappings"] = len(rag_llm_mappings["table_standards"])
                rag_llm_mappings["total_column_mappings"] = len(rag_llm_mappings["column_standards"])
                
                summary["rag_llm_mappings"] = rag_llm_mappings
                
                # 전처리 매핑 전후 비교 (간단한 요약)
                if search_info.get("mapped_draft_options") and len(draft_options) > 0:
                    preprocessing_comparison = []
                    for i, (original, mapped) in enumerate(zip(draft_options, search_info["mapped_draft_options"])):
                        original_structure = original.get("structure", [])
                        mapped_structure = mapped.get("structure", [])
                        
                        # Aggregate 이름 변경 추적 (선처리 매핑만)
                        aggregate_changes = []
                        for orig_item, mapped_item in zip(original_structure, mapped_structure):
                            orig_agg = orig_item.get("aggregate", {})
                            mapped_agg = mapped_item.get("aggregate", {})
                            orig_name = orig_agg.get("name", "")
                            mapped_name = mapped_agg.get("name", "")
                            if orig_name != mapped_name:
                                # 선처리 매핑인지 확인
                                if orig_name in actual_table_mappings and actual_table_mappings[orig_name] == mapped_name:
                                    aggregate_changes.append({
                                        "alias": orig_agg.get("alias", ""),
                                        "before": orig_name,
                                        "after": mapped_name,
                                        "method": "preprocessing"
                                    })
                        
                        # Enum/VO 이름 변경 추적 (선처리 매핑만)
                        enum_vo_changes = []
                        for orig_item, mapped_item in zip(original_structure, mapped_structure):
                            orig_enums = orig_item.get("enumerations", [])
                            mapped_enums = mapped_item.get("enumerations", [])
                            for orig_enum, mapped_enum in zip(orig_enums, mapped_enums):
                                orig_enum_name = orig_enum.get("name", "")
                                mapped_enum_name = mapped_enum.get("name", "")
                                if orig_enum_name != mapped_enum_name:
                                    # 선처리 매핑인지 확인
                                    if orig_enum_name in actual_table_mappings and actual_table_mappings[orig_enum_name] == mapped_enum_name:
                                        enum_vo_changes.append({
                                            "type": "enum",
                                            "alias": orig_enum.get("alias", ""),
                                            "before": orig_enum_name,
                                            "after": mapped_enum_name,
                                            "method": "preprocessing"
                                        })
                            
                            orig_vos = orig_item.get("valueObjects", [])
                            mapped_vos = mapped_item.get("valueObjects", [])
                            for orig_vo, mapped_vo in zip(orig_vos, mapped_vos):
                                orig_vo_name = orig_vo.get("name", "")
                                mapped_vo_name = mapped_vo.get("name", "")
                                if orig_vo_name != mapped_vo_name:
                                    # 선처리 매핑인지 확인
                                    if orig_vo_name in actual_table_mappings and actual_table_mappings[orig_vo_name] == mapped_vo_name:
                                        enum_vo_changes.append({
                                            "type": "value_object",
                                            "alias": orig_vo.get("alias", ""),
                                            "before": orig_vo_name,
                                            "after": mapped_vo_name,
                                            "method": "preprocessing"
                                        })
                        
                        # 필드명 변경 추적 (선처리 매핑만)
                        field_changes = []
                        for orig_item, mapped_item in zip(original_structure, mapped_structure):
                            orig_attrs = orig_item.get("previewAttributes", [])
                            mapped_attrs = mapped_item.get("previewAttributes", [])
                            for orig_attr, mapped_attr in zip(orig_attrs, mapped_attrs):
                                if isinstance(orig_attr, dict) and isinstance(mapped_attr, dict):
                                    orig_field = orig_attr.get("fieldName", "")
                                    mapped_field = mapped_attr.get("fieldName", "")
                                    if orig_field != mapped_field:
                                        # 선처리 매핑인지 확인
                                        if orig_field in actual_column_mappings and actual_column_mappings[orig_field] == mapped_field:
                                            field_changes.append({
                                                "aggregate_alias": orig_item.get("aggregate", {}).get("alias", ""),
                                                "before": orig_field,
                                                "after": mapped_field,
                                                "method": "preprocessing"
                                            })
                        
                        if aggregate_changes or enum_vo_changes or field_changes:
                            preprocessing_comparison.append({
                                "option_index": i,
                                "aggregate_changes": aggregate_changes,
                                "enum_vo_changes": enum_vo_changes,
                                "field_changes": field_changes
                            })
                    
                    if preprocessing_comparison:
                        summary["preprocessing_comparison"] = preprocessing_comparison
                
                # 유사도 검색(LLM) 결과 비교 (preprocessing_comparison과 동일한 형식)
                if len(transformed_options) > 0:
                    rag_llm_comparison = []
                    # 유사도 검색 결과가 실제로 사용되었는지 확인
                    method_name = "rag_llm" if has_rag_results else "llm_only"
                    
                    for i, (original, transformed) in enumerate(zip(draft_options, transformed_options)):
                        original_structure = original.get("structure", [])
                        transformed_structure = transformed.get("structure", [])
                        
                        # Aggregate 이름 변경 추적 (유사도 검색 + LLM만)
                        aggregate_changes = []
                        for orig_item, trans_item in zip(original_structure, transformed_structure):
                            orig_agg = orig_item.get("aggregate", {})
                            trans_agg = trans_item.get("aggregate", {})
                            orig_name = orig_agg.get("name", "")
                            trans_name = trans_agg.get("name", "")
                            
                            # 선처리 매핑이 아니고 변환된 경우만 추가
                            if orig_name != trans_name:
                                is_preprocessing = orig_name in actual_table_mappings and actual_table_mappings[orig_name] == trans_name
                                if not is_preprocessing:
                                    aggregate_changes.append({
                                        "alias": orig_agg.get("alias", ""),
                                        "before": orig_name,
                                        "after": trans_name,
                                        "method": method_name  # "rag_llm" 또는 "llm_only"
                                    })
                        
                        # Enum/VO 이름 변경 추적 (유사도 검색 + LLM만)
                        enum_vo_changes = []
                        for orig_item, trans_item in zip(original_structure, transformed_structure):
                            orig_enums = orig_item.get("enumerations", [])
                            trans_enums = trans_item.get("enumerations", [])
                            for orig_enum, trans_enum in zip(orig_enums, trans_enums):
                                orig_enum_name = orig_enum.get("name", "")
                                trans_enum_name = trans_enum.get("name", "")
                                if orig_enum_name != trans_enum_name:
                                    # 선처리 매핑이 아닌 경우만 추가
                                    is_preprocessing = orig_enum_name in actual_table_mappings and actual_table_mappings[orig_enum_name] == trans_enum_name
                                    if not is_preprocessing:
                                        enum_vo_changes.append({
                                            "type": "enum",
                                            "alias": orig_enum.get("alias", ""),
                                            "before": orig_enum_name,
                                            "after": trans_enum_name,
                                            "method": method_name  # "rag_llm" 또는 "llm_only"
                                        })
                            
                            orig_vos = orig_item.get("valueObjects", [])
                            trans_vos = trans_item.get("valueObjects", [])
                            for orig_vo, trans_vo in zip(orig_vos, trans_vos):
                                orig_vo_name = orig_vo.get("name", "")
                                trans_vo_name = trans_vo.get("name", "")
                                if orig_vo_name != trans_vo_name:
                                    # 선처리 매핑이 아닌 경우만 추가
                                    is_preprocessing = orig_vo_name in actual_table_mappings and actual_table_mappings[orig_vo_name] == trans_vo_name
                                    if not is_preprocessing:
                                        enum_vo_changes.append({
                                            "type": "value_object",
                                            "alias": orig_vo.get("alias", ""),
                                            "before": orig_vo_name,
                                            "after": trans_vo_name,
                                            "method": method_name  # "rag_llm" 또는 "llm_only"
                                        })
                        
                        # 필드명 변경 추적 (유사도 검색 + LLM만)
                        field_changes = []
                        for orig_item, trans_item in zip(original_structure, transformed_structure):
                            orig_attrs = orig_item.get("previewAttributes", [])
                            trans_attrs = trans_item.get("previewAttributes", [])
                            for orig_attr, trans_attr in zip(orig_attrs, trans_attrs):
                                if isinstance(orig_attr, dict) and isinstance(trans_attr, dict):
                                    orig_field = orig_attr.get("fieldName", "")
                                    trans_field = trans_attr.get("fieldName", "")
                                    if orig_field != trans_field:
                                        # 선처리 매핑이 아닌 경우만 추가
                                        is_preprocessing = orig_field in actual_column_mappings and actual_column_mappings[orig_field] == trans_field
                                        if not is_preprocessing:
                                            field_changes.append({
                                                "aggregate_alias": orig_item.get("aggregate", {}).get("alias", ""),
                                                "before": orig_field,
                                                "after": trans_field,
                                                "method": method_name  # "rag_llm" 또는 "llm_only"
                                            })
                        
                        if aggregate_changes or enum_vo_changes or field_changes:
                            rag_llm_comparison.append({
                                "option_index": i,
                                "aggregate_changes": aggregate_changes,
                                "enum_vo_changes": enum_vo_changes,
                                "field_changes": field_changes
                            })
                    
                    if rag_llm_comparison:
                        summary["rag_llm_comparison"] = rag_llm_comparison
                        
                        # 유사도 채택으로 바뀐 결과 (간단한 리스트 형식)
                        summary["rag_transformations"] = []
                        for comp in rag_llm_comparison:
                            option_index = comp.get("option_index", 0)
                            for change in comp.get("aggregate_changes", []):
                                summary["rag_transformations"].append({
                                    "option_index": option_index,
                                    "type": "aggregate",
                                    "alias": change.get("alias", ""),
                                    "before": change.get("before", ""),
                                    "after": change.get("after", ""),
                                    "method": change.get("method", "rag_llm")
                                })
                            for change in comp.get("enum_vo_changes", []):
                                summary["rag_transformations"].append({
                                    "option_index": option_index,
                                    "type": change.get("type", "enum"),
                                    "alias": change.get("alias", ""),
                                    "before": change.get("before", ""),
                                    "after": change.get("after", ""),
                                    "method": change.get("method", "rag_llm")
                                })
                            for change in comp.get("field_changes", []):
                                summary["rag_transformations"].append({
                                    "option_index": option_index,
                                    "type": "field",
                                    "alias": change.get("aggregate_alias", ""),
                                    "before": change.get("before", ""),
                                    "after": change.get("after", ""),
                                    "method": change.get("method", "rag_llm")
                                })
            
            summary_file = result_dir / f'{bc_name_safe}_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            LoggingUtil.info("StandardTransformer", 
                          f"✅ 변환 결과 저장 완료: {result_dir}")
            
        except Exception as e:
            LoggingUtil.error("StandardTransformer", 
                            f"❌ 변환 결과 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_response_schema(self) -> Dict:
        """응답 스키마 정의 (aggregate_draft_generator와 동일한 패턴)"""
        return {
            "title": "StandardTransformationResponse",
            "description": "Response schema for standard transformation",
            "type": "object",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Brief explanation of transformations applied"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "transformedOptions": {
                            "type": "array",
                            "description": "Transformed aggregate draft options",
                            "items": {
                                "type": "object",
                                "properties": {
                                    # 항상 있는 필드
                                    "structure": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "aggregate": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                        "alias": {"type": "string"}
                                                    },
                                                    "required": ["name", "alias"],
                                                    "additionalProperties": True  # refs 등 추가 필드 허용
                                                },
                                                "enumerations": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "alias": {"type": "string"}
                                                        },
                                                        "required": ["name", "alias"],
                                                        "additionalProperties": True  # refs 등 추가 필드 허용
                                                    }
                                                },
                                                "valueObjects": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "alias": {"type": "string"},
                                                            "referencedAggregateName": {"type": "string"}
                                                        },
                                                        "required": ["name", "alias", "referencedAggregateName"],
                                                        "additionalProperties": True  # refs 등 추가 필드 허용
                                                    }
                                                },
                                                "previewAttributes": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "fieldName": {"type": "string"},
                                                            "fieldAlias": {"type": "string"}
                                                        },
                                                        "required": ["fieldName", "fieldAlias"],
                                                        "additionalProperties": True
                                                    }
                                                },
                                                "ddlFields": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "fieldName": {"type": "string"},
                                                            "fieldAlias": {"type": "string"}
                                                        },
                                                        "required": ["fieldName", "fieldAlias"],
                                                        "additionalProperties": True
                                                    }
                                                }
                                            },
                                            "required": ["aggregate", "enumerations", "valueObjects", "previewAttributes", "ddlFields"],
                                            "additionalProperties": True  # 기타 추가 필드 허용
                                        },
                                        "description": "Aggregate structure with transformed names (preserve all aggregates from original)"
                                    },
                                    "pros": {
                                        "type": "object",
                                        "properties": {
                                            "cohesion": {"type": "string"},
                                            "coupling": {"type": "string"},
                                            "consistency": {"type": "string"},
                                            "encapsulation": {"type": "string"},
                                            "complexity": {"type": "string"},
                                            "independence": {"type": "string"},
                                            "performance": {"type": "string"}
                                        },
                                        "required": ["cohesion", "coupling", "consistency", "encapsulation", "complexity", "independence", "performance"],
                                        "additionalProperties": False,
                                        "description": "Pros analysis (preserve from original)"
                                    },
                                    "cons": {
                                        "type": "object",
                                        "properties": {
                                            "cohesion": {"type": "string"},
                                            "coupling": {"type": "string"},
                                            "consistency": {"type": "string"},
                                            "encapsulation": {"type": "string"},
                                            "complexity": {"type": "string"},
                                            "independence": {"type": "string"},
                                            "performance": {"type": "string"}
                                        },
                                        "required": ["cohesion", "coupling", "consistency", "encapsulation", "complexity", "independence", "performance"],
                                        "additionalProperties": False,
                                        "description": "Cons analysis (preserve from original)"
                                    }
                                },
                                # required에는 항상 있는 필드만 포함 (aggregate_draft_generator와 동일)
                                "required": ["structure", "pros", "cons"],
                                "additionalProperties": True  # boundedContext, description 등 원본의 다른 필드들도 허용
                            }
                        }
                    },
                    "required": ["transformedOptions"],
                    "additionalProperties": False
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }
    
    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """
        프롬프트의 대략적인 토큰 수 추정
        (대략적으로 문자 수 / 4를 사용, 더 정확한 추정이 필요하면 tiktoken 사용 가능)
        
        Args:
            prompt: 프롬프트 문자열
            
        Returns:
            예상 토큰 수
        """
        # 대략적인 추정: 영어는 평균 4자당 1토큰, 한글은 평균 1.5자당 1토큰
        # 혼합 텍스트를 고려하여 평균적으로 문자 수 / 3.5 사용
        return len(prompt) // 3
    
    def _chunk_preview_attributes(self, preview_attrs: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """
        previewAttributes를 청크로 나누기
        
        Args:
            preview_attrs: previewAttributes 리스트
            chunk_size: 청크당 필드 수
            
        Returns:
            청크 리스트
        """
        chunks = []
        for i in range(0, len(preview_attrs), chunk_size):
            chunks.append(preview_attrs[i:i + chunk_size])
        return chunks
    
    def _chunk_ddl_fields(self, ddl_fields: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """
        ddlFields를 청크로 나누기
        
        Args:
            ddl_fields: ddlFields 리스트
            chunk_size: 청크당 필드 수
            
        Returns:
            청크 리스트
        """
        chunks = []
        for i in range(0, len(ddl_fields), chunk_size):
            chunks.append(ddl_fields[i:i + chunk_size])
        return chunks
    
    def _chunk_enumerations(self, enumerations: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """
        enumerations를 청크로 나누기
        
        Args:
            enumerations: enumerations 리스트
            chunk_size: 청크당 항목 수
            
        Returns:
            청크 리스트
        """
        chunks = []
        for i in range(0, len(enumerations), chunk_size):
            chunks.append(enumerations[i:i + chunk_size])
        return chunks
    
    def _chunk_value_objects(self, value_objects: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """
        valueObjects를 청크로 나누기
        
        Args:
            value_objects: valueObjects 리스트
            chunk_size: 청크당 항목 수
            
        Returns:
            청크 리스트
        """
        chunks = []
        for i in range(0, len(value_objects), chunk_size):
            chunks.append(value_objects[i:i + chunk_size])
        return chunks
    
    def _transform_structure_with_chunking(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        original_structure_item: Optional[Dict] = None,
        mapping_context: Optional[StandardMappingContext] = None,
        update_progress_callback: Optional[callable] = None,
        estimated_prompt_tokens: int = 0,
        bc_name: Optional[str] = None,
        agg_name: Optional[str] = None,
        original_option_bounded_context: Optional[Dict] = None  # 원본 option의 boundedContext 정보 (청킹 처리용)
    ) -> Dict:
        """
        청킹 처리를 통한 structure 변환 (대량 필드 처리용)
        
        전략:
        1. Aggregate, Enum, VO는 첫 번째 호출에서 처리
        2. previewAttributes와 ddlFields는 청크로 나누어 각각 처리
        3. 모든 응답을 병합
        
        Args:
            structure_item: 변환할 단일 structure 항목
            bounded_context: Bounded Context 정보
            relevant_standards: 검색된 표준 청크들
            query_search_results: 쿼리별 검색 결과 (top-k=3)
            original_structure_item: 원본 structure 항목 (선처리 전)
            mapping_context: 선처리 매핑 컨텍스트
            update_progress_callback: 진행 상황 업데이트 콜백
            estimated_prompt_tokens: 예상 프롬프트 토큰 수
            
        Returns:
            변환된 structure 항목
        """
        import copy
        # bc_name과 agg_name이 없으면 structure_item에서 추출 (클로저를 위해 로컬 변수로 저장)
        if not bc_name:
            bc_name = bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
        if not agg_name:
            agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
        
        bc_name_val = bc_name
        agg_name_val = agg_name
        
        enumerations = structure_item.get("enumerations", [])
        value_objects = structure_item.get("valueObjects", [])
        preview_attrs = structure_item.get("previewAttributes", [])
        ddl_fields = structure_item.get("ddlFields", [])
        
        # 청크 크기 결정: 프롬프트 토큰 수에 따라 조정
        # 안전한 범위: 프롬프트 15000 토큰 + 응답 10000 토큰 = 총 25000 토큰 (여유 있게)
        # 각 아이템(enum/vo/field)당 약 100 토큰, 검색 결과 3개당 약 500 토큰, 프롬프트 기본 약 2000 토큰
        # 예: 10개 아이템 = 1000 + 500 + 2000 = 3500 토큰 (프롬프트), 응답 1000 토큰 = 총 4500 토큰
        base_chunk_size = 10  # 더 안전한 기본값
        if estimated_prompt_tokens > 20000:
            chunk_size = 5  # 매우 작은 청크
        elif estimated_prompt_tokens > 15000:
            chunk_size = 8
        elif estimated_prompt_tokens > 10000:
            chunk_size = 10
        else:
            chunk_size = 12  # 토큰이 적으면 조금 더 큰 청크 가능
        
        LoggingUtil.info("StandardTransformer", 
                       f"      📦 [청킹] 청크 크기: {chunk_size}개 (예상 프롬프트 토큰: {estimated_prompt_tokens})")
        
        # 1단계: Aggregate만 변환 (enum, vo, 필드 없이)
        LoggingUtil.info("StandardTransformer", 
                       f"      📦 [청킹] 1단계: Aggregate만 변환 시작")
        
        structure_item_agg_only = {
            "aggregate": structure_item.get("aggregate", {}),
            "enumerations": [],
            "valueObjects": [],
            "previewAttributes": [],
            "ddlFields": []
        }
        
        # Aggregate 관련 쿼리만 필터링
        agg_name_val = agg_name or structure_item.get("aggregate", {}).get("name", "")
        agg_alias = structure_item.get("aggregate", {}).get("alias", "")
        agg_related_queries = set()
        
        if agg_name_val:
            for qr in query_search_results:
                query = qr.get("query", "")
                query_lower = query.lower()
                # Aggregate 이름이나 alias가 쿼리에 포함되어 있거나, 쿼리가 Aggregate 이름에 포함되어 있는지 확인
                if (agg_name_val.lower() in query_lower or 
                    query_lower in agg_name_val.lower() or
                    (agg_alias and (agg_alias in query or query in agg_alias))):
                    agg_related_queries.add(query)
        
        # Aggregate 관련 쿼리가 없으면 전체 쿼리의 일부만 사용 (최대 3개 쿼리, 각 쿼리별 top-3)
        if not agg_related_queries and query_search_results:
            # Aggregate 관련 쿼리가 없으면 일반적인 쿼리 중 최대 3개만 사용
            unique_queries = list(set([qr.get("query", "") for qr in query_search_results]))[:3]
            agg_query_results = []
            for query in unique_queries:
                query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                if query_results:
                    qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                    # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                    if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                        qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                    agg_query_results.append(qr_item)
            LoggingUtil.info("StandardTransformer", 
                           f"      📋 Aggregate 관련 쿼리 없음, 일반 쿼리 {len(agg_query_results)}개 사용 (최대 3개 쿼리, 각 top-3)")
        else:
            # 관련 쿼리 중 최대 3개만 선택
            agg_related_queries_list = list(agg_related_queries)[:3]
            # 각 쿼리별로 top-3 결과 사용 (k_per_query=3이므로 최대 3개)
            agg_query_results = []
            for query in agg_related_queries_list:
                query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                if query_results:
                    qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                    # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                    if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                        qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                    agg_query_results.append(qr_item)
            LoggingUtil.info("StandardTransformer", 
                           f"      📋 Aggregate 관련 쿼리 {len(agg_query_results)}개 필터링됨 (최대 3개 쿼리, 각 top-3, 전체 {len(query_search_results)}개 중)")
        
        # bc_name과 agg_name을 포함한 콜백 래퍼 생성
        def _update_progress_with_chunking_context(progress: int, stage: str, 
                                                   property_type: Optional[str] = None,
                                                   chunk_info: Optional[str] = None,
                                                   status: str = "processing",
                                                   error_message: Optional[str] = None,
                                                   bc_name: Optional[str] = None,
                                                   agg_name: Optional[str] = None):
            bc_name_val = bc_name or bounded_context.get("name", bounded_context.get("boundedContext", "Unknown"))
            agg_name_val = agg_name or structure_item.get("aggregate", {}).get("name", "Unknown")
            
            if update_progress_callback:
                try:
                    update_progress_callback(progress, stage,
                                            bc_name=bc_name or bc_name_val,
                                            agg_name=agg_name or agg_name_val,
                                            property_type=property_type,
                                            chunk_info=chunk_info,
                                            status=status,
                                            error_message=error_message)
                except Exception as e:
                    LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
        
        base_result = self._transform_single_structure_with_llm(
            structure_item=structure_item_agg_only,
            bounded_context=bounded_context,
            relevant_standards=relevant_standards,
            query_search_results=agg_query_results,  # Aggregate 관련 검색 결과만
            original_structure_item=original_structure_item,
            mapping_context=mapping_context,
            update_progress_callback=_update_progress_with_chunking_context,
            skip_chunking=True,  # 청킹 내부 호출이므로 무한 재귀 방지
            bc_name=bc_name,
            agg_name=agg_name,
            original_option_bounded_context=original_option_bounded_context
        )
        
        # 2단계: enumerations 청크별 변환
        enum_chunks = self._chunk_enumerations(enumerations, chunk_size)
        transformed_enums = []
        
        if enum_chunks:
            LoggingUtil.info("StandardTransformer", 
                           f"      📦 [청킹] 2단계: enumerations {len(enumerations)}개를 {len(enum_chunks)}개 청크로 분할")
            
            for chunk_idx, chunk in enumerate(enum_chunks):
                if update_progress_callback:
                    try:
                        update_progress_callback(0, f"LLM 변환 중: {agg_name_val} (Enum 청크 {chunk_idx + 1}/{len(enum_chunks)})",
                                                bc_name=bc_name_val,
                                                agg_name=agg_name_val,
                                                property_type="enum",
                                                chunk_info=f"청크 {chunk_idx + 1}/{len(enum_chunks)}",
                                                status="processing")
                    except Exception as e:
                        LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                LoggingUtil.info("StandardTransformer", 
                               f"      📦 [청킹] enumerations 청크 {chunk_idx + 1}/{len(enum_chunks)} 처리 중 ({len(chunk)}개)")
                
                # Enum만 포함한 최소 structure 생성
                chunk_structure = {
                    "aggregate": {
                        "name": base_result.get("aggregate", {}).get("name", ""),
                        "alias": base_result.get("aggregate", {}).get("alias", "")
                    },
                    "enumerations": chunk,
                    "valueObjects": [],
                    "previewAttributes": [],
                    "ddlFields": []
                }
                
                # 청크에 관련된 검색 결과만 필터링 (각 쿼리별 top-1만 사용, 최대 3개 쿼리)
                chunk_related_queries = set()
                for enum in chunk:
                    if isinstance(enum, dict):
                        enum_name = enum.get("name", "")
                        if enum_name:
                            for qr in query_search_results:
                                query = qr.get("query", "")
                                if enum_name.lower() in query.lower() or query.lower() in enum_name.lower():
                                    chunk_related_queries.add(query)
                
                # 관련 쿼리 중 최대 3개만 선택
                chunk_related_queries_list = list(chunk_related_queries)[:3]
                # 각 쿼리별로 top-3 결과 사용 (k_per_query=3이므로 최대 3개)
                chunk_query_results = []
                for query in chunk_related_queries_list:
                    query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                    if query_results:
                        qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                        # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                        if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                            qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                        chunk_query_results.append(qr_item)
                
                # Enum 전용 변환 (bc_name, agg_name 포함한 콜백 래퍼)
                def _enum_update_callback(progress: int, stage: str, 
                                          property_type: Optional[str] = None,
                                          chunk_info: Optional[str] = None,
                                          status: str = "processing",
                                          error_message: Optional[str] = None):
                    if update_progress_callback:
                        try:
                            update_progress_callback(progress, stage,
                                                    property_type=property_type or "enum",
                                                    chunk_info=chunk_info or f"청크 {chunk_idx + 1}/{len(enum_chunks)}",
                                                    status=status,
                                                    error_message=error_message)
                        except Exception as e:
                            LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                chunk_result = self._transform_enums_vos_only_with_llm(
                    structure_item=chunk_structure,
                    bounded_context=bounded_context,
                    relevant_standards=relevant_standards,
                    query_search_results=chunk_query_results,
                    item_type="enumerations",
                    update_progress_callback=_enum_update_callback
                )
                
                # 변환된 enumerations 추출 및 병합
                chunk_transformed_enums = chunk_result.get("enumerations", [])
                transformed_enums.extend(chunk_transformed_enums)
        
        # 3단계: valueObjects 청크별 변환
        vo_chunks = self._chunk_value_objects(value_objects, chunk_size)
        transformed_vos = []
        
        if vo_chunks:
            LoggingUtil.info("StandardTransformer", 
                           f"      📦 [청킹] 3단계: valueObjects {len(value_objects)}개를 {len(vo_chunks)}개 청크로 분할")
            
            for chunk_idx, chunk in enumerate(vo_chunks):
                if update_progress_callback:
                    try:
                        update_progress_callback(0, f"LLM 변환 중: {agg_name} (VO 청크 {chunk_idx + 1}/{len(vo_chunks)})",
                                                property_type="vo",
                                                chunk_info=f"청크 {chunk_idx + 1}/{len(vo_chunks)}",
                                                status="processing")
                    except Exception as e:
                        LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                LoggingUtil.info("StandardTransformer", 
                               f"      📦 [청킹] valueObjects 청크 {chunk_idx + 1}/{len(vo_chunks)} 처리 중 ({len(chunk)}개)")
                
                # VO만 포함한 최소 structure 생성
                chunk_structure = {
                    "aggregate": {
                        "name": base_result.get("aggregate", {}).get("name", ""),
                        "alias": base_result.get("aggregate", {}).get("alias", "")
                    },
                    "enumerations": [],
                    "valueObjects": chunk,
                    "previewAttributes": [],
                    "ddlFields": []
                }
                
                # 청크에 관련된 검색 결과만 필터링 (각 쿼리별 top-1만 사용, 최대 3개 쿼리)
                chunk_related_queries = set()
                for vo in chunk:
                    if isinstance(vo, dict):
                        vo_name = vo.get("name", "")
                        if vo_name:
                            for qr in query_search_results:
                                query = qr.get("query", "")
                                if vo_name.lower() in query.lower() or query.lower() in vo_name.lower():
                                    chunk_related_queries.add(query)
                
                # 관련 쿼리 중 최대 3개만 선택
                chunk_related_queries_list = list(chunk_related_queries)[:3]
                # 각 쿼리별로 top-3 결과 사용 (k_per_query=3이므로 최대 3개)
                chunk_query_results = []
                for query in chunk_related_queries_list:
                    query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                    if query_results:
                        qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                        # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                        if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                            qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                        chunk_query_results.append(qr_item)
                
                # VO 전용 변환 (bc_name, agg_name 포함한 콜백 래퍼)
                def _vo_update_callback(progress: int, stage: str, 
                                        property_type: Optional[str] = None,
                                        chunk_info: Optional[str] = None,
                                        status: str = "processing",
                                        error_message: Optional[str] = None):
                    if update_progress_callback:
                        try:
                            update_progress_callback(progress, stage,
                                                    property_type=property_type or "vo",
                                                    chunk_info=chunk_info or f"청크 {chunk_idx + 1}/{len(vo_chunks)}",
                                                    status=status,
                                                    error_message=error_message)
                        except Exception as e:
                            LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                chunk_result = self._transform_enums_vos_only_with_llm(
                    structure_item=chunk_structure,
                    bounded_context=bounded_context,
                    relevant_standards=relevant_standards,
                    query_search_results=chunk_query_results,
                    item_type="valueObjects",
                    update_progress_callback=_vo_update_callback
                )
                
                # 변환된 valueObjects 추출 및 병합
                chunk_transformed_vos = chunk_result.get("valueObjects", [])
                transformed_vos.extend(chunk_transformed_vos)
        
        # 4단계: previewAttributes 청크별 변환
        preview_attr_chunks = self._chunk_preview_attributes(preview_attrs, chunk_size)
        transformed_preview_attrs = []
        
        if preview_attr_chunks:
            LoggingUtil.info("StandardTransformer", 
                           f"      📦 [청킹] 2단계: previewAttributes {len(preview_attrs)}개를 {len(preview_attr_chunks)}개 청크로 분할")
            
            for chunk_idx, chunk in enumerate(preview_attr_chunks):
                if update_progress_callback:
                    try:
                        update_progress_callback(0, f"LLM 변환 중: {agg_name_val} (필드 청크 {chunk_idx + 1}/{len(preview_attr_chunks)})",
                                                bc_name=bc_name_val,
                                                agg_name=agg_name_val,
                                                property_type="field",
                                                chunk_info=f"청크 {chunk_idx + 1}/{len(preview_attr_chunks)}",
                                                status="processing")
                    except Exception as e:
                        LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                LoggingUtil.info("StandardTransformer", 
                               f"      📦 [청킹] previewAttributes 청크 {chunk_idx + 1}/{len(preview_attr_chunks)} 처리 중 ({len(chunk)}개 필드)")
                
                # 필드만 포함한 최소 structure 생성 (Agg 정보는 최소한만 포함)
                chunk_structure = {
                    "aggregate": {
                        "name": structure_item.get("aggregate", {}).get("name", ""),
                        "alias": structure_item.get("aggregate", {}).get("alias", "")
                    },
                    "previewAttributes": chunk,
                    "enumerations": [],  # 필드 변환에는 불필요
                    "valueObjects": [],  # 필드 변환에는 불필요
                    "ddlFields": []  # previewAttributes만 처리
                }
                
                # 청크에 관련된 검색 결과만 필터링 (각 쿼리별 top-1만 사용, 최대 3개 쿼리)
                chunk_related_queries = set()
                for attr in chunk:
                    if isinstance(attr, dict):
                        field_name = attr.get("fieldName", "")
                        if field_name:
                            # 필드명과 관련된 쿼리 찾기
                            for qr in query_search_results:
                                query = qr.get("query", "")
                                if field_name.lower() in query.lower() or query.lower() in field_name.lower():
                                    chunk_related_queries.add(query)
                
                # 관련 쿼리 중 최대 3개만 선택
                chunk_related_queries_list = list(chunk_related_queries)[:3]
                # 각 쿼리별로 top-3 결과 사용 (k_per_query=3이므로 최대 3개)
                chunk_query_results = []
                for query in chunk_related_queries_list:
                    query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                    if query_results:
                        qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                        # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                        if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                            qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                        chunk_query_results.append(qr_item)
                
                # 필드 전용 변환 (bc_name, agg_name 포함한 콜백 래퍼)
                def _field_update_callback(progress: int, stage: str, 
                                          property_type: Optional[str] = None,
                                          chunk_info: Optional[str] = None,
                                          status: str = "processing",
                                          error_message: Optional[str] = None):
                    if update_progress_callback:
                        try:
                            update_progress_callback(progress, stage,
                                                    property_type=property_type or "field",
                                                    chunk_info=chunk_info or f"청크 {chunk_idx + 1}/{len(preview_attr_chunks)}",
                                                    status=status,
                                                    error_message=error_message)
                        except Exception as e:
                            LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                chunk_result = self._transform_fields_only_with_llm(
                    structure_item=chunk_structure,
                    bounded_context=bounded_context,
                    relevant_standards=relevant_standards,
                    query_search_results=chunk_query_results,
                    field_type="previewAttributes",
                    update_progress_callback=_field_update_callback
                )
                
                # 변환된 previewAttributes 추출 및 병합
                chunk_transformed_attrs = chunk_result.get("previewAttributes", [])
                transformed_preview_attrs.extend(chunk_transformed_attrs)
        
        # 5단계: ddlFields 청크별 변환
        ddl_field_chunks = self._chunk_ddl_fields(ddl_fields, chunk_size)
        transformed_ddl_fields = []
        
        if ddl_field_chunks:
            LoggingUtil.info("StandardTransformer", 
                           f"      📦 [청킹] 5단계: ddlFields {len(ddl_fields)}개를 {len(ddl_field_chunks)}개 청크로 분할")
            
            for chunk_idx, chunk in enumerate(ddl_field_chunks):
                if update_progress_callback:
                    try:
                        update_progress_callback(0, f"LLM 변환 중: {agg_name_val} (DDL 필드 청크 {chunk_idx + 1}/{len(ddl_field_chunks)})",
                                                bc_name=bc_name_val,
                                                agg_name=agg_name_val,
                                                property_type="field",
                                                chunk_info=f"DDL 청크 {chunk_idx + 1}/{len(ddl_field_chunks)}",
                                                status="processing")
                    except Exception as e:
                        LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                LoggingUtil.info("StandardTransformer", 
                               f"      📦 [청킹] ddlFields 청크 {chunk_idx + 1}/{len(ddl_field_chunks)} 처리 중 ({len(chunk)}개 필드)")
                
                # 필드만 포함한 최소 structure 생성 (Agg 정보는 최소한만 포함)
                chunk_structure = {
                    "aggregate": {
                        "name": structure_item.get("aggregate", {}).get("name", ""),
                        "alias": structure_item.get("aggregate", {}).get("alias", "")
                    },
                    "ddlFields": chunk,
                    "enumerations": [],  # 필드 변환에는 불필요
                    "valueObjects": [],  # 필드 변환에는 불필요
                    "previewAttributes": []  # ddlFields만 처리
                }
                
                # 청크에 관련된 검색 결과만 필터링 (각 쿼리별 top-1만 사용, 최대 3개 쿼리)
                chunk_related_queries = set()
                for field in chunk:
                    if isinstance(field, dict):
                        field_name = field.get("fieldName", "")
                        if field_name:
                            # 필드명과 관련된 쿼리 찾기
                            for qr in query_search_results:
                                query = qr.get("query", "")
                                if field_name.lower() in query.lower() or query.lower() in field_name.lower():
                                    chunk_related_queries.add(query)
                
                # 관련 쿼리 중 최대 3개만 선택
                chunk_related_queries_list = list(chunk_related_queries)[:3]
                # 각 쿼리별로 top-3 결과 사용 (k_per_query=3이므로 최대 3개)
                chunk_query_results = []
                for query in chunk_related_queries_list:
                    query_results = [qr for qr in query_search_results if qr.get("query", "") == query]
                    if query_results:
                        qr_item = query_results[0].copy()  # 첫 번째 쿼리 결과 사용
                        # 🔒 CRITICAL: results 리스트는 top-3 유지 (k_per_query=3이므로 최대 3개)
                        if "results" in qr_item and isinstance(qr_item["results"], list) and len(qr_item["results"]) > 3:
                            qr_item["results"] = qr_item["results"][:3]  # top-3만 유지
                        chunk_query_results.append(qr_item)
                
                # DDL 필드 전용 변환 (bc_name, agg_name 포함한 콜백 래퍼)
                def _ddl_field_update_callback(progress: int, stage: str, 
                                               property_type: Optional[str] = None,
                                               chunk_info: Optional[str] = None,
                                               status: str = "processing",
                                               error_message: Optional[str] = None):
                    if update_progress_callback:
                        try:
                            update_progress_callback(progress, stage,
                                                    property_type=property_type or "field",
                                                    chunk_info=chunk_info or f"DDL 청크 {chunk_idx + 1}/{len(ddl_field_chunks)}",
                                                    status=status,
                                                    error_message=error_message)
                        except Exception as e:
                            LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
                
                chunk_result = self._transform_fields_only_with_llm(
                    structure_item=chunk_structure,
                    bounded_context=bounded_context,
                    relevant_standards=relevant_standards,
                    query_search_results=chunk_query_results,
                    field_type="ddlFields",
                    update_progress_callback=_ddl_field_update_callback
                )
                
                # 변환된 ddlFields 추출 및 병합
                chunk_transformed_ddl = chunk_result.get("ddlFields", [])
                transformed_ddl_fields.extend(chunk_transformed_ddl)
        
        # 6단계: 모든 결과 병합
        LoggingUtil.info("StandardTransformer", 
                       f"      📦 [청킹] 4단계: 모든 청크 결과 병합 중")
        
        # 🔒 CRITICAL: original_structure_item에서 원본 구조 가져오기 (refs 포함)
        if original_structure_item:
            merged_result = copy.deepcopy(original_structure_item)
            # 변환된 aggregate 이름만 업데이트
            if "aggregate" in base_result and "aggregate" in merged_result:
                merged_result["aggregate"]["name"] = base_result["aggregate"].get("name", merged_result["aggregate"].get("name"))
                merged_result["aggregate"]["alias"] = base_result["aggregate"].get("alias", merged_result["aggregate"].get("alias"))
        else:
            merged_result = copy.deepcopy(base_result)
        
        # 🔒 CRITICAL: original_structure_item에서 원본 enumerations, valueObjects, previewAttributes, ddlFields 가져오기
        original_enumerations = []
        original_value_objects = []
        original_preview_attrs = []
        original_ddl_fields = []
        if original_structure_item:
            original_enumerations = original_structure_item.get("enumerations", [])
            original_value_objects = original_structure_item.get("valueObjects", [])
            original_preview_attrs = original_structure_item.get("previewAttributes", [])
            original_ddl_fields = original_structure_item.get("ddlFields", [])
        
        # enumerations 병합 (refs 복원)
        if transformed_enums and len(transformed_enums) == len(enumerations):
            merged_enums = []
            for i, trans_enum in enumerate(transformed_enums):
                if i < len(original_enumerations):
                    # original_structure_item에서 원본 enum 가져오기 (refs 포함)
                    merged_enum = copy.deepcopy(original_enumerations[i])
                    # 변환된 이름만 업데이트
                    if "name" in trans_enum:
                        merged_enum["name"] = trans_enum["name"]
                    if "alias" in trans_enum:
                        merged_enum["alias"] = trans_enum["alias"]
                    merged_enums.append(merged_enum)
                else:
                    merged_enums.append(trans_enum)
            merged_result["enumerations"] = merged_enums
        else:
            # 변환 실패 시 original_structure_item에서 복원
            if original_structure_item:
                merged_result["enumerations"] = copy.deepcopy(original_enumerations)
            else:
                merged_result["enumerations"] = enumerations
            if transformed_enums:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [청킹] enumerations 병합 실패: 원본 {len(enumerations)}개, 변환 {len(transformed_enums)}개")
        
        # valueObjects 병합 (refs 복원)
        if transformed_vos and len(transformed_vos) == len(value_objects):
            merged_vos = []
            for i, trans_vo in enumerate(transformed_vos):
                if i < len(original_value_objects):
                    # original_structure_item에서 원본 VO 가져오기 (refs 포함)
                    merged_vo = copy.deepcopy(original_value_objects[i])
                    # 변환된 이름만 업데이트
                    if "name" in trans_vo:
                        merged_vo["name"] = trans_vo["name"]
                    if "alias" in trans_vo:
                        merged_vo["alias"] = trans_vo["alias"]
                    if "referencedAggregateName" in trans_vo:
                        merged_vo["referencedAggregateName"] = trans_vo["referencedAggregateName"]
                    merged_vos.append(merged_vo)
                else:
                    merged_vos.append(trans_vo)
            merged_result["valueObjects"] = merged_vos
        else:
            # 변환 실패 시 original_structure_item에서 복원
            if original_structure_item:
                merged_result["valueObjects"] = copy.deepcopy(original_value_objects)
            else:
                merged_result["valueObjects"] = value_objects
            if transformed_vos:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [청킹] valueObjects 병합 실패: 원본 {len(value_objects)}개, 변환 {len(transformed_vos)}개")
        
        # previewAttributes 병합 (fieldAlias 기반 매칭 - 인덱스 기반은 LLM이 순서를 바꿀 수 있어 불안정)
        if transformed_preview_attrs and len(transformed_preview_attrs) == len(preview_attrs):
            # 원본 필드를 fieldAlias로 인덱싱 (fieldAlias는 변환되지 않으므로 안전한 키)
            original_attrs_by_alias = {}  # {fieldAlias: original_attr}
            for orig_attr in original_preview_attrs:
                if isinstance(orig_attr, dict):
                    field_alias = orig_attr.get("fieldAlias")
                    if field_alias:
                        original_attrs_by_alias[field_alias] = orig_attr
            
            # 변환된 필드를 fieldAlias로 매칭하여 병합
            merged_preview_attrs = []
            for transformed_attr in transformed_preview_attrs:
                if isinstance(transformed_attr, dict):
                    trans_field_name = transformed_attr.get("fieldName")
                    trans_field_alias = transformed_attr.get("fieldAlias")
                    
                    # fieldAlias로 매칭 시도 (가장 안전 - 변환되지 않음)
                    matched_original = None
                    if trans_field_alias and trans_field_alias in original_attrs_by_alias:
                        matched_original = original_attrs_by_alias[trans_field_alias]
                    elif len(merged_preview_attrs) < len(original_preview_attrs):
                        # fieldAlias 매칭 실패 시 인덱스 기반 fallback (순서가 같다고 가정)
                        idx = len(merged_preview_attrs)
                        if idx < len(original_preview_attrs):
                            candidate = original_preview_attrs[idx]
                            if isinstance(candidate, dict):
                                # 원본의 fieldAlias가 transformed_attr의 fieldAlias와 일치하는지 확인
                                orig_field_alias = candidate.get("fieldAlias")
                                if not trans_field_alias or orig_field_alias == trans_field_alias:
                                    matched_original = candidate
                    
                    # 매칭된 원본이 있으면 복사하고 fieldName만 업데이트
                    if matched_original:
                        merged_attr = copy.deepcopy(matched_original)
                        merged_attr["fieldName"] = trans_field_name
                        merged_preview_attrs.append(merged_attr)
                    else:
                        # 매칭 실패 시 transformed_attr 그대로 사용 (refs는 원본에서 복원 불가)
                        merged_preview_attrs.append(transformed_attr)
                else:
                    merged_preview_attrs.append(transformed_attr)
            merged_result["previewAttributes"] = merged_preview_attrs
        else:
            # 변환 실패 시 original_structure_item에서 복원
            if original_structure_item:
                merged_result["previewAttributes"] = copy.deepcopy(original_preview_attrs)
            else:
                merged_result["previewAttributes"] = preview_attrs
            if transformed_preview_attrs:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [청킹] previewAttributes 병합 실패: 원본 {len(preview_attrs)}개, 변환 {len(transformed_preview_attrs)}개")
        
        # ddlFields 병합 (fieldAlias 기반 매칭 - 인덱스 기반은 LLM이 순서를 바꿀 수 있어 불안정)
        if transformed_ddl_fields and len(transformed_ddl_fields) == len(ddl_fields):
            # 원본 필드를 fieldAlias로 인덱싱 (fieldAlias는 변환되지 않으므로 안전한 키)
            original_ddl_by_alias = {}  # {fieldAlias: original_field}
            for orig_field in original_ddl_fields:
                if isinstance(orig_field, dict):
                    field_alias = orig_field.get("fieldAlias")
                    if field_alias:
                        original_ddl_by_alias[field_alias] = orig_field
            
            # 변환된 필드를 fieldAlias로 매칭하여 병합
            merged_ddl_fields = []
            for transformed_field in transformed_ddl_fields:
                if isinstance(transformed_field, dict):
                    trans_field_name = transformed_field.get("fieldName")
                    trans_field_alias = transformed_field.get("fieldAlias")
                    
                    # fieldAlias로 매칭 시도 (가장 안전 - 변환되지 않음)
                    matched_original = None
                    if trans_field_alias and trans_field_alias in original_ddl_by_alias:
                        matched_original = original_ddl_by_alias[trans_field_alias]
                    elif len(merged_ddl_fields) < len(original_ddl_fields):
                        # fieldAlias 매칭 실패 시 인덱스 기반 fallback (순서가 같다고 가정)
                        idx = len(merged_ddl_fields)
                        if idx < len(original_ddl_fields):
                            candidate = original_ddl_fields[idx]
                            if isinstance(candidate, dict):
                                # 원본의 fieldAlias가 transformed_field의 fieldAlias와 일치하는지 확인
                                orig_field_alias = candidate.get("fieldAlias")
                                if not trans_field_alias or orig_field_alias == trans_field_alias:
                                    matched_original = candidate
                    
                    # 매칭된 원본이 있으면 복사하고 fieldName만 업데이트
                    if matched_original:
                        merged_field = copy.deepcopy(matched_original)
                        merged_field["fieldName"] = trans_field_name
                        merged_ddl_fields.append(merged_field)
                    else:
                        # 매칭 실패 시 transformed_field 그대로 사용 (refs는 원본에서 복원 불가)
                        merged_ddl_fields.append(transformed_field)
                else:
                    merged_ddl_fields.append(transformed_field)
            merged_result["ddlFields"] = merged_ddl_fields
        else:
            # 변환 실패 시 original_structure_item에서 복원
            if original_structure_item:
                merged_result["ddlFields"] = copy.deepcopy(original_ddl_fields)
            else:
                merged_result["ddlFields"] = ddl_fields
            if transformed_ddl_fields:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [청킹] ddlFields 병합 실패: 원본 {len(ddl_fields)}개, 변환 {len(transformed_ddl_fields)}개")
        
        LoggingUtil.info("StandardTransformer", 
                       f"      ✅ [청킹] 모든 청크 처리 완료: previewAttributes={len(merged_result.get('previewAttributes', []))}개, ddlFields={len(merged_result.get('ddlFields', []))}개")
        
        # 청킹 완료 알림
        if update_progress_callback:
            try:
                update_progress_callback(100, f"청킹 처리 완료: {agg_name_val}",
                                        bc_name=bc_name_val,
                                        agg_name=agg_name_val,
                                        property_type="aggregate",
                                        status="completed")
            except Exception as e:
                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {e}")
        
        return merged_result
    
    def _transform_fields_only_with_llm(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        field_type: str,  # "previewAttributes" or "ddlFields"
        update_progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        필드만 변환하는 전용 메서드 (Agg 정보는 이미 변환됨, 필드와 검색 결과만 청킹 처리)
        
        Args:
            structure_item: 필드만 포함한 최소 structure (aggregate.name, aggregate.alias, 필드 배열만)
            bounded_context: Bounded Context 정보
            relevant_standards: 검색된 표준 청크들
            query_search_results: 필터링된 검색 결과 (청크 관련 쿼리만)
            field_type: "previewAttributes" or "ddlFields"
            update_progress_callback: 진행 상황 업데이트 콜백
            
        Returns:
            변환된 필드 배열을 포함한 structure
        """
        # 필드 전용 프롬프트 생성
        prompt = self._build_field_transformation_prompt(
            structure_item=structure_item,
            bounded_context=bounded_context,
            relevant_standards=relevant_standards,
            query_search_results=query_search_results,
            field_type=field_type
        )
        
        # 필드 전용 스키마 생성
        field_schema = self._get_field_response_schema(field_type)
        
        # LLM 호출
        try:
            agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
            field_count = len(structure_item.get(field_type, []))
            
            LoggingUtil.info("StandardTransformer", 
                           f"      📤 [필드 전용] LLM API 호출 시작: {field_type} {field_count}개")
            
            # 필드 전용 structured output 생성
            llm_structured = self.llm.with_structured_output(field_schema)
            
            max_retries = 2
            retry_count = 0
            response = None
            
            while retry_count <= max_retries:
                try:
                    response = llm_structured.invoke(prompt)
                    break
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    LoggingUtil.warning("StandardTransformer", 
                                      f"      ⚠️  [필드 전용] LLM API 호출 실패 (재시도 {retry_count}/{max_retries}): {e}")
                    
                    if retry_count > max_retries:
                        if update_progress_callback:
                            try:
                                update_progress_callback(0, f"❌ 필드 변환 실패: {field_type} (최대 재시도 횟수 초과)",
                                                        property_type="field",
                                                        status="error",
                                                        error_message=f"최대 재시도 횟수 초과: {error_msg}")
                            except Exception as update_e:
                                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                        raise
                    
                    import time
                    time.sleep(2)
            
            LoggingUtil.info("StandardTransformer", 
                           f"      📥 [필드 전용] LLM API 응답 수신 완료")
            
            result = response.get("result", {})
            transformed_fields = result.get(field_type, [])
            
            if not transformed_fields:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [필드 전용] LLM 응답에 {field_type}가 없음, 원본 반환")
                return structure_item
            
            # 변환된 필드만 반환
            import copy
            result_structure = copy.deepcopy(structure_item)
            result_structure[field_type] = transformed_fields
            
            return result_structure
            
        except Exception as e:
            LoggingUtil.error("StandardTransformer", 
                            f"❌ [필드 전용] LLM 호출 실패: {e}")
            import traceback
            LoggingUtil.error("StandardTransformer", traceback.format_exc())
            if update_progress_callback:
                try:
                    update_progress_callback(0, f"❌ 필드 변환 실패: {field_type}",
                                            property_type="field",
                                            status="error",
                                            error_message=str(e))
                except Exception as update_e:
                    LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
            return structure_item
    
    def _build_field_transformation_prompt(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        field_type: str
    ) -> str:
        """
        필드 전용 변환 프롬프트 생성 (Agg 정보는 최소한만 포함, 필드와 검색 결과만 집중)
        """
        # 표준 문서 포맷팅 (기존과 동일)
        standards_text = ""
        
        # 쿼리별 검색 결과 변환
        transformed_query_results = {}
        if query_search_results:
            for qr in query_search_results:
                query = qr.get("query", "")
                if not query:
                    continue
                
                if "results" in qr:
                    results_list = qr["results"]
                    query_results = []
                    for result_item in results_list:
                        query_results.append(result_item.get("result", {}))
                    transformed_query_results[query] = query_results
        
        if transformed_query_results:
            standards_text += "\n## Standard Transformation Reference (Field-Related Queries Only):\n"
            standards_text += json.dumps(transformed_query_results, ensure_ascii=False, indent=2)
        
        # Aggregate 정보 (최소한만)
        agg_name = structure_item.get("aggregate", {}).get("name", "")
        agg_alias = structure_item.get("aggregate", {}).get("alias", "")
        
        # 필드 정보
        fields = structure_item.get(field_type, [])
        fields_json = json.dumps(fields, ensure_ascii=False, indent=2)
        
        bc_name = bounded_context.get("name", "")
        
        prompt = f"""You are a Standard Naming Transformer specialized in field name transformation.

## Task:
Transform ONLY the `fieldName` values in the `{field_type}` array based on the "Standard Transformation Reference" below.
**Aggregate information is already transformed - DO NOT modify aggregate.name or aggregate.alias.**

## Context:
- **Bounded Context**: {bc_name}
- **Aggregate**: {agg_name} ({agg_alias})

## Input Structure:
```json
{{
  "aggregate": {{
    "name": "{agg_name}",
    "alias": "{agg_alias}"
  }},
  "{field_type}": {fields_json}
}}
```

{standards_text}

## Instructions:
1. **ONLY transform `{field_type}[].fieldName`** - Check EVERY field in the array
2. **DO NOT modify** aggregate.name, aggregate.alias, or any other fields
3. **Preserve ALL fields** - Every field in input must exist in output
4. **Match and Transform**: For each `fieldName`, check if it matches a key in "Standard Transformation Reference"
   - If matched, use the most appropriate `표준명` from the candidate list
   - If no match, keep original unchanged
5. **Preserve fieldAlias**: Keep all `fieldAlias` values unchanged (they are Korean text for matching)

## Output Format:
Return JSON with the EXACT same structure as input, ONLY changing `fieldName` values that match the reference.

## CRITICAL:
- Return ALL fields from input (same count, same order)
- Only `fieldName` values may change
- All other fields (fieldAlias, className, type, etc.) must remain unchanged
"""
        
        return prompt
    
    def _get_field_response_schema(self, field_type: str) -> Dict:
        """
        필드 전용 응답 스키마 (필드 배열만 포함)
        """
        field_properties = {
            "fieldName": {"type": "string"},
            "fieldAlias": {"type": "string"}
        }
        
        if field_type == "ddlFields":
            field_properties["className"] = {"type": "string"}
            field_properties["type"] = {"type": "string"}
        
        return {
            "title": "FieldTransformationResponse",
            "description": "Response schema for field-only transformation",
            "type": "object",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Brief explanation of transformation process"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "aggregate": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "alias": {"type": "string"}
                            },
                            "required": ["name", "alias"],
                            "additionalProperties": True
                        },
                        field_type: {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": field_properties,
                                "required": ["fieldName"],
                                "additionalProperties": True
                            }
                        }
                    },
                    "required": ["aggregate", field_type],
                    "additionalProperties": True
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }
    
    def _transform_enums_vos_only_with_llm(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        item_type: str,  # "enumerations" or "valueObjects"
        update_progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Enum/VO만 변환하는 전용 메서드 (Agg 정보는 이미 변환됨, Enum/VO와 검색 결과만 청킹 처리)
        
        Args:
            structure_item: Enum/VO만 포함한 최소 structure (aggregate.name, aggregate.alias, enum/vo 배열만)
            bounded_context: Bounded Context 정보
            relevant_standards: 검색된 표준 청크들
            query_search_results: 필터링된 검색 결과 (청크 관련 쿼리만)
            item_type: "enumerations" or "valueObjects"
            update_progress_callback: 진행 상황 업데이트 콜백
            
        Returns:
            변환된 Enum/VO 배열을 포함한 structure
        """
        # Enum/VO 전용 프롬프트 생성
        prompt = self._build_enum_vo_transformation_prompt(
            structure_item=structure_item,
            bounded_context=bounded_context,
            relevant_standards=relevant_standards,
            query_search_results=query_search_results,
            item_type=item_type
        )
        
        # Enum/VO 전용 스키마 생성
        enum_vo_schema = self._get_enum_vo_response_schema(item_type)
        
        # LLM 호출
        try:
            agg_name = structure_item.get("aggregate", {}).get("name", "Unknown")
            item_count = len(structure_item.get(item_type, []))
            
            LoggingUtil.info("StandardTransformer", 
                           f"      📤 [Enum/VO 전용] LLM API 호출 시작: {item_type} {item_count}개")
            
            # Enum/VO 전용 structured output 생성
            llm_structured = self.llm.with_structured_output(enum_vo_schema)
            
            max_retries = 2
            retry_count = 0
            response = None
            
            while retry_count <= max_retries:
                try:
                    response = llm_structured.invoke(prompt)
                    break
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    LoggingUtil.warning("StandardTransformer", 
                                      f"      ⚠️  [Enum/VO 전용] LLM API 호출 실패 (재시도 {retry_count}/{max_retries}): {e}")
                    
                    if retry_count > max_retries:
                        if update_progress_callback:
                            try:
                                item_label = "Enum" if item_type == "enumerations" else "ValueObject"
                                update_progress_callback(0, f"❌ {item_label} 변환 실패 (최대 재시도 횟수 초과)",
                                                        property_type="enum" if item_type == "enumerations" else "vo",
                                                        status="error",
                                                        error_message=f"최대 재시도 횟수 초과: {error_msg}")
                            except Exception as update_e:
                                LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
                        raise
                    
                    import time
                    time.sleep(2)

            LoggingUtil.info("StandardTransformer", 
                           f"      📥 [Enum/VO 전용] LLM API 응답 수신 완료")
            
            result = response.get("result", {})
            transformed_items = result.get(item_type, [])
            
            if not transformed_items:
                LoggingUtil.warning("StandardTransformer", 
                                  f"⚠️  [Enum/VO 전용] LLM 응답에 {item_type}가 없음, 원본 반환")
                return structure_item
            
            # 변환된 Enum/VO만 반환
            import copy
            result_structure = copy.deepcopy(structure_item)
            result_structure[item_type] = transformed_items
            
            return result_structure
            
        except Exception as e:
            LoggingUtil.error("StandardTransformer", 
                            f"❌ [Enum/VO 전용] LLM 호출 실패: {e}")
            import traceback
            LoggingUtil.error("StandardTransformer", traceback.format_exc())
            if update_progress_callback:
                try:
                    item_label = "Enum" if item_type == "enumerations" else "ValueObject"
                    update_progress_callback(0, f"❌ {item_label} 변환 실패",
                                            property_type="enum" if item_type == "enumerations" else "vo",
                                            status="error",
                                            error_message=str(e))
                except Exception as update_e:
                    LoggingUtil.warning("StandardTransformer", f"진행 상황 업데이트 실패: {update_e}")
            return structure_item
    
    def _build_enum_vo_transformation_prompt(
        self,
        structure_item: Dict,
        bounded_context: Dict,
        relevant_standards: List[Dict],
        query_search_results: List[Dict],
        item_type: str
    ) -> str:
        """
        Enum/VO 전용 변환 프롬프트 생성 (Agg 정보는 최소한만 포함, Enum/VO와 검색 결과만 집중)
        """
        # 표준 문서 포맷팅
        standards_text = ""
        
        # 쿼리별 검색 결과 변환
        transformed_query_results = {}
        if query_search_results:
            for qr in query_search_results:
                query = qr.get("query", "")
                if not query:
                    continue
                
                if "results" in qr:
                    results_list = qr["results"]
                    query_results = []
                    for result_item in results_list:
                        query_results.append(result_item.get("result", {}))
                    transformed_query_results[query] = query_results
        
        if transformed_query_results:
            standards_text += "\n## Standard Transformation Reference (Related Queries Only):\n"
            standards_text += json.dumps(transformed_query_results, ensure_ascii=False, indent=2)
        
        # Aggregate 정보 (최소한만)
        agg_name = structure_item.get("aggregate", {}).get("name", "")
        agg_alias = structure_item.get("aggregate", {}).get("alias", "")
        
        # Enum/VO 정보
        items = structure_item.get(item_type, [])
        items_json = json.dumps(items, ensure_ascii=False, indent=2)
        
        bc_name = bounded_context.get("name", "")
        item_label = "enumerations" if item_type == "enumerations" else "value objects"
        
        prompt = f"""You are a Standard Naming Transformer specialized in {item_label} name transformation.

## Task:
Transform ONLY the `name` values in the `{item_type}` array based on the "Standard Transformation Reference" below.
**Aggregate information is already transformed - DO NOT modify aggregate.name or aggregate.alias.**

## Context:
- **Bounded Context**: {bc_name}
- **Aggregate**: {agg_name} ({agg_alias})

## Input Structure:
```json
{{
  "aggregate": {{
    "name": "{agg_name}",
    "alias": "{agg_alias}"
  }},
  "{item_type}": {items_json}
}}
```

{standards_text}

## Instructions:
1. **ONLY transform `{item_type}[].name`** - Check EVERY item in the array
2. **DO NOT modify** aggregate.name, aggregate.alias, or any other fields
3. **Preserve ALL items** - Every item in input must exist in output
4. **Match and Transform**: For each `name`, check if it matches a key in "Standard Transformation Reference"
   - If matched, use the most appropriate `표준명` from the candidate list
   - If no match, keep original unchanged
5. **Preserve alias**: Keep all `alias` values unchanged (they are Korean text for matching)
6. **Preserve referencedAggregateName**: For valueObjects, keep `referencedAggregateName` unchanged

## Output Format:
Return JSON with the EXACT same structure as input, ONLY changing `name` values that match the reference.

## CRITICAL:
- Return ALL items from input (same count, same order)
- Only `name` values may change
- All other fields (alias, referencedAggregateName, etc.) must remain unchanged
"""
        
        return prompt
    
    def _get_enum_vo_response_schema(self, item_type: str) -> Dict:
        """
        Enum/VO 전용 응답 스키마 (Enum/VO 배열만 포함)
        """
        item_properties = {
            "name": {"type": "string"},
            "alias": {"type": "string"}
        }
        
        if item_type == "valueObjects":
            item_properties["referencedAggregateName"] = {"type": "string"}
        
        return {
            "title": "EnumVOTransformationResponse",
            "description": "Response schema for enum/vo-only transformation",
            "type": "object",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Brief explanation of transformation process"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "aggregate": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "alias": {"type": "string"}
                            },
                            "required": ["name", "alias"],
                            "additionalProperties": True
                        },
                        item_type: {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": item_properties,
                                "required": ["name", "alias"] if item_type == "enumerations" else ["name", "alias", "referencedAggregateName"],
                                "additionalProperties": True
                            }
                        }
                    },
                    "required": ["aggregate", item_type],
                    "additionalProperties": True
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }

