"""
RAG Retriever - 공통 RAG 검색 모듈
모든 워크플로우에서 재사용 가능한 RAG 검색 기능 제공
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # 새로운 패키지로 import 시도 (deprecation warning 해결)
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        # fallback: 기존 패키지 사용
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
    # Document는 langchain_core에서 import
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠️  chromadb not installed. RAG features will be disabled.")

from src.project_generator.config import Config

# 기본 유사도 임계값 (0.0~1.0)
# 자연어 + 도메인 텍스트에서 코사인 기반으로 0.3~0.4 이하를 컷으로 쓰는 경우가 많음
# 0.7은 거의 "거의 같은 문장 수준"이라 너무 높음
DEFAULT_SIM_THRESHOLD = 0.3


class RAGRetriever:
    """
    RAG 검색 공통 클래스
    
    Knowledge Base에서 관련 정보를 검색하여 AI 프롬프트에 컨텍스트를 추가
    """
    
    def __init__(self, vectorstore_path: Optional[str] = None):
        """
        Args:
            vectorstore_path: Vector Store 경로 (None이면 Config에서 가져옴)
        """
        self.vectorstore_path = vectorstore_path or Config.VECTORSTORE_PATH
        self.vectorstore = None
        self._initialized = False
        
        if HAS_CHROMA:
            self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Vector Store 초기화"""
        try:
            if Path(self.vectorstore_path).exists():
                self.vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_path),
                    embedding_function=OpenAIEmbeddings(
                        model=Config.EMBEDDING_MODEL
                    )
                )
                self._initialized = True
                print(f"✅ Vector Store loaded from {self.vectorstore_path}")
            else:
                # Vector Store가 없으면 생성
                Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
                self.vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_path),
                    embedding_function=OpenAIEmbeddings(
                        model=Config.EMBEDDING_MODEL
                    )
                )
                self._initialized = True
                print(f"✅ Vector Store created at {self.vectorstore_path}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Vector Store: {e}")
            print("   RAG features will work with fallback mode.")
    
    def clear_vectorstore(self) -> bool:
        """
        Vector Store의 모든 문서를 삭제 (컬렉션 클리어)
        
        Returns:
            성공 여부
        """
        if not self._initialized or not self.vectorstore:
            print("⚠️  Vector Store not initialized. Cannot clear.")
            return False
        
        try:
            # ChromaDB 컬렉션 삭제
            self.vectorstore.delete_collection()
            print(f"🗑️  Vector Store cleared: {self.vectorstore_path}")
            
            # 새로운 컬렉션으로 재초기화
            self.vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=OpenAIEmbeddings(
                    model=Config.EMBEDDING_MODEL
                )
            )
            self._initialized = True
            print(f"✅ Vector Store reinitialized at {self.vectorstore_path}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to clear Vector Store: {e}")
            return False
    
    def add_documents(self, documents: List[Document], check_duplicates: bool = True) -> bool:
        """
        Vector Store에 문서를 동적으로 추가
        
        Args:
            documents: 추가할 Document 리스트
            check_duplicates: 중복 체크 여부 (기본값: True)
            
        Returns:
            성공 여부
        """
        if not self._initialized or not self.vectorstore:
            print("⚠️  Vector Store not initialized. Cannot add documents.")
            return False
        
        try:
            if check_duplicates:
                # 중복 체크: source + sheet + has_draft_context 조합으로 고유 키 생성
                # ChromaDB에서 기존 문서 확인
                documents_to_add = []
                skipped_count = 0
                
                for doc in documents:
                    metadata = doc.metadata
                    source = metadata.get("source", "")
                    sheet = metadata.get("sheet", "")
                    has_draft_context = metadata.get("has_draft_context", False)
                    
                    # 고유 ID 생성: source + sheet + has_draft_context
                    # 같은 source+sheet에 대해 초안 정보 포함 버전은 별도 문서로 취급
                    unique_id = f"{source}::{sheet}::{has_draft_context}"
                    metadata["_unique_id"] = unique_id
                    
                    # 기존 문서 확인: ChromaDB의 get 메서드로 메타데이터 필터링
                    try:
                        # ChromaDB에서 같은 source, sheet, has_draft_context를 가진 문서 검색
                        existing_docs = self.vectorstore.get(
                            where={
                                "source": source,
                                "sheet": sheet,
                                "has_draft_context": has_draft_context
                            }
                        )
                        
                        if existing_docs and len(existing_docs.get("ids", [])) > 0:
                            # 이미 존재하는 문서는 스킵
                            skipped_count += 1
                            continue
                    except Exception as e:
                        # 필터 검색 실패 시 일단 추가 (안전한 방식)
                        # ChromaDB 버전에 따라 get 메서드가 다를 수 있음
                        pass
                    
                    documents_to_add.append(doc)
                
                if documents_to_add:
                    self.vectorstore.add_documents(documents_to_add)
                    if skipped_count > 0:
                        print(f"✅ Added {len(documents_to_add)}/{len(documents)} documents to Vector Store ({skipped_count} duplicates skipped)")
                    else:
                        print(f"✅ Added {len(documents_to_add)} documents to Vector Store")
                else:
                    print(f"⚠️  All {len(documents)} documents are duplicates, skipping...")
            else:
                self.vectorstore.add_documents(documents)
                print(f"✅ Added {len(documents)} documents to Vector Store")
            
            return True
        except Exception as e:
            print(f"⚠️  Failed to add documents to Vector Store: {e}")
            return False
    
    def search_ddd_patterns(self, query: str, k: int = 10) -> List[Dict]:
        """
        DDD 패턴 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_ddd_patterns(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "ddd_pattern"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  DDD pattern search failed: {e}")
            return self._fallback_search_ddd_patterns(query, k)
    
    def search_project_templates(self, query: str, k: int = 5) -> List[Dict]:
        """
        유사 프로젝트 사례 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_project_templates(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "project_template"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  Project template search failed: {e}")
            return self._fallback_search_project_templates(query, k)
    
    def search_vocabulary(self, query: str, k: int = 20) -> List[Dict]:
        """
        도메인 용어 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_vocabulary(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "vocabulary"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  Vocabulary search failed: {e}")
            return self._fallback_search_vocabulary(query, k)
    
    def search_ui_patterns(self, query: str, k: int = 10) -> List[Dict]:
        """
        UI 패턴 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_ui_patterns(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "ui_pattern"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  UI pattern search failed: {e}")
            return self._fallback_search_ui_patterns(query, k)
    
    # Fallback methods (Vector Store가 없을 때 JSON 파일에서 직접 검색)
    
    def _fallback_search_ddd_patterns(self, query: str, k: int) -> List[Dict]:
        """DDD 패턴 Fallback 검색 (JSON 파일에서 직접)"""
        try:
            pattern_files = list(Config.DOMAIN_PATTERNS_PATH.glob("*.json"))
            results = []
            
            for file_path in pattern_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "ddd_pattern"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback DDD search failed: {e}")
            return []
    
    def _fallback_search_project_templates(self, query: str, k: int) -> List[Dict]:
        """프로젝트 템플릿 Fallback 검색"""
        try:
            template_files = list(Config.PROJECT_TEMPLATES_PATH.glob("*.json"))
            results = []
            
            for file_path in template_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "project_template"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback project search failed: {e}")
            return []
    
    def _fallback_search_vocabulary(self, query: str, k: int) -> List[Dict]:
        """용어 Fallback 검색"""
        try:
            vocab_files = list(Config.VOCABULARY_PATH.glob("*.json"))
            results = []
            
            for file_path in vocab_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "vocabulary"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback vocabulary search failed: {e}")
            return []
    
    def _fallback_search_ui_patterns(self, query: str, k: int) -> List[Dict]:
        """UI 패턴 Fallback 검색"""
        try:
            ui_files = list(Config.UI_PATTERNS_PATH.glob("*.json"))
            results = []
            
            for file_path in ui_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "ui_pattern"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback UI search failed: {e}")
            return []
    
    def search_company_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        회사 표준 검색 (데이터베이스, API, 용어 등 모든 표준)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_company_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,  # 필터링을 위해 더 많이 가져옴
                    filter={"type": {"$in": ["database_standard", "api_standard", "terminology_standard"]}}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error)
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                all_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 5  # 더 많이 가져와서 필터링
                )
                # 수동 필터링
                results_with_scores = []
                for doc, score in all_results:
                    doc_type = doc.metadata.get("type", "")
                    if doc_type in ["database_standard", "api_standard", "terminology_standard"]:
                        results_with_scores.append((doc, score))
            # 점수 필터링
            # ChromaDB의 similarity_search_with_score는 거리(distance)를 반환
            # ChromaDB는 기본적으로 코사인 거리(cosine distance)를 사용
            # 
            # 거리 범위는 상황에 따라 다를 수 있음:
            # - 정규화된 벡터: 0~1 범위 (distance = 1 - cosine_similarity)
            # - 일반 코사인 거리: 0~2 범위 (distance = 1 - cos(θ), cos(θ) = -1~1)
            # 
            # 실제 거리 값의 범위를 동적으로 감지하여 변환
            filtered_results = []
            all_scores = []  # 디버깅용
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_min, dist_max = min(distances), max(distances)
                # 거리 범위에 따라 변환 방식 결정
                # 대부분의 거리가 1.0을 넘으면 0~2 범위로 가정, 아니면 0~1 범위로 가정
                if dist_max > 1.0:
                    # 0~2 범위: similarity = 1 - (distance / 2)
                    distance_range = 2.0
                else:
                    # 0~1 범위: similarity = 1 - distance
                    distance_range = 1.0
            else:
                # 기본값: 0~2 범위로 가정 (안전한 선택)
                distance_range = 2.0
            
            for doc, score_value in results_with_scores:
                # 원본 값을 확인
                raw_score = float(score_value)
                distance = abs(raw_score)
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    # 0~2 범위: similarity = 1 - (distance / 2)
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    # 0~1 범위: similarity = 1 - distance
                    similarity = max(0.0, 1.0 - distance)
                
                all_scores.append((raw_score, distance, similarity))
                
                # 점수 필터링: 유사도가 임계값 이상인 것만 포함
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": raw_score  # 원본 값도 저장
                    })
                # 상위 k개를 가져오되, 임계값 이상인 것만 포함
                # k개를 채우지 못해도 임계값 이상인 것들은 모두 포함
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 디버깅: 전체 점수 분포 출력 (처음 10개)
            if all_scores:
                print(f"  [DEBUG] 검색 결과 점수 분포 (처음 10개, 총 {len(all_scores)}개):")
                for i, (raw, dist, sim) in enumerate(all_scores[:10]):
                    print(f"    [{i+1}] 원본값: {raw:.6f}, 거리: {dist:.6f}, 유사도: {sim:.6f}")
                print(f"  [DEBUG] 필터링 후 결과: {len(filtered_results)}/{len(all_scores)}")
                if len(all_scores) > 0:
                    raw_min, raw_max = min(s[0] for s in all_scores), max(s[0] for s in all_scores)
                    dist_min, dist_max = min(s[1] for s in all_scores), max(s[1] for s in all_scores)
                    sim_min, sim_max = min(s[2] for s in all_scores), max(s[2] for s in all_scores)
                    print(f"  [DEBUG] 원본값 범위: {raw_min:.6f} ~ {raw_max:.6f}")
                    print(f"  [DEBUG] 거리 범위: {dist_min:.6f} ~ {dist_max:.6f}")
                    print(f"  [DEBUG] 유사도 범위: {sim_min:.6f} ~ {sim_max:.6f}")
                    
                    # 필터링 전후 비교
                    print(f"  [DEBUG] 필터링 전 결과: {len(results_with_scores)}개")
                    print(f"  [DEBUG] 필터링 후 결과: {len(filtered_results)}개 (임계값: {score_threshold:.3f} 이상)")
                    
                    # 만약 유사도가 모두 0이면 경고 및 거리 범위 분석
                    if sim_max == 0.0:
                        print(f"  [WARNING] ⚠️  모든 유사도가 0입니다! 거리 범위를 확인하세요.")
                        print(f"  [WARNING] 거리 범위: {dist_min:.6f} ~ {dist_max:.6f}")
                        # 실제 변환 로직과 일치하도록 distance_range 사용
                        if distance_range == 2.0:
                            sim_est = max(0.0, 1.0 - (dist_max / 2.0))
                        else:
                            sim_est = max(0.0, 1.0 - dist_max)
                        print(f"  [WARNING] 거리 {dist_max:.3f}는 유사도 {sim_est:.3f}로 변환됩니다.")
                        print(f"  [WARNING] 임계값 {score_threshold:.3f}보다 낮아서 필터링되었습니다.")
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # similarity_search_with_score가 지원되지 않는 경우 기본 검색 사용 (점수 필터링 없음)
            print(f"⚠️  similarity_search_with_score 실패, 기본 검색 사용: {e}")
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": {"$in": ["database_standard", "api_standard", "terminology_standard"]}}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None  # 점수 없음 (필터링 안 함)
                    }
                    for doc in results
                ]
            except Exception as e2:
                # 필터가 지원되지 않는 경우 필터 없이 검색
                try:
                    results = self.vectorstore.similarity_search(query, k=k)
                    return [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": None
                        }
                        for doc in results
                        if doc.metadata.get("type") in ["database_standard", "api_standard", "terminology_standard"]
                    ]
                except Exception as e3:
                    print(f"⚠️  Company standards search failed: {e3}")
                    return self._fallback_search_company_standards(query, k)
    
    def search_api_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        API 표준 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_api_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,
                    filter={"type": "api_standard"}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error)
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                all_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 5  # 더 많이 가져와서 필터링
                )
                # 수동 필터링
                results_with_scores = []
                for doc, score in all_results:
                    doc_type = doc.metadata.get("type", "")
                    if doc_type == "api_standard":
                        results_with_scores.append((doc, score))
            # 점수 필터링 (코사인 거리 기반)
            # 거리 범위를 동적으로 감지하여 변환
            filtered_results = []
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_max = max(distances)
                distance_range = 2.0 if dist_max > 1.0 else 1.0
            else:
                distance_range = 2.0  # 기본값
            
            for doc, score_value in results_with_scores:
                raw_score = float(score_value)
                distance = abs(raw_score)
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = max(0.0, 1.0 - distance)
                
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": raw_score  # 원본 값도 저장
                    })
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # fallback: 기본 검색 사용
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": "api_standard"}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    }
                    for doc in results
                ]
            except Exception as e2:
                print(f"⚠️  API standards search failed: {e2}")
                return self._fallback_search_api_standards(query, k)
    
    def search_terminology_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        용어 표준 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_terminology_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,
                    filter={"type": "terminology_standard"}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error)
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                all_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 5  # 더 많이 가져와서 필터링
                )
                # 수동 필터링
                results_with_scores = []
                for doc, score in all_results:
                    doc_type = doc.metadata.get("type", "")
                    if doc_type == "terminology_standard":
                        results_with_scores.append((doc, score))
            # 점수 필터링 (코사인 거리 기반)
            # 거리 범위를 동적으로 감지하여 변환
            filtered_results = []
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_max = max(distances)
                distance_range = 2.0 if dist_max > 1.0 else 1.0
            else:
                distance_range = 2.0  # 기본값
            
            for doc, score_value in results_with_scores:
                distance = abs(float(score_value))
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = max(0.0, 1.0 - distance)
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": float(score_value)  # 원본 값도 저장
                    })
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # fallback: 기본 검색 사용
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": "terminology_standard"}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    }
                    for doc in results
                ]
            except Exception as e2:
                print(f"⚠️  Terminology standards search failed: {e2}")
                return self._fallback_search_terminology_standards(query, k)
    
    def _fallback_search_company_standards(self, query: str, k: int) -> List[Dict]:
        """회사 표준 Fallback 검색"""
        try:
            standards_path = Config.COMPANY_STANDARDS_PATH
            if not standards_path.exists():
                return []
            
            results = []
            # 표준 문서 파일 찾기
            for file_path in standards_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.xls', '.pptx', '.txt', '.md']:
                    # 간단한 텍스트 매칭 (실제로는 Vector Store가 필요)
                    results.append({
                        "content": f"Standard document: {file_path.name}",
                        "metadata": {
                            "source": str(file_path),
                            "type": "database_standard"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback company standards search failed: {e}")
            return []
    
    def _fallback_search_api_standards(self, query: str, k: int) -> List[Dict]:
        """API 표준 Fallback 검색"""
        return self._fallback_search_company_standards(query, k)
    
    def _fallback_search_terminology_standards(self, query: str, k: int) -> List[Dict]:
        """용어 표준 Fallback 검색"""
        return self._fallback_search_company_standards(query, k)

