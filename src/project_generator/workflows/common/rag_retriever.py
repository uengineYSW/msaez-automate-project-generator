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
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import Document
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠️  chromadb not installed. RAG features will be disabled.")

from src.project_generator.config import Config


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
                print(f"⚠️  Vector Store not found at {self.vectorstore_path}")
                print("   RAG features will work with fallback mode.")
        except Exception as e:
            print(f"⚠️  Failed to initialize Vector Store: {e}")
            print("   RAG features will work with fallback mode.")
    
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

