"""
표준 문서 인덱서
표준 문서를 Vector Store에 인덱싱
"""
from typing import List, Optional
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # 최신 langchain에서는 별도 패키지에서 import
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        # 구버전 호환성
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠️  chromadb not installed. Indexing will be disabled.")

from src.project_generator.config import Config
from src.project_generator.workflows.common.standard_loader import StandardLoader


class StandardIndexer:
    """
    표준 문서 인덱서
    표준 문서를 로드하고 Vector Store에 인덱싱
    """
    
    def __init__(self, vectorstore_path: Optional[str] = None):
        """
        Args:
            vectorstore_path: Vector Store 경로 (None이면 Config에서 가져옴)
        """
        self.vectorstore_path = vectorstore_path or Config.VECTORSTORE_PATH
        # LLM 기반 semantic_text 생성 활성화
        self.loader = StandardLoader(enable_llm=True)
        self.vectorstore = None
    
    def index_standards(self, standards_path: Optional[Path] = None, 
                       force_reindex: bool = False) -> bool:
        """
        표준 문서를 Vector Store에 인덱싱
        
        Args:
            standards_path: 표준 문서 경로 (None이면 Config에서 가져옴)
            force_reindex: 기존 인덱스 삭제 후 재인덱싱
            
        Returns:
            인덱싱 성공 여부
        """
        if not HAS_CHROMA:
            print("❌ ChromaDB is not installed. Cannot index standards.")
            return False
        
        try:
            # 표준 문서 로드
            print("📚 Loading standard documents...")
            documents = self.loader.load_standards(standards_path)
            
            if not documents:
                print("⚠️  No standard documents found.")
                return False
            
            print(f"📊 Total documents to index: {len(documents)}")
            
            # 표준 타입 결정 및 메타데이터 업데이트
            for doc in documents:
                source_path = Path(doc.metadata.get('source', ''))
                if 'type' not in doc.metadata or doc.metadata['type'] == 'database_standard':
                    # 파일명 기반으로 타입 결정
                    standard_type = self.loader.determine_standard_type(source_path)
                    doc.metadata['type'] = standard_type
            
            # Vector Store 초기화
            vectorstore_path = Path(self.vectorstore_path)
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            if force_reindex and vectorstore_path.exists():
                print("🗑️  Clearing existing Vector Store...")
                try:
                    existing_store = Chroma(
                        persist_directory=str(vectorstore_path),
                        embedding_function=OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
                    )
                    existing_store.delete_collection()
                except Exception as e:
                    print(f"⚠️  Failed to clear existing store: {e}")
            
            # Vector Store 생성
            print("🔧 Initializing Vector Store...")
            self.vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
            )
            
            # 문서 인덱싱
            print(f"📝 Indexing {len(documents)} documents...")
            print("   This may take a few minutes (generating embeddings)...")
            
            self.vectorstore.add_documents(documents)
            
            # 인덱싱 완료 확인
            collection = self.vectorstore._collection
            final_count = collection.count()
            
            print(f"\n✅ Indexing completed!")
            print(f"   Total documents indexed: {final_count}")
            print(f"   Vector Store location: {self.vectorstore_path}")
            
            return True
        
        except Exception as e:
            print(f"\n❌ Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_indexed_count(self) -> int:
        """인덱싱된 문서 수 반환"""
        if not HAS_CHROMA:
            return 0
        
        try:
            if not Path(self.vectorstore_path).exists():
                return 0
            
            vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
            )
            collection = vectorstore._collection
            return collection.count()
        except Exception as e:
            print(f"⚠️  Failed to get indexed count: {e}")
            return 0

