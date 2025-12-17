#!/usr/bin/env python3
"""
표준 문서 인덱싱 스크립트
PPT, 엑셀 파일을 Vector Store에 인덱싱
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

from src.project_generator.workflows.common.standard_indexer import StandardIndexer
from src.project_generator.config import Config


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Index standard documents (PPT, Excel) to Vector Store')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reindexing (clear existing index)'
    )
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Path to standards directory (default: Config.COMPANY_STANDARDS_PATH)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Standard Documents indexing...")
    print(f"📁 Standards path: {args.path or Config.COMPANY_STANDARDS_PATH}")
    print(f"🤖 Embedding model: {Config.EMBEDDING_MODEL}")
    
    if args.force:
        print("⚠️  Force reindexing enabled - existing index will be cleared")
    
    # 인덱서 생성
    indexer = StandardIndexer()
    
    # 표준 문서 인덱싱
    standards_path = Path(args.path) if args.path else None
    success = indexer.index_standards(standards_path=standards_path, force_reindex=args.force)
    
    if success:
        count = indexer.get_indexed_count()
        print(f"\n✅ Successfully indexed {count} documents")
        return 0
    else:
        print("\n❌ Indexing failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

