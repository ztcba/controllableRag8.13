#!/usr/bin/env python3
"""
测试preprocessing组件与项目配置系统的集成
"""

import sys
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_integration():
    """测试配置系统集成"""
    print("=== 测试preprocessing组件配置集成 ===")
    
    try:
        # 测试settings导入
        from rag_pipeline.settings import settings
        print(f"✅ Settings导入成功")
        print(f"   - OpenAI API Key: {'已设置' if settings.openai_api_key else '未设置'}")
        print(f"   - Embedding Model: {settings.embedding_model}")
        print(f"   - Vector Store Path: {settings.vector_store_path}")
        
        # 测试ProcessingConfig
        from rag_pipeline.components.preprocessing.document_processor import ProcessingConfig
        config = ProcessingConfig()
        print(f"✅ ProcessingConfig创建成功")
        print(f"   - Embedding Model: {config.embedding_model}")
        print(f"   - Chunk Size: {config.chunk_size}")
        
        # 测试RetrievalConfig
        from rag_pipeline.components.preprocessing.retriever_factory import RetrievalConfig
        retrieval_config = RetrievalConfig()
        print(f"✅ RetrievalConfig创建成功")
        print(f"   - Embedding Model: {retrieval_config.embedding_model}")
        print(f"   - Top K: {retrieval_config.top_k}")
        
        # 测试DocumentProcessor初始化
        from rag_pipeline.components.preprocessing.document_processor import DocumentProcessor
        processor = DocumentProcessor(config)
        print(f"✅ DocumentProcessor初始化成功")
        print(f"   - Embeddings模型: {processor.embeddings.model}")
        
        print("\n🎉 所有配置集成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_compatibility():
    """测试导入兼容性"""
    print("\n=== 测试导入兼容性 ===")
    
    try:
        # 测试rank_bm25
        from rank_bm25 import BM25Okapi
        print("✅ rank_bm25导入成功")
        
        # 测试langchain组件
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        print("✅ langchain组件导入成功")
        
        # 测试其他依赖
        import tiktoken
        import numpy as np
        import faiss
        print("✅ 其他依赖导入成功")
        
        print("🎉 所有导入兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试preprocessing组件集成...")
    
    # 运行测试
    config_ok = test_config_integration()
    import_ok = test_import_compatibility()
    
    if config_ok and import_ok:
        print("\n✅ 所有测试通过！preprocessing组件已成功集成到项目中。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查配置和依赖。")
        sys.exit(1)
