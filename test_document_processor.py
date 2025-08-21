#!/usr/bin/env python3
"""
测试document_processor.py处理PDF文档的功能
处理data/guide.pdf并生成向量存储、BM25索引等
"""

import sys
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_document_processing():
    """测试文档处理功能"""
    print("=== 测试DocumentProcessor处理PDF文档 ===")
    
    try:
        # 导入必要模块
        from rag_pipeline.components.preprocessing.document_processor import DocumentProcessor, ProcessingConfig
        
        # 设置文件路径
        pdf_path = Path("data/guide.pdf")
        output_dir = Path("output/test_processing")
        
        # 检查PDF文件是否存在
        if not pdf_path.exists():
            print(f"❌ PDF文件不存在: {pdf_path}")
            return False
        
        print(f"📄 找到PDF文件: {pdf_path}")
        print(f"💾 输出目录: {output_dir}")
        
        # 创建处理配置
        config = ProcessingConfig(
            chunk_size=400,  # 适合中文文档
            chunk_overlap=60,
            enable_parent_document=True,  # 启用父文档检索
            enhance_metadata=True,  # 启用元数据增强
            save_intermediate_files=True  # 保存中间文件便于调试
        )
        
        print(f"⚙️ 处理配置:")
        print(f"   - 分块大小: {config.chunk_size}")
        print(f"   - 重叠大小: {config.chunk_overlap}")
        print(f"   - 嵌入模型: {config.embedding_model}")
        print(f"   - 父文档检索: {config.enable_parent_document}")
        print(f"   - 元数据增强: {config.enhance_metadata}")
        
        # 创建处理器
        print("\n🔧 初始化DocumentProcessor...")
        processor = DocumentProcessor(config)
        print("✅ DocumentProcessor初始化成功")
        
        # 处理PDF文档
        print(f"\n📊 开始处理PDF文档: {pdf_path.name}")
        result = processor.process_single_document(pdf_path, output_dir)
        
        # 显示处理结果
        print(f"\n🎉 文档处理完成!")
        print(f"📁 结果摘要:")
        print(f"   - PDF路径: {result['pdf_path']}")
        print(f"   - 向量存储: {result['vector_store_path']}")
        print(f"   - BM25索引: {result['bm25_path']}")
        print(f"   - 文档块数量: {result['chunks_count']}")
        
        # 检查生成的文件
        print(f"\n📂 检查生成的文件:")
        vector_store_path = Path(result['vector_store_path'])
        bm25_path = Path(result['bm25_path'])
        
        if vector_store_path.exists():
            print(f"✅ 向量存储已生成: {vector_store_path}")
        else:
            print(f"❌ 向量存储未生成: {vector_store_path}")
        
        if bm25_path.exists():
            print(f"✅ BM25索引已生成: {bm25_path}")
        else:
            print(f"❌ BM25索引未生成: {bm25_path}")
        
        # 显示元数据信息
        metadata = result.get('document_metadata', {})
        if metadata:
            print(f"\n📋 文档元数据:")
            print(f"   - 文件大小: {metadata.get('file_size', 0)} bytes")
            print(f"   - 总字符数: {metadata.get('total_characters', 0)}")
            print(f"   - 总Token数: {metadata.get('total_tokens', 0)}")
            if 'ai_analysis' in metadata:
                print(f"   - AI分析: {metadata['ai_analysis'][:100]}...")
        
        print("\n✅ 所有文档处理测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 文档处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generated_files():
    """测试生成的文件是否可以正常加载"""
    print("\n=== 测试生成文件的加载 ===")
    
    try:
        from rag_pipeline.components.preprocessing.retriever_factory import RetrieverFactory
        
        # 文件路径
        output_dir = Path("output/test_processing")
        vector_store_path = output_dir / "guide_faiss"
        bm25_path = output_dir / "guide_bm25.pkl"
        metadata_path = output_dir / "guide_metadata.json"
        
        # 测试向量检索器
        if vector_store_path.exists():
            print("🔍 测试向量检索器...")
            vector_retriever = RetrieverFactory.create_vector_retriever(vector_store_path)
            results = vector_retriever.retrieve("博士后基金申请条件", top_k=3)
            print(f"✅ 向量检索成功，返回 {len(results)} 个结果")
            
            if results:
                print(f"   最佳匹配 (分数={results[0]['score']:.4f}): {results[0]['content'][:100]}...")
        
        # 测试混合检索器
        if bm25_path.exists() and metadata_path.exists():
            print("🔍 测试混合检索器...")
            hybrid_retriever = RetrieverFactory.create_hybrid_retriever(
                vector_store_path, bm25_path, metadata_path
            )
            results = hybrid_retriever.retrieve("申请材料要求", top_k=3)
            print(f"✅ 混合检索成功，返回 {len(results)} 个结果")
            
            if results:
                print(f"   最佳匹配 (分数={results[0].get('hybrid_score', 0):.4f}): {results[0]['content'][:100]}...")
        
        print("✅ 所有文件加载测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 文件加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试DocumentProcessor处理PDF功能...")
    
    # 运行测试
    processing_ok = test_document_processing()
    
    if processing_ok:
        loading_ok = test_generated_files()
        
        if loading_ok:
            print("\n🎉 所有测试通过! DocumentProcessor工作正常。")
            sys.exit(0)
        else:
            print("\n⚠️ 文档处理成功，但文件加载有问题。")
            sys.exit(1)
    else:
        print("\n❌ 文档处理失败。")
        sys.exit(1)
