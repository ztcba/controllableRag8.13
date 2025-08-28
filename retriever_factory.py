#!/usr/bin/env python3
"""
检索器工厂

提供统一的检索器创建接口。该工厂函数从磁盘加载预先构建好的索引，
以实现快速初始化。使用全局缓存来确保在单个程序运行期间只加载一次。
"""

import sys
import pickle
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

# --- 配置部分 ---

# 确保能导入项目模块
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 预处理索引文件的存放路径 (必须与 build_indices.py 中的 OUTPUT_INDEX_DIR 一致)
INDEX_DIR = project_root / "output" / "retriever_indices"
BM25_PATH = INDEX_DIR / "bm25_retriever.pkl"
FAISS_PATH = INDEX_DIR / "faiss_index"

# --- 全局缓存 ---
_global_retriever: Optional[BaseRetriever] = None

# --- 加载函数 ---

def _load_bm25_retriever(k: int) -> BM25Retriever:
    """从磁盘加载BM25检索器并配置k值"""
    print(f"🔄 从磁盘加载 BM25 索引...")
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = k
    print("✅ BM25 检索器加载成功")
    return bm25_retriever

def _load_vector_retriever(k: int) -> BaseRetriever:
    """从磁盘加载FAISS向量存储并创建检索器"""
    print(f"🔄 从磁盘加载 FAISS 向量存储...")
    try:
        from rag_pipeline.components.llms import get_embedding_model
        embedding_model = get_embedding_model()
    except ImportError as e:
        raise RuntimeError(f"无法导入嵌入模型: {e}")
    
    vector_store = FAISS.load_local(
        folder_path=str(FAISS_PATH), 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True # FAISS需要允许pickle反序列化
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={'k': k})
    print("✅ FAISS 检索器加载成功")
    return vector_retriever

# --- 主工厂函数 ---

def create_hybrid_retriever(
    bm25_k: int = 5,
    vector_k: int = 5,
    hybrid_weights: List[float] = [0.4, 0.6],
    force_reload: bool = False
) -> BaseRetriever:
    """
    创建混合检索器的优化版本 - 优先使用缓存，否则从磁盘加载预处理索引。
    
    Args:
        bm25_k: BM25检索返回的文档数量。
        vector_k: 向量检索返回的文档数量。
        hybrid_weights: 混合检索权重 [BM25权重, 向量权重]。
        force_reload: 如果为True，则强制从磁盘重新加载，忽略缓存。
    
    Returns:
        配置好的混合检索器实例。
        
    Raises:
        FileNotFoundError: 如果索引文件不存在。
    """
    global _global_retriever
    
    if _global_retriever is not None and not force_reload:
        print("✅ 使用缓存的检索器实例")
        return _global_retriever

    print("🔄 首次创建或强制重载，从磁盘加载预处理索引...")
    
    # 检查索引文件是否存在，提供清晰的错误提示
    if not BM25_PATH.exists() or not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"索引文件未找到！请先运行 'build_indices.py' 脚本来构建索引。\n"
            f"  - 检查路径: {BM25_PATH}\n"
            f"  - 检查路径: {FAISS_PATH}"
        )
    
    # 1. 从磁盘加载各个检索器
    bm25_retriever = _load_bm25_retriever(bm25_k)
    vector_retriever = _load_vector_retriever(vector_k)
    
    # 2. 创建混合检索器
    print(f"🔄 组装混合检索器 (weights={hybrid_weights})...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=hybrid_weights
    )
    
    # 3. 缓存并返回
    _global_retriever = ensemble_retriever
    print("✅ 混合检索器创建成功并已缓存！")
    
    return _global_retriever

# --- 测试代码 ---

def test_retriever(retriever: BaseRetriever, test_queries: Optional[List[str]] = None):
    """测试检索器功能"""
    if test_queries is None:
        test_queries = ["博士后科学基金申请条件", "资助金额标准"]
    
    print(f"\n🧪 开始测试检索器功能...")
    try:
        for query in test_queries:
            print(f"\n🔍 查询: '{query}'")
            results = retriever.invoke(query)
            print(f"📋 返回 {len(results)} 个结果")
            if results:
                preview = results[0].page_content[:100].replace('\n', ' ') + "..."
                print(f"   - 最佳匹配预览: {preview}")
        
        print("\n✅ 检索器测试通过")
    except Exception as e:
        print(f"\n❌ 检索器测试失败: {e}")

if __name__ == "__main__":
    print("--- 测试优化后的检索器工厂 ---")
    try:
        # 第一次调用，应该从磁盘加载
        retriever_instance_1 = create_hybrid_retriever()
        print(f"Retriever 1 ID: {id(retriever_instance_1)}")
        test_retriever(retriever_instance_1)
        
        print("\n--- 第二次调用 ---")
        # 第二次调用，应该直接使用缓存
        retriever_instance_2 = create_hybrid_retriever()
        print(f"Retriever 2 ID: {id(retriever_instance_2)}")

        # 验证是否为同一个实例
        if id(retriever_instance_1) == id(retriever_instance_2):
            print("\n✅ 验证成功：两次调用返回的是同一个缓存实例。")
        else:
            print("\n❌ 验证失败：缓存未生效！")

        print("\n🎉 检索器工厂测试成功！")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 检索器工厂测试失败: {e}")
        traceback.print_exc()