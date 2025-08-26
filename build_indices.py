#!/usr/bin/env python3
"""
索引构建脚本

该脚本负责执行耗时的预处理步骤：
1. 从指定路径加载原始文档。
2. 对文档进行切分。
3. 创建 BM25 索引并将其序列化到磁盘。
4. 创建 FAISS 向量存储并将其保存到磁盘。

请在您的原始文档更新后运行此脚本，以重新生成索引。
"""

import sys
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 配置部分 ---

# 确保能导入项目模块 (如果 get_embedding_model 在其他模块中)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 输入路径：您的 Markdown 文件存放位置
SOURCE_DATA_PATH = project_root / "data" / "markdown"

# 输出路径：生成的索引文件将保存在这里
OUTPUT_INDEX_DIR = project_root / "output" / "retriever_indices"

# 文档切分参数 (与您原来的设置保持一致)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

# --- 脚本主逻辑 ---

def _load_documents_from_path(data_path: Path) -> List[Document]:
    """从指定路径加载文档"""
    if not data_path.exists():
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    print(f"📂 从路径加载文档: {data_path}")
    loader = DirectoryLoader(str(data_path), glob="**/*.md", recursive=True, show_progress=True)
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 个文档")
    return documents

def _split_documents(documents: List[Document]) -> List[Document]:
    """切分文档"""
    print(f"🔄 开始文档切分 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )
    split_docs = text_splitter.split_documents(documents)
    
    lengths = [len(doc.page_content) for doc in split_docs]
    if lengths:
        print(f"📊 切分统计: {len(split_docs)}个片段，最短{min(lengths)}，最长{max(lengths)}，平均{sum(lengths)//len(lengths)}字符")
    return split_docs

def build_and_save_indices():
    """执行索引构建和保存的完整流程"""
    print("--- 开始构建检索器索引 ---")
    
    # 0. 准备输出目录
    OUTPUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载和切分文档
    documents = _load_documents_from_path(SOURCE_DATA_PATH)
    if not documents:
        print("❌ 未找到任何文档，脚本终止。")
        return
        
    split_docs = _split_documents(documents)
    if not split_docs:
        print("❌ 文档切分后未产生任何片段，脚本终止。")
        return

    # 2. 构建并保存 BM25 索引
    print("\n🔄 正在构建 BM25 索引...")
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_path = OUTPUT_INDEX_DIR / "bm25_retriever.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"✅ BM25 索引已成功保存到: {bm25_path}")

    # 3. 构建并保存 FAISS 向量存储
    print("\n🔄 正在构建 FAISS 向量存储...")
    try:
        from rag_pipeline.components.llms import get_embedding_model
        embedding_model = get_embedding_model()
        print(f"   - 使用嵌入模型: {type(embedding_model).__name__}")
    except ImportError as e:
        print(f"❌ 无法导入嵌入模型: {e}")
        print("   请确保 'rag_pipeline.components.llms' 模块及其依赖项已正确安装。")
        return

    print("   - 正在为文档计算嵌入向量 (这可能需要一些时间)...")
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    faiss_path = OUTPUT_INDEX_DIR / "faiss_index"
    vector_store.save_local(str(faiss_path))
    print(f"✅ FAISS 向量存储已成功保存到: {faiss_path}")
    
    print("\n🎉 --- 所有索引构建并保存完毕！ ---")


if __name__ == "__main__":
    build_and_save_indices()