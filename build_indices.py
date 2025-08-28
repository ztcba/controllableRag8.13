#!/usr/bin/env python3
"""
索引构建脚本 (已优化：父文档检索策略)

该脚本负责执行耗时的预处理步骤：
1. 从指定路径加载原始文档。
2. (优化) 将文档按Markdown标题切分为“父块”。
3. (优化) 将每个“父块”切分为更小的“子块”，并将父块内容存入子块元数据。
4. 基于“子块”创建 BM25 索引并将其序列化到磁盘。
5. 基于“子块”创建 FAISS 向量存储并将其保存到磁盘。

请在您的原始文档更新后运行此脚本，以重新生成索引。
"""

import sys
import pickle
from pathlib import Path
from typing import List, Optional
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader # 【关键改动】导入TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# --- 配置部分 ---

# 确保能导入项目模块
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 输入路径：您的 Markdown 文件存放位置
SOURCE_DATA_PATH = project_root / "data" / "markdown"

# 输出路径：生成的索引文件将保存在这里
OUTPUT_INDEX_DIR = project_root / "output" / "retriever_indices"

# --- 优化开始: 定义父子分块参数 ---

# 父块切分配置 (这个配置本身是正确的)
HEADERS_TO_SPLIT_ON = [
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# 子块切分参数 (用于在父块内部切分)
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 40
CHILD_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

# --- 优化结束 ---


# --- 【关键改动】修改加载函数 ---
def _load_documents_from_path(data_path: Path) -> List[Document]:
    """
    从指定路径加载文档。
    【重要】使用 TextLoader 按原文加载 Markdown 文件，以保留 "##" 等标题标记用于切分。
    """
    if not data_path.exists():
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    print(f"📂 从路径加载文档: {data_path}")
    # 使用 loader_cls=TextLoader 和 utf-8 编码来确保 Markdown 标题被作为纯文本读取
    loader = DirectoryLoader(
        str(data_path), 
        glob="**/*.md", 
        recursive=True, 
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 个文档 (作为纯文本)")
    return documents


def _split_documents_parent_child(documents: List[Document]) -> List[Document]:
    """
    使用“父文档”策略切分文档。
    1. 使用 MarkdownHeaderTextSplitter 切分出父块。
    2. 使用 RecursiveCharacterTextSplitter 将父块切分为子块。
    3. 在子块的元数据中存储父块的内容。
    """
    print("🔄 开始父子文档切分策略...")
    
    # 1. 定义切分器
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=CHILD_SEPARATORS
    )

    all_child_chunks = []
    parent_doc_count = 0

    for doc in documents:
        # 由于现在加载的是纯文本，之前的清洗步骤依然有益，可以保留以增加稳健性
        content = doc.page_content.replace('\u3000', ' ').replace('\u00A0', ' ')
        cleaned_content = "\n".join(line.strip() for line in content.splitlines())
        
        # 2. 对每个原始文档切分出父块
        parent_chunks = parent_splitter.split_text(cleaned_content)
        parent_doc_count += len(parent_chunks)
        
        for parent_chunk in parent_chunks:
            # 3. 将父块内容切分为子块
            child_chunks = child_splitter.split_documents([parent_chunk])
            
            for child_chunk in child_chunks:
                # 4. 关键步骤：在每个子块的元数据中添加父块内容
                child_chunk.metadata["parent_content"] = parent_chunk.page_content
                child_chunk.metadata.update(doc.metadata)
            all_child_chunks.extend(child_chunks)

    lengths = [len(doc.page_content) for doc in all_child_chunks]
    if lengths:
        print(f"📊 切分统计: 共处理 {parent_doc_count} 个父块，生成 {len(all_child_chunks)} 个子块。")
        print(f"   子块长度: 最短{min(lengths)}，最长{max(lengths)}，平均{sum(lengths)//len(lengths)}字符")
        
    return all_child_chunks


def build_and_save_indices():
    """执行索引构建和保存的完整流程"""
    print("--- 开始构建检索器索引 ---")
    
    OUTPUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    documents = _load_documents_from_path(SOURCE_DATA_PATH)
    if not documents:
        print("❌ 未找到任何文档，脚本终止。")
        return
    
    split_docs = _split_documents_parent_child(documents)
    
    if not split_docs:
        print("❌ 文档切分后未产生任何片段，脚本终止。")
        return

    print("\n🔄 正在构建 BM25 索引...")
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_path = OUTPUT_INDEX_DIR / "bm25_retriever.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"✅ BM25 索引已成功保存到: {bm25_path}")

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