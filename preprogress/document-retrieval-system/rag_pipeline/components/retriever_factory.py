
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import CrossEncoderReranker # LangChain v0.2+
# from langchain.retrievers.document_compressors import CrossEncoderReranker # LangChain v0.1
from loguru import logger
import os
from typing import List
from langchain_core.documents import Document

def create_retriever(
    faiss_index_path: str = "faiss_index",
    embedding_model = None,
    docs_path: str = "source_documents", 
    use_reranker: bool = True,
    k: int = 10
):
    """
    创建结合了BM25和FAISS向量搜索的混合检索器，并可选地添加重排功能。

    Parameters:
    - faiss_index_path: FAISS索引的本地路径.
    - embedding_model: 已实例化的嵌入模型.
    - docs_path: 原始Markdown文档所在的目录路径.
    - use_reranker: 是否启用重排模型.
    - k: 在重排前，每个基础检索器要检索的文档数量.

    Returns:
    一个配置好的检索器对象.
    """
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

    logger.info("Loading FAISS vector store from: {}", faiss_index_path)
    # 1. 加载FAISS向量检索器
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
    faiss_retriever = vector_store.as_retriever(search_kwargs={'k': k})
    
    logger.info("Initializing BM25 retriever from source documents: {}", docs_path)
    # 2. 【修正】从原始文档创建BM25检索器
    # 注意：为了性能，这一步可以在ingest时预处理并缓存
    loader = DirectoryLoader(docs_path, glob="**/*.md", recursive=True)
    raw_documents = loader.load()
    
    # 确保切分方式与ingest时一致，以保证BM25处理的文本块粒度相同
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_documents = splitter.split_documents(raw_documents)

    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = k

    # 3. 【修正】创建混合检索器 (EnsembleRetriever)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]  # 可根据需求调整权重
    )

    if not use_reranker:
        logger.info("Hybrid retriever created without reranker.")
        return ensemble_retriever

    # 4. 【补充】如果启用，则添加重排模型
    logger.info("Creating reranker with BAAI/bge-reranker-base model.")
    # 请注意，根据你的langchain版本，导入路径可能不同
    reranker_model = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_model, 
        base_retriever=ensemble_retriever
    )
    logger.info("Hybrid retriever with reranker created successfully.")
    
    return compression_retriever