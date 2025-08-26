# ingest.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from rag_pipeline.components.llms import get_embedding_model  # 假设你有一个封装好的获取嵌入模型的函数
from loguru import logger
import os
import glob

DATA_PATH = 'data/raw/*.md'
VECTOR_STORE_PATH = 'data/processed/vector_store.faiss'

def ingest_markdown_files():
    logger.info("Starting to process Markdown files from the directory: {}", DATA_PATH)
    
    # 1. Load Markdown files
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", recursive=True) # 建议加上 recursive=True
    documents = loader.load()
    
    if not documents:
        logger.warning("No documents found in {}. Exiting.", DATA_PATH)
        return
        
    logger.info("Loaded {} documents", len(documents))
    
    # 2. Split documents correctly while preserving metadata
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5")
    ]
    # 【修正】直接调用 split_documents，它会正确处理并保留每个块的元数据
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_documents = splitter.split_documents(documents)
    
    logger.info("Split documents into {} segments", len(split_documents))
    
    # 3. Create FAISS vector store with your embedding model
    # 假设 get_embedding_model() 是您封装好的函数
    embeddings = get_embedding_model() 
    vector_store = FAISS.from_documents(split_documents, embeddings)
    
    # 4. Save the FAISS index correctly
    # 【修正】使用 save_local 方法
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    logger.info("FAISS vector store saved at: {}", VECTOR_STORE_PATH)

if __name__ == "__main__":
    ingest_markdown_files()