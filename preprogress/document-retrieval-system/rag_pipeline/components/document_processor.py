from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from loguru import logger
import os

DATA_PATH = "data/raw"
VECTOR_STORE_PATH = "data/processed/vector_store.faiss"

def process_markdown_files():
    logger.info("Starting to process Markdown files from the directory: {}", DATA_PATH)
    
    # Load Markdown files
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    
    logger.info("Loaded {} documents", len(documents))
    
    # Split documents while preserving structure
    splitter = MarkdownHeaderTextSplitter()
    split_documents = []
    
    for doc in documents:
        split_docs = splitter.split(doc)
        split_documents.extend(split_docs)
    
    logger.info("Split documents into {} segments", len(split_documents))
    
    # Create FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_documents, embeddings)
    
    # Save the FAISS index
    vector_store.save(VECTOR_STORE_PATH)
    logger.info("FAISS vector store saved at: {}", VECTOR_STORE_PATH)

if __name__ == "__main__":
    process_markdown_files()