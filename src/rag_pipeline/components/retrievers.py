# 创建用于加载 FAISS 索引并返回检索器的函数
# 将数据加载和检索器创建的逻辑分离，使核心应用代码与数据存储细节解耦。
# 使用 settings 来获取路径，并保留了创建全局可导入实例的做法，以提高效率
# src/rag_pipeline/components/retrievers.py
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from src.rag_pipeline.settings import settings

def create_retrievers():
    """
    Loads FAISS vector stores and creates retrievers.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    
    chunks_vector_store = FAISS.load_local(
        settings.chunks_vector_store_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    chapter_summaries_vector_store = FAISS.load_local(
        settings.chapter_summaries_vector_store_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    book_quotes_vectorstore = FAISS.load_local(
        settings.book_quotes_vectorstore_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})     
    chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})
    book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})
    
    return chunks_query_retriever, chapter_summaries_query_retriever, book_quotes_query_retriever

# Create singleton instances of the retrievers to be imported elsewhere
# This avoids reloading the FAISS indexes every time.
(
    chunks_query_retriever, 
    chapter_summaries_query_retriever, 
    book_quotes_query_retriever
) = create_retrievers()