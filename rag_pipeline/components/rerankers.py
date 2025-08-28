import requests
from typing import List, Sequence

from langchain_core.documents import Document
# 【关键修复】使用诊断脚本在您的环境中找到的、绝对正确的导入路径
from langchain_community.document_transformers.beautiful_soup_transformer import BaseDocumentTransformer
from rag_pipeline.settings import settings

class AiHubMixReranker(BaseDocumentTransformer):
    """
    一个自定义的 LangChain 文档转换器，
    通过调用 AiHubMix 平台的 API 来对文档进行重排。
    """
    top_n: int = 3
    """最终返回的文档数量。"""

    def __init__(self, top_n: int = 3):
        super().__init__()
        self.top_n = top_n
        if not settings.aihubmix_api_key:
            raise ValueError("请在 .env 文件中设置 AIHUBMIX_API_KEY")

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[dict]:
        """调用 AiHubMix API 执行重排。"""
        passages = [doc.page_content for doc in documents]
        api_url = f"{settings.reranker_base_url}"
        
        headers = {
            "Authorization": f"Bearer {settings.aihubmix_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": passages,
            "top_n": len(passages)
        }
        
        print(f"🔄 正在调用 AiHubMix Reranker API: {api_url} with model {settings.reranker_model}")
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            reranked_results = response.json()
            
            if "results" not in reranked_results:
                 raise ValueError(f"API 响应中未找到 'results' 键。收到的响应: {reranked_results}")

            print("✅ AiHubMix Reranker API 调用成功")
            return reranked_results["results"]

        except requests.exceptions.RequestException as e:
            print(f"❌ 调用重排 API 失败: {e}")
            return [{"index": i, "relevance_score": 0.0} for i in range(len(documents))]

    def transform_documents(
        self, documents: Sequence[Document], **kwargs
    ) -> Sequence[Document]:
        query = kwargs.get("query")
        if not query:
            raise ValueError("调用 compress_documents 时必须提供 'query' 参数。")

        if not documents:
            return []
            
        reranked_results = self._rerank_documents(query=query, documents=list(documents))
        
        final_docs = []
        for result in reranked_results[:self.top_n]:
            original_index = result.get("index")
            if original_index is not None and original_index < len(documents):
                doc_copy = Document(
                    page_content=documents[original_index].page_content,
                    metadata=documents[original_index].metadata.copy()
                )
                doc_copy.metadata["relevance_score"] = result.get("relevance_score")
                final_docs.append(doc_copy)
        
        return final_docs
    
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs
    ) -> Sequence[Document]:
        return self.transform_documents(documents, **kwargs)


def get_reranker_model(top_n: int = 3) -> AiHubMixReranker:
    return AiHubMixReranker(top_n=top_n)