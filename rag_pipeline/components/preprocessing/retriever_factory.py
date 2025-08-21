"""
检索器工厂 - 基于RAG-Challenge-2冠军方案设计的检索系统
提供多种检索策略：向量检索、BM25检索、混合检索、父文档检索、LLM重排
适用于博士后基金申请手册等复杂文档的精确检索
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Langchain imports - 使用新版本的导入路径
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
import tiktoken

# Import project settings
from ...settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """检索配置类 - 参考冠军方案的检索参数"""
    # 基础检索配置
    top_k: int = 10  # 返回的文档数量
    score_threshold: float = 0.0  # 分数阈值
    
    # 混合检索配置
    vector_weight: float = 0.7  # 向量检索权重
    bm25_weight: float = 0.3   # BM25检索权重
    
    # 父文档检索配置
    enable_parent_retrieval: bool = True  # 是否启用父文档检索
    parent_k: int = 20  # 检索父文档数量
    
    # LLM重排配置
    enable_llm_reranking: bool = False  # 是否启用LLM重排
    rerank_top_k: int = 20  # 重排前的候选数量
    rerank_batch_size: int = 4  # 重排批处理大小
    llm_rerank_weight: float = 0.7  # LLM重排权重
    
    # 模型配置 - 优先使用项目配置
    embedding_model: str = None  # 将在__post_init__中设置
    rerank_model: str = "gpt-4o-mini"
    
    def __post_init__(self):
        """设置默认模型"""
        if self.embedding_model is None:
            self.embedding_model = settings.embedding_model

class BaseRetriever(ABC):
    """基础检索器抽象类"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._setup_environment()
    
    def _setup_environment(self):
        """设置环境"""
        # 使用统一的项目配置
        if not settings.openai_api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """检索接口"""
        pass

class VectorRetriever(BaseRetriever):
    """
    向量检索器 - 参考冠军方案的VectorRetriever
    """
    
    def __init__(self, vector_store_path: Path, config: RetrievalConfig = None):
        super().__init__(config or RetrievalConfig())
        self.vector_store_path = vector_store_path
        self._load_vector_store()
    
    def _load_vector_store(self):
        """加载向量存储"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.llm_base_url  # 使用配置的API基础URL
            )
            
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"向量存储加载成功: {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"向量存储加载失败: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        向量检索
        """
        k = top_k or self.config.top_k
        
        try:
            # 执行相似性搜索
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            
            results = []
            for doc, score in docs_with_scores:
                if score >= self.config.score_threshold:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score),
                        'retrieval_type': 'vector'
                    }
                    results.append(result)
            
            logger.info(f"向量检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

class BM25Retriever(BaseRetriever):
    """
    BM25检索器 - 参考冠军方案的BM25Retriever
    """
    
    def __init__(self, bm25_path: Path, chunks_metadata_path: Path, config: RetrievalConfig = None):
        super().__init__(config or RetrievalConfig())
        self.bm25_path = bm25_path
        self.chunks_metadata_path = chunks_metadata_path
        self._load_bm25_index()
    
    def _load_bm25_index(self):
        """加载BM25索引和元数据"""
        try:
            # 加载BM25索引
            with open(self.bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # 加载分块元数据
            with open(self.chunks_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.chunks_info = metadata['chunks_info']
            
            logger.info(f"BM25索引加载成功: {self.bm25_path}")
            
        except Exception as e:
            logger.error(f"BM25索引加载失败: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        BM25检索
        """
        k = top_k or self.config.top_k
        
        try:
            # 分词查询
            tokenized_query = query.split()
            
            # 获取BM25分数
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # 获取top-k结果
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= self.config.score_threshold:
                    chunk_info = self.chunks_info[idx]
                    result = {
                        'content': chunk_info.get('content_preview', ''),  # 需要完整内容的话需要重新加载
                        'metadata': {
                            'chunk_id': chunk_info.get('chunk_id'),
                            'chunk_size': chunk_info.get('chunk_size'),
                            'chunk_tokens': chunk_info.get('chunk_tokens')
                        },
                        'score': score,
                        'retrieval_type': 'bm25'
                    }
                    results.append(result)
            
            logger.info(f"BM25检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []

class LLMReranker:
    """
    LLM重排器 - 参考冠军方案的LLMReranker
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url  # 使用配置的API基础URL
        )
    
    def rerank_single_document(self, query: str, document: str) -> Dict[str, Any]:
        """
        对单个文档进行重排评分
        """
        system_prompt = """
        你是一个检索相关性评估专家。
        请评估给定文档与查询的相关性，并给出0到1之间的分数（1表示最相关）。
        
        评估标准：
        1. 内容相关性：文档是否直接回答或涉及查询问题
        2. 信息完整性：文档提供的信息是否充分
        3. 准确性：文档信息是否准确可靠
        
        请返回JSON格式：{"score": 分数, "reasoning": "评估理由"}
        """
        
        user_prompt = f"""
        查询：{query}
        
        文档内容：
        {document}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            try:
                result = json.loads(response_text)
                return {
                    'relevance_score': result.get('score', 0.0),
                    'reasoning': result.get('reasoning', '')
                }
            except json.JSONDecodeError:
                return {'relevance_score': 0.5, 'reasoning': 'JSON解析失败'}
                
        except Exception as e:
            logger.warning(f"LLM重排失败: {e}")
            return {'relevance_score': 0.5, 'reasoning': 'LLM调用失败'}
    
    def rerank_documents(self, query: str, documents: List[Dict], batch_size: int = 4) -> List[Dict]:
        """
        批量重排文档 - 参考冠军方案的并行处理
        """
        reranked_docs = []
        
        # 批处理文档
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 并行处理批次
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(self.rerank_single_document, query, doc['content'])
                    for doc in batch
                ]
                
                for doc, future in zip(batch, futures):
                    try:
                        rerank_result = future.result()
                        doc['llm_score'] = rerank_result['relevance_score']
                        doc['llm_reasoning'] = rerank_result['reasoning']
                        reranked_docs.append(doc)
                    except Exception as e:
                        logger.warning(f"文档重排失败: {e}")
                        doc['llm_score'] = 0.5
                        doc['llm_reasoning'] = '重排失败'
                        reranked_docs.append(doc)
        
        return reranked_docs

class HybridRetriever(BaseRetriever):
    """
    混合检索器 - 参考冠军方案的HybridRetriever
    结合向量检索、BM25检索和LLM重排
    """
    
    def __init__(self, vector_store_path: Path, bm25_path: Path, 
                 chunks_metadata_path: Path, config: RetrievalConfig = None):
        super().__init__(config or RetrievalConfig())
        
        # 初始化各个检索器
        self.vector_retriever = VectorRetriever(vector_store_path, config)
        self.bm25_retriever = BM25Retriever(bm25_path, chunks_metadata_path, config)
        
        if self.config.enable_llm_reranking:
            self.llm_reranker = LLMReranker(self.config.rerank_model)
    
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        混合检索策略
        """
        k = top_k or self.config.top_k
        
        # 1. 分别进行向量检索和BM25检索
        vector_results = self.vector_retriever.retrieve(
            query, top_k=self.config.rerank_top_k if self.config.enable_llm_reranking else k
        )
        bm25_results = self.bm25_retriever.retrieve(
            query, top_k=self.config.rerank_top_k if self.config.enable_llm_reranking else k
        )
        
        # 2. 合并结果并去重
        all_results = self._merge_results(vector_results, bm25_results)
        
        # 3. 计算混合分数
        for result in all_results:
            vector_score = result.get('vector_score', 0.0)
            bm25_score = result.get('bm25_score', 0.0)
            
            hybrid_score = (
                self.config.vector_weight * vector_score +
                self.config.bm25_weight * bm25_score
            )
            result['hybrid_score'] = hybrid_score
        
        # 4. LLM重排（如果启用）
        if self.config.enable_llm_reranking and self.llm_reranker:
            all_results = self.llm_reranker.rerank_documents(
                query, all_results, self.config.rerank_batch_size
            )
            
            # 计算最终分数
            for result in all_results:
                llm_score = result.get('llm_score', 0.5)
                hybrid_score = result.get('hybrid_score', 0.0)
                
                final_score = (
                    self.config.llm_rerank_weight * llm_score +
                    (1 - self.config.llm_rerank_weight) * hybrid_score
                )
                result['final_score'] = final_score
            
            # 按最终分数排序
            all_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        else:
            # 按混合分数排序
            all_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # 5. 返回top-k结果
        final_results = all_results[:k]
        
        logger.info(f"混合检索返回 {len(final_results)} 个结果")
        return final_results
    
    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        合并并去重检索结果
        """
        # 使用内容hash去重
        seen_contents = set()
        merged_results = []
        
        # 处理向量检索结果
        for result in vector_results:
            content_hash = hash(result['content'])
            if content_hash not in seen_contents:
                result['vector_score'] = result['score']
                result['bm25_score'] = 0.0
                merged_results.append(result)
                seen_contents.add(content_hash)
        
        # 处理BM25检索结果
        for result in bm25_results:
            content_hash = hash(result['content'])
            if content_hash not in seen_contents:
                result['vector_score'] = 0.0
                result['bm25_score'] = result['score']
                merged_results.append(result)
                seen_contents.add(content_hash)
            else:
                # 如果已存在，更新BM25分数
                for existing in merged_results:
                    if hash(existing['content']) == content_hash:
                        existing['bm25_score'] = result['score']
                        break
        
        return merged_results

class ParentDocumentRetriever(BaseRetriever):
    """
    父文档检索器 - 参考冠军方案的父文档检索策略
    """
    
    def __init__(self, base_retriever: BaseRetriever, config: RetrievalConfig = None):
        super().__init__(config or RetrievalConfig())
        self.base_retriever = base_retriever
    
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        父文档检索策略
        """
        k = top_k or self.config.top_k
        
        # 1. 使用基础检索器获取子文档
        child_results = self.base_retriever.retrieve(query, top_k=self.config.parent_k)
        
        # 2. 提取父文档并去重
        parent_docs = {}
        
        for result in child_results:
            metadata = result.get('metadata', {})
            
            if metadata.get('chunk_type') == 'child' and 'parent_text' in metadata:
                parent_id = metadata.get('parent_id')
                
                if parent_id not in parent_docs:
                    parent_docs[parent_id] = {
                        'content': metadata['parent_text'],
                        'metadata': {
                            'parent_id': parent_id,
                            'retrieval_type': 'parent_document',
                            'child_chunks': []
                        },
                        'score': result['score'],
                        'child_count': 1
                    }
                else:
                    # 更新分数（取最高分或平均分）
                    parent_docs[parent_id]['score'] = max(
                        parent_docs[parent_id]['score'], 
                        result['score']
                    )
                    parent_docs[parent_id]['child_count'] += 1
                
                # 记录子块信息
                parent_docs[parent_id]['metadata']['child_chunks'].append({
                    'chunk_id': metadata.get('chunk_id'),
                    'score': result['score']
                })
        
        # 3. 排序并返回top-k父文档
        parent_results = list(parent_docs.values())
        parent_results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results = parent_results[:k]
        
        logger.info(f"父文档检索返回 {len(final_results)} 个结果")
        return final_results

class RetrieverFactory:
    """
    检索器工厂 - 统一管理不同类型的检索器
    """
    
    @staticmethod
    def create_vector_retriever(vector_store_path: Path, config: RetrievalConfig = None) -> VectorRetriever:
        """创建向量检索器"""
        return VectorRetriever(vector_store_path, config)
    
    @staticmethod
    def create_bm25_retriever(bm25_path: Path, chunks_metadata_path: Path, 
                             config: RetrievalConfig = None) -> BM25Retriever:
        """创建BM25检索器"""
        return BM25Retriever(bm25_path, chunks_metadata_path, config)
    
    @staticmethod
    def create_hybrid_retriever(vector_store_path: Path, bm25_path: Path, 
                               chunks_metadata_path: Path, config: RetrievalConfig = None) -> HybridRetriever:
        """创建混合检索器"""
        return HybridRetriever(vector_store_path, bm25_path, chunks_metadata_path, config)
    
    @staticmethod
    def create_parent_document_retriever(base_retriever: BaseRetriever, 
                                       config: RetrievalConfig = None) -> ParentDocumentRetriever:
        """创建父文档检索器"""
        return ParentDocumentRetriever(base_retriever, config)
    
    @staticmethod
    def create_advanced_retriever(vector_store_path: Path, bm25_path: Path, 
                                 chunks_metadata_path: Path, 
                                 enable_parent_retrieval: bool = True,
                                 enable_llm_reranking: bool = False) -> BaseRetriever:
        """
        创建高级检索器 - 根据需求自动组合检索策略
        """
        # 基础配置
        config = RetrievalConfig(
            enable_parent_retrieval=enable_parent_retrieval,
            enable_llm_reranking=enable_llm_reranking,
            top_k=10,
            rerank_top_k=20
        )
        
        # 创建混合检索器
        hybrid_retriever = RetrieverFactory.create_hybrid_retriever(
            vector_store_path, bm25_path, chunks_metadata_path, config
        )
        
        # 如果启用父文档检索，包装为父文档检索器
        if enable_parent_retrieval:
            return RetrieverFactory.create_parent_document_retriever(hybrid_retriever, config)
        else:
            return hybrid_retriever

def demo_usage():
    """
    演示检索器的使用方法
    """
    # 设置路径（根据实际情况修改）
    vector_store_path = Path("output/processed_documents/guide_faiss")
    bm25_path = Path("output/processed_documents/guide_bm25.pkl")
    chunks_metadata_path = Path("output/processed_documents/guide_metadata.json")
    
    # 1. 创建简单向量检索器
    print("=== 向量检索示例 ===")
    try:
        vector_retriever = RetrieverFactory.create_vector_retriever(vector_store_path)
        vector_results = vector_retriever.retrieve("博士后基金申请条件")
        print(f"向量检索结果数量: {len(vector_results)}")
        for i, result in enumerate(vector_results[:3]):
            print(f"结果 {i+1}: 分数={result['score']:.4f}")
            print(f"内容预览: {result['content'][:100]}...")
            print()
    except Exception as e:
        print(f"向量检索演示失败: {e}")
    
    # 2. 创建高级混合检索器
    print("=== 高级混合检索示例 ===")
    try:
        # 创建最强配置的检索器
        advanced_retriever = RetrieverFactory.create_advanced_retriever(
            vector_store_path=vector_store_path,
            bm25_path=bm25_path,
            chunks_metadata_path=chunks_metadata_path,
            enable_parent_retrieval=True,   # 启用父文档检索
            enable_llm_reranking=True       # 启用LLM重排
        )
        
        # 执行检索
        query = "申请博士后基金需要什么材料？"
        results = advanced_retriever.retrieve(query, top_k=5)
        
        print(f"查询: {query}")
        print(f"高级检索结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  类型: {result['metadata'].get('retrieval_type', 'unknown')}")
            print(f"  分数: {result.get('final_score', result.get('score', 0)):.4f}")
            if 'llm_reasoning' in result:
                print(f"  LLM评价: {result['llm_reasoning'][:100]}...")
            print(f"  内容预览: {result['content'][:200]}...")
            
    except Exception as e:
        print(f"高级检索演示失败: {e}")
    
    # 3. 批量查询示例
    print("\n=== 批量查询示例 ===")
    queries = [
        "博士后基金申请截止时间",
        "申请材料包括什么",
        "评审标准是什么",
        "资助金额多少"
    ]
    
    try:
        config = RetrievalConfig(top_k=3, enable_llm_reranking=False)  # 关闭LLM重排以加快速度
        retriever = RetrieverFactory.create_hybrid_retriever(
            vector_store_path, bm25_path, chunks_metadata_path, config
        )
        
        for query in queries:
            results = retriever.retrieve(query)
            print(f"\n查询: {query}")
            print(f"结果: {len(results)} 个文档片段")
            if results:
                best_result = results[0]
                print(f"最佳匹配 (分数={best_result.get('hybrid_score', 0):.4f}): {best_result['content'][:150]}...")
                
    except Exception as e:
        print(f"批量查询演示失败: {e}")

if __name__ == "__main__":
    demo_usage()
