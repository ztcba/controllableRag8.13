"""
文档处理器 - 从PDF到向量数据库的完整流程
基于RAG-Challenge-2冠军方案的设计思路，使用langchain和faiss实现
适用于类似博士后基金申请手册的文档处理
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# Langchain imports - 使用新版本的导入路径
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Other imports
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi

# Import project settings
from rag_pipeline.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """处理配置类 - 参考冠军方案的参数设置"""
    # 分块配置
    chunk_size: int = 400  # 适合中文文档，比原方案稍大
    chunk_overlap: int = 60  # 保证上下文连续性
    chunk_min_size: int = 50  # 最小分块大小
    
    # 嵌入配置 - 优先使用项目配置
    embedding_model: str = None  # 将在__post_init__中设置
    embedding_batch_size: int = 100  # 批处理大小
    
    # Faiss配置
    faiss_index_type: str = "IndexFlatIP"  # 内积索引，支持余弦相似度
    normalize_embeddings: bool = True  # 归一化向量
    
    # 并行处理配置
    max_workers: int = 4  # 并行处理的最大工作线程数
    
    # 父文档配置（参考冠军方案的parent_document_retrieval）
    enable_parent_document: bool = True  # 启用父文档检索
    parent_chunk_size: int = 1200  # 父文档块大小
    
    # 元数据增强
    enhance_metadata: bool = True  # 是否增强元数据
    
    # 输出配置
    save_intermediate_files: bool = True  # 保存中间文件用于调试
    
    def __post_init__(self):
        """设置默认嵌入模型"""
        if self.embedding_model is None:
            self.embedding_model = settings.embedding_model

class DocumentProcessor:
    """
    文档处理器主类
    参考RAG-Challenge-2的PDFParser和相关组件设计
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self._setup_environment()
        self._setup_components()
        
    def _setup_environment(self):
        """设置环境"""
        # 使用统一的项目配置
        if not settings.openai_api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
    
    def _setup_components(self):
        """初始化组件"""
        # 初始化嵌入器 - 使用项目的LLM配置
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.llm_base_url  # 使用配置的API基础URL
        )
        
        # 初始化文本分割器 - 参考冠军方案的TextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",  # 使用具体模型的tokenizer确保token计数准确
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 父文档分割器
        if self.config.enable_parent_document:
            self.parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4o",
                chunk_size=self.config.parent_chunk_size,
                chunk_overlap=100
            )
        
        # 初始化OpenAI客户端用于额外功能
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url  # 使用配置的API基础URL
        )
    
    def _count_tokens(self, text: str, encoding_name: str = "o200k_base") -> int:
        """计算token数量 - 参考冠军方案"""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    
    def _clean_text(self, text: str) -> str:
        """
        清洁文本 - 参考冠军方案的_clean_text方法
        处理PDF解析中的常见问题
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 修复常见的PDF解析问题
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # 修复断词
        text = re.sub(r'\n+', '\n', text)  # 合并多个换行
        text = text.strip()
        
        return text
    
    def _extract_metadata(self, pdf_path: Path, document_text: str) -> Dict[str, Any]:
        """
        提取文档元数据 - 参考冠军方案的metainfo结构
        """
        metadata = {
            'filename': pdf_path.name,
            'file_path': str(pdf_path),
            'file_size': pdf_path.stat().st_size,
            'total_characters': len(document_text),
            'total_tokens': self._count_tokens(document_text),
            'processing_config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'embedding_model': self.config.embedding_model
            }
        }
        
        # 如果启用元数据增强，使用LLM提取结构化信息
        if self.config.enhance_metadata:
            try:
                enhanced_metadata = self._extract_enhanced_metadata(document_text[:2000])  # 只用前2000字符
                metadata.update(enhanced_metadata)
            except Exception as e:
                logger.warning(f"元数据增强失败: {e}")
        
        return metadata
    
    def _extract_enhanced_metadata(self, text_sample: str) -> Dict[str, Any]:
        """
        使用LLM提取增强的元数据信息
        """
        system_prompt = """
        您是一个文档分析专家。请分析以下文档片段，提取关键信息：
        1. 文档类型（如：申请指南、手册、报告等）
        2. 主要主题（如：博士后基金、研究申请等）
        3. 文档语言
        4. 是否包含表格、列表或结构化信息
        
        请以JSON格式返回结果。
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下文档片段：\n\n{text_sample}"}
                ],
                temperature=0
            )
            
            # 尝试解析JSON响应
            response_text = response.choices[0].message.content
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # 如果不是标准JSON，返回文本分析
                return {"ai_analysis": response_text}
                
        except Exception as e:
            logger.warning(f"LLM元数据提取失败: {e}")
            return {}
    
    def _load_pdf_with_langchain(self, pdf_path: Path) -> List[Document]:
        """
        使用langchain加载PDF文档
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # 清洁和处理文档
            processed_docs = []
            for i, doc in enumerate(documents):
                cleaned_text = self._clean_text(doc.page_content)
                
                # 增强页面级元数据
                doc.metadata.update({
                    'page_number': i + 1,
                    'total_pages': len(documents),
                    'characters_count': len(cleaned_text),
                    'tokens_count': self._count_tokens(cleaned_text)
                })
                
                doc.page_content = cleaned_text
                processed_docs.append(doc)
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"PDF加载失败 {pdf_path}: {e}")
            raise
    
    def _create_chunks(self, documents: List[Document], document_metadata: Dict) -> List[Document]:
        """
        创建文档块 - 实现父子文档检索策略
        参考冠军方案的分块和父文档检索设计
        """
        all_chunks = []
        
        # 合并所有页面文本
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        if self.config.enable_parent_document:
            # 创建父文档块
            parent_docs = self.parent_splitter.split_text(full_text)
            
            # 为每个父文档创建子块
            for parent_idx, parent_text in enumerate(parent_docs):
                child_chunks = self.text_splitter.split_text(parent_text)
                
                for child_idx, child_text in enumerate(child_chunks):
                    if len(child_text.strip()) < self.config.chunk_min_size:
                        continue
                    
                    # 创建子文档，包含父文档信息
                    chunk_metadata = {
                        **document_metadata,
                        'chunk_type': 'child',
                        'parent_id': parent_idx,
                        'child_id': child_idx,
                        'chunk_id': f"{parent_idx}_{child_idx}",
                        'parent_text': parent_text,  # 保存父文档文本用于检索
                        'chunk_size': len(child_text),
                        'chunk_tokens': self._count_tokens(child_text)
                    }
                    
                    chunk_doc = Document(
                        page_content=child_text,
                        metadata=chunk_metadata
                    )
                    all_chunks.append(chunk_doc)
        else:
            # 常规分块
            chunks = self.text_splitter.split_text(full_text)
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < self.config.chunk_min_size:
                    continue
                
                chunk_metadata = {
                    **document_metadata,
                    'chunk_type': 'standard',
                    'chunk_id': i,
                    'chunk_size': len(chunk_text),
                    'chunk_tokens': self._count_tokens(chunk_text)
                }
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"创建了 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def _create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        创建FAISS向量存储
        参考冠军方案的VectorDBIngestor设计
        """
        try:
            logger.info("正在创建向量存储...")
            
            # 使用langchain的FAISS包装器
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            logger.info(f"向量存储创建完成，包含 {len(chunks)} 个向量")
            return vector_store
            
        except Exception as e:
            logger.error(f"向量存储创建失败: {e}")
            raise
    
    def _create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        """
        创建BM25索引 - 参考冠军方案的BM25Ingestor
        """
        try:
            logger.info("正在创建BM25索引...")
            
            # 提取文本并分词
            texts = [chunk.page_content for chunk in chunks]
            tokenized_texts = [text.split() for text in texts]
            
            # 创建BM25索引
            bm25_index = BM25Okapi(tokenized_texts)
            
            logger.info("BM25索引创建完成")
            return bm25_index
            
        except Exception as e:
            logger.error(f"BM25索引创建失败: {e}")
            raise
    
    def process_single_document(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        处理单个PDF文档
        """
        logger.info(f"开始处理文档: {pdf_path.name}")
        
        # 1. 加载PDF
        documents = self._load_pdf_with_langchain(pdf_path)
        
        # 2. 提取元数据
        full_text = "\n".join([doc.page_content for doc in documents])
        document_metadata = self._extract_metadata(pdf_path, full_text)
        
        # 3. 创建文档块
        chunks = self._create_chunks(documents, document_metadata)
        
        # 4. 创建向量存储
        vector_store = self._create_vector_store(chunks)
        
        # 5. 创建BM25索引
        bm25_index = self._create_bm25_index(chunks)
        
        # 6. 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存向量存储
        vector_store_path = output_dir / f"{pdf_path.stem}_faiss"
        vector_store_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        vector_store.save_local(str(vector_store_path))
        
        # 保存BM25索引
        bm25_path = output_dir / f"{pdf_path.stem}_bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        
        # 保存元数据和分块信息
        if self.config.save_intermediate_files:
            metadata_path = output_dir / f"{pdf_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_metadata': document_metadata,
                    'chunks_info': [
                        {
                            'chunk_id': chunk.metadata.get('chunk_id'),
                            'chunk_size': chunk.metadata.get('chunk_size'),
                            'chunk_tokens': chunk.metadata.get('chunk_tokens'),
                            'content_preview': chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                        }
                        for chunk in chunks
                    ]
                }, f, ensure_ascii=False, indent=2)
        
        result = {
            'pdf_path': str(pdf_path),
            'vector_store_path': str(vector_store_path),
            'bm25_path': str(bm25_path),
            'chunks_count': len(chunks),
            'document_metadata': document_metadata
        }
        
        logger.info(f"文档处理完成: {pdf_path.name}")
        return result
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """
        批量处理目录中的PDF文档
        """
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"在 {input_dir} 中未找到PDF文件")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        results = []
        
        # 串行处理（避免OpenAI API并发限制）
        for pdf_path in tqdm(pdf_files, desc="处理PDF文件"):
            try:
                result = self.process_single_document(pdf_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件 {pdf_path.name} 时出错: {e}")
                continue
        
        # 保存批处理结果摘要
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_files': len(results),
                'total_files': len(pdf_files),
                'results': results,
                'config': {
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'embedding_model': self.config.embedding_model,
                    'enable_parent_document': self.config.enable_parent_document
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批处理完成，成功处理 {len(results)}/{len(pdf_files)} 个文件")
        return results

def main():
    """
    主函数 - 演示使用方法
    """
    # 配置处理参数
    config = ProcessingConfig(
        chunk_size=400,  # 适合中文文档
        chunk_overlap=60,
        embedding_model="text-embedding-3-large",
        enable_parent_document=True,  # 启用父文档检索
        enhance_metadata=True  # 启用元数据增强
    )
    
    # 创建处理器
    processor = DocumentProcessor(config)
    
    # 处理单个文档
    input_path = Path("data/pdfs")  # 替换为实际PDF路径(目录或者单个文件)
    output_dir = Path("output/processed_documents")
    
    if input_path.suffix.lower() == '.pdf':
        result = processor.process_single_document(input_path, output_dir)
        print(f"处理结果: {result}")
    else:
        # 如果是目录，批量处理
        results = processor.process_directory(input_path, output_dir)
        print(f"批量处理完成，处理了 {len(results)} 个文件")

if __name__ == "__main__":
    main()
