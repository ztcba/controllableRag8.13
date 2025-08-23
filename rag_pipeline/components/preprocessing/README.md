# 基于RAG-Challenge-2冠军方案的文档处理和检索系统

## 概述

本系统基于RAG-Challenge-2冠军方案的设计思路，使用langchain和faiss重新实现，专门用于处理类似博士后基金申请手册的复杂文档。系统包含两个核心组件：

1. **document_processor.py** - 从PDF到向量数据库的完整处理流程
2. **retriever_factory.py** - 多策略检索器工厂

## 核心特性

### 基于冠军方案的设计理念
- **父子文档检索**: 参考原方案的parent_document_retrieval策略
- **混合检索**: 结合向量检索和BM25检索
- **LLM重排**: 使用大模型对检索结果进行相关性重排
- **分块策略**: 优化的文本分块，适合中文文档
- **元数据增强**: 使用LLM提取文档结构化信息

### 检索策略对比

| 策略 | 适用场景 | 优势 | 参考原方案组件 |
|------|----------|------|----------------|
| 向量检索 | 语义相似性查询 | 理解语义，跨语言支持 | VectorRetriever |
| BM25检索 | 关键词精确匹配 | 快速，术语匹配准确 | BM25Retriever |
| 混合检索 | 平衡语义和关键词 | 综合优势 | HybridRetriever |
| 父文档检索 | 需要完整上下文 | 保持文档结构完整性 | parent_document_retrieval |
| LLM重排 | 复杂查询理解 | 最高质量的相关性判断 | LLMReranker |

## 安装和配置

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements_custom.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，添加OPENAI_API_KEY
```

### 2. 环境变量配置
创建`.env`文件：
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用指南

### 1. 文档处理（从PDF到向量数据库）

```python
from document_processor import DocumentProcessor, ProcessingConfig
from pathlib import Path

# 配置处理参数（基于冠军方案优化）
config = ProcessingConfig(
    chunk_size=400,              # 适合中文文档
    chunk_overlap=60,            # 保证上下文连续性
    embedding_model="text-embedding-3-large",  # 最佳嵌入模型
    enable_parent_document=True,  # 启用父文档检索
    enhance_metadata=True         # 启用LLM元数据增强
)

# 创建处理器
processor = DocumentProcessor(config)

# 处理单个PDF
pdf_path = Path("documents/guide.pdf")
output_dir = Path("output/processed")
result = processor.process_single_document(pdf_path, output_dir)

# 批量处理目录中的PDF
input_dir = Path("documents/")
results = processor.process_directory(input_dir, output_dir)
```

### 2. 检索器使用

#### 基础检索示例
```python
from retriever_factory import RetrieverFactory, RetrievalConfig
from pathlib import Path

# 设置文件路径
vector_store_path = Path("output/processed/guide_faiss")
bm25_path = Path("output/processed/guide_bm25.pkl")
metadata_path = Path("output/processed/guide_metadata.json")

# 创建向量检索器
vector_retriever = RetrieverFactory.create_vector_retriever(vector_store_path)
results = vector_retriever.retrieve("博士后基金申请条件", top_k=5)
```

#### 高级混合检索（推荐）
```python
# 创建最强配置的检索器
advanced_retriever = RetrieverFactory.create_advanced_retriever(
    vector_store_path=vector_store_path,
    bm25_path=bm25_path,
    chunks_metadata_path=metadata_path,
    enable_parent_retrieval=True,    # 启用父文档检索
    enable_llm_reranking=True        # 启用LLM重排
)

# 执行查询
query = "申请博士后基金需要什么材料？"
results = advanced_retriever.retrieve(query, top_k=5)

# 查看结果
for i, result in enumerate(results):
    print(f"结果 {i+1}:")
    print(f"  分数: {result.get('final_score', 0):.4f}")
    print(f"  内容: {result['content'][:200]}...")
    if 'llm_reasoning' in result:
        print(f"  LLM评价: {result['llm_reasoning']}")
```

#### 自定义检索配置
```python
# 自定义检索配置
config = RetrievalConfig(
    top_k=10,                     # 返回结果数量
    vector_weight=0.7,            # 向量检索权重
    bm25_weight=0.3,              # BM25检索权重
    enable_llm_reranking=True,    # 启用LLM重排
    rerank_top_k=20,              # 重排候选数量
    llm_rerank_weight=0.8,        # LLM重排权重
    rerank_model="gpt-4o-mini"    # 重排使用的模型
)

hybrid_retriever = RetrieverFactory.create_hybrid_retriever(
    vector_store_path, bm25_path, metadata_path, config
)
```

## 配置参数详解

### ProcessingConfig（文档处理配置）
```python
@dataclass
class ProcessingConfig:
    # 分块配置 - 参考冠军方案的TextSplitter
    chunk_size: int = 400          # 文档分块大小（tokens）
    chunk_overlap: int = 60        # 分块重叠大小
    chunk_min_size: int = 50       # 最小分块大小
    
    # 嵌入配置 - 参考冠军方案的嵌入设置
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    
    # 父文档配置 - 参考parent_document_retrieval
    enable_parent_document: bool = True   # 启用父文档检索
    parent_chunk_size: int = 1200        # 父文档块大小
    
    # 元数据增强 - 参考冠军方案的元数据处理
    enhance_metadata: bool = True         # 使用LLM增强元数据
```

### RetrievalConfig（检索配置）
```python
@dataclass
class RetrievalConfig:
    # 基础检索 - 参考冠军方案的检索参数
    top_k: int = 10                      # 返回文档数量
    score_threshold: float = 0.0         # 分数阈值
    
    # 混合检索 - 参考HybridRetriever权重设置
    vector_weight: float = 0.7           # 向量检索权重
    bm25_weight: float = 0.3             # BM25检索权重
    
    # 父文档检索 - 参考parent_document_retrieval
    enable_parent_retrieval: bool = True  # 启用父文档检索
    parent_k: int = 20                   # 检索父文档数量
    
    # LLM重排 - 参考LLMReranker设置
    enable_llm_reranking: bool = False   # 启用LLM重排
    rerank_top_k: int = 20               # 重排候选数量
    llm_rerank_weight: float = 0.7       # LLM重排权重
```

## 性能优化建议

### 1. 根据文档特点选择策略
- **简单查询**: 使用向量检索
- **关键词查询**: 使用BM25检索
- **复杂查询**: 使用混合检索 + LLM重排
- **需要完整上下文**: 启用父文档检索

### 2. 参数调优
```python
# 中文文档优化配置
chinese_config = ProcessingConfig(
    chunk_size=400,               # 中文字符密度高，适当增大
    chunk_overlap=60,             # 保证词语完整性
    embedding_model="text-embedding-3-large"  # 对中文支持最好
)

# 高精度检索配置
high_precision_config = RetrievalConfig(
    enable_llm_reranking=True,    # 启用LLM重排提高精度
    rerank_top_k=30,              # 增大候选池
    llm_rerank_weight=0.8         # 更信任LLM判断
)

# 高速检索配置
high_speed_config = RetrievalConfig(
    enable_llm_reranking=False,   # 关闭LLM重排加快速度
    top_k=5,                      # 减少返回数量
    vector_weight=0.8             # 主要依赖向量检索
)
```

## 与原冠军方案的对应关系

| 本系统组件 | 原方案组件 | 功能对应 |
|------------|------------|----------|
| DocumentProcessor | PDFParser + PageTextPreparation | PDF解析和文本预处理 |
| VectorRetriever | VectorRetriever | 向量检索 |
| BM25Retriever | BM25Retriever | BM25检索 |
| HybridRetriever | HybridRetriever | 混合检索策略 |
| LLMReranker | LLMReranker | LLM重排 |
| ParentDocumentRetriever | parent_document_retrieval | 父文档检索 |
| ProcessingConfig | TextSplitter配置 | 分块参数 |
| RetrievalConfig | QuestionsProcessor配置 | 检索参数 |

## 常见问题

### Q1: 如何处理大型PDF文档？
```python
# 启用并行处理和分批处理
config = ProcessingConfig(
    chunk_size=300,               # 减小分块避免token限制
    max_workers=2,                # 控制并行数避免API限制
    save_intermediate_files=True  # 保存中间结果便于断点续传
)
```

### Q2: 如何提高中文检索准确性？
```python
# 中文优化配置
config = RetrievalConfig(
    vector_weight=0.8,            # 向量检索对中文语义理解更好
    bm25_weight=0.2,              # 降低BM25权重（分词可能不准确）
    enable_llm_reranking=True     # LLM对中文理解最准确
)
```

### Q3: 如何平衡检索速度和精度？
```python
# 根据应用场景选择不同配置

# 实时查询场景（速度优先）
fast_config = RetrievalConfig(
    top_k=5,
    enable_llm_reranking=False,
    enable_parent_retrieval=False
)

# 深度分析场景（精度优先）
accurate_config = RetrievalConfig(
    top_k=15,
    enable_llm_reranking=True,
    enable_parent_retrieval=True,
    rerank_top_k=30
)
```

## 扩展和定制

### 1. 自定义检索器
```python
class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        # 实现自定义检索逻辑
        pass
```

### 2. 自定义重排器
```python
class CustomReranker:
    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        # 实现自定义重排逻辑
        pass
```

### 3. 集成其他向量数据库
系统设计支持轻松集成其他向量数据库，只需实现相应的接口。

## 许可证

本项目基于RAG-Challenge-2冠军方案的开源设计思路开发，遵循相同的开源许可证。
