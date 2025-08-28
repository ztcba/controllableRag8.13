#!/usr/bin/env python3
"""
端到端测试脚本：验证“父文档检索”策略 (使用 AiHubMix 重排器)
"""

import sys
from pathlib import Path
from pprint import pprint

# --- 准备工作 ---
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from retriever_factory import create_hybrid_retriever
# 【关键改动】从我们创建的新文件中导入重排器工厂函数
from rag_pipeline.components.rerankers import get_reranker_model

def test_parent_retrieval_strategy(question: str):
    print("="*80)
    print(f"🚀 开始测试父文档检索策略 (使用 AiHubMix 重排器)...")
    print(f"❓ 测试问题: \"{question}\"")
    print("="*80)

    try:
        # --- 步骤一：检索 (Retrieve) ---
        print("\n--- 步骤 1: 检索子块 (Retrieve Child Chunks) ---")
        retriever = create_hybrid_retriever(bm25_k=10, vector_k=10)
        child_chunks = retriever.invoke(question)
        print(f"✅ 成功检索到 {len(child_chunks)} 个子块。")

        # --- 步骤二：扩展 (Expand) ---
        print("\n--- 步骤 2: 扩展为唯一的父块 (Expand to Parent Chunks) ---")
        parent_chunks_map = {}
        for chunk in child_chunks:
            parent_content = chunk.metadata.get("parent_content")
            if parent_content:
                parent_chunks_map[parent_content] = Document(
                    page_content=parent_content, metadata=chunk.metadata
                )
        unique_parent_chunks = list(parent_chunks_map.values())
        print(f"✅ 成功扩展并去重，得到 {len(unique_parent_chunks)} 个唯一的父块。")

        # --- 步骤三：重排 (Rerank) ---
        print("\n--- 步骤 3: 使用 AiHubMix 模型重排父块 (Rerank Parent Chunks) ---")
        
        # 【关键改动】使用我们的自定义重排器工厂函数
        reranker = get_reranker_model(top_n=3)
        
        reranked_docs = reranker.transform_documents(
            documents=unique_parent_chunks,
            query=question
        )
        print(f"✅ 重排完成，选出 Top {len(reranked_docs)} 个最相关的父块。")
        
        # --- 步骤四：结果展示与生成 (Display & Generate) ---
        print("\n--- 步骤 4: 展示最终结果 (Display Final Context) ---")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('relevance_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"\n✨ --- Top {i+1} 相关父块 (得分: {score_str}) --- ✨")
            print(doc.page_content)
        
        final_context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
        print("\n\n" + "="*80)
        print("✅ 最终将提供给 LLM 的聚合上下文:")
        print("="*80)
        pprint(final_context)
        print("\n🎉 测试成功完成！")

    except Exception as e:
        import traceback
        print(f"\n❌ 测试过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_query = "根据《2025中国博士后科学基金资助指南》，面上资助中，自然科学领域的资助标准是多少"
    test_parent_retrieval_strategy(test_query)