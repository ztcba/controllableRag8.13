#!/usr/bin/env python3
"""
测试优化后的检索器工厂加载功能

该脚本验证 `retriever_factory_optimized.py` 是否能够：
1. 成功从磁盘加载预处理好的 BM25 和 FAISS 索引。
2. 正确组装成一个可用的混合检索器 (EnsembleRetriever)。
3. 针对指定问题执行检索并返回结果。
"""

import sys
from pathlib import Path
import traceback

# --- 准备环境 ---
# 确保脚本可以找到 retriever_factory_optimized 模块
try:
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    from retriever_factory import create_hybrid_retriever
except ImportError:
    print("❌ 错误：无法导入 'create_hybrid_retriever'。")
    print("   请确保此脚本与 'retriever_factory.py' 在同一个目录下。")
    sys.exit(1)

# --- 测试主逻辑 ---
def run_test():
    """执行完整的测试流程"""
    print("--- 开始测试检索器加载与查询功能 ---")

    # 1. 定义测试问题
    # 注意：将"面相"修正为更通顺的"面向"
    test_query = "自然科学基金委面向哪些学校"
    print(f"\n🎯 测试问题: '{test_query}'")

    try:
        # 2. 从工厂创建检索器实例
        # 第一次调用会从磁盘加载，后续调用会使用缓存
        print("\n步骤 1: 正在调用 create_hybrid_retriever() 来创建检索器...")
        retriever = create_hybrid_retriever(
            bm25_k=3,       # 为了测试，我们只取前3个结果
            vector_k=3,     # 同上
            hybrid_weights=[0.5, 0.5] # 使用50/50的权重
        )
        print(f"✅ 检索器创建成功！类型: {type(retriever)}")

        # 3. 使用检索器执行查询
        print(f"\n步骤 2: 正在使用检索器查询相关文档...")
        results = retriever.invoke(test_query)
        print(f"✅ 查询完成！共找到 {len(results)} 个相关文档片段。")

        # 4. 打印和分析结果
        print("\n--- 查询结果展示 ---")
        if not results:
            print("⚠️ 未找到任何相关结果。这可能是因为文档内容与问题不匹配，或者索引构建有问题。")
        else:
            for i, doc in enumerate(results, 1):
                # 提取元数据中的源文件名
                source_file = doc.metadata.get('source', '未知来源')
                
                # 清理和截断页面内容以便预览
                content_preview = doc.page_content.strip().replace('\n', ' ')
                if len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."
                    
                print(f"\n📄 结果 {i}:")
                print(f"   来源: {source_file}")
                print(f"   内容: {content_preview}")
        
        print("\n\n--- 测试结论 ---")
        print("🎉 检索器工厂函数运行正常，成功加载索引并返回了查询结果！")

    except FileNotFoundError as e:
        print("\n❌ 测试失败：索引文件未找到！")
        print(f"   错误详情: {e}")
        print("   请确保您已经成功运行了 'build_indices.py' 脚本，")
        print("   并且索引文件位于 'output/retriever_indices/' 目录下。")
        traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ 测试失败：在测试过程中发生未知错误。")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误详情: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()