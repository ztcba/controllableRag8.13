# main_factual_test.py
"""
测试事实检索Agent的主函数
用于验证Factual Retrieval Sub-Graph和查询分类的逻辑是否正常工作
"""
import uuid
from pprint import pprint
from rag_pipeline.graph.agent_factual import create_factual_test_agent
from rag_pipeline.utils.helpers import text_wrap

def run_factual_test_agent():
    """
    初始化并运行事实检索测试Agent。
    """
    # 1. 创建agent
    # agent在调用此函数时被编译
    agent_app = create_factual_test_agent()
    print("Factual Test Agent created successfully.")

    # 2. 定义agent的输入
    # 输入必须匹配MainGraphState的结构
    question = "中国博士后科学基金面上资助的金额是多少？"
    
    initial_state = {
        "question": question,
        "user_profile": None,
        "query_type": "",
        "final_answer": "",
        "error": None,
        "history": []
    }
    
    # 3. 定义运行配置
    # `thread_id`对于LangGraph管理对话或运行状态是必需的
    # 每次运行都应该有唯一的ID
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n=============================================")
    print(f"Running factual test agent with question:\n{text_wrap(question)}")
    print("=============================================\n")
    
    # 4. 流式执行agent
    # 流式执行允许我们在每个节点运行时查看输出
    try:
        for event in agent_app.stream(initial_state, config=config):
            pprint(event)
            print("\n---\n")

        # 5. 获取最终响应
        # 流式执行完成后，最终状态可用
        final_state = agent_app.get_state(config)
        final_response = final_state.values.get('final_answer', '未能获取最终答案')
        
        print("\n=============================================")
        print("Agent execution finished.")
        print(f"Final Answer:\n{text_wrap(final_response)}")
        print("=============================================\n")
        
    except Exception as e:
        print(f"\n❌ Error during agent execution: {str(e)}")
        print("This might be due to:")
        print("1. Vector store loading issues")
        print("2. Missing .pkl file for FAISS index")
        print("3. API connection problems")
        print("4. Configuration issues")
        
        # 尝试诊断问题
        print("\n--- Diagnostic Information ---")
        try:
            from rag_pipeline.settings import settings
            print(f"Vector store path: {settings.vector_store_path}")
            print(f"Embedding model: {settings.embedding_model}")
            print(f"Chat model: {settings.chat_model}")
        except Exception as settings_error:
            print(f"Settings error: {settings_error}")


def test_simple_retrieval_queries():
    """
    测试多个简单检索查询
    """
    print("=" * 60)
    print("TESTING MULTIPLE SIMPLE RETRIEVAL QUERIES")
    print("=" * 60)
    
    test_questions = [
        "博士后科学基金面上资助的金额是多少？",
        "申请特别资助（站中）需要满足哪些基本条件？",
        "博士后科学基金的申请流程是什么？",
        "面上资助的评审标准有哪些？"
    ]
    
    agent_app = create_factual_test_agent()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} Test {i}/4 {'='*20}")
        print(f"Question: {question}")
        print("-" * 50)
        
        initial_state = {
            "question": question,
            "user_profile": None,
            "query_type": "",
            "final_answer": "",
            "error": None,
            "history": []
        }
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        try:
            # 简化输出，只显示最终结果
            result = agent_app.invoke(initial_state, config=config)
            final_answer = result.get('final_answer', '未能获取答案')
            print(f"Answer: {text_wrap(final_answer[:200])}...")
            print(f"Query Type: {result.get('query_type', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)


if __name__ == "__main__":
    # 运行单个问题测试
    print("🧪 Starting single question test...")
    run_factual_test_agent()
    
    # 运行多问题测试
    print("\n🧪 Starting multiple questions test...")
    test_simple_retrieval_queries()
