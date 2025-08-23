# rag_pipeline/graph/agent_factual.py
"""
测试用的事实检索Agent，用于验证Factual Retrieval Sub-Graph的逻辑和代码是否正常工作。
包含查询分类和事实检索的完整流程。
"""

from langgraph.graph import END, StateGraph

# Import the main state and all the nodes/conditions
from rag_pipeline.graph import state
from rag_pipeline.graph import nodes

def create_factual_test_agent():
    """
    构建测试用的事实检索agent图。
    
    流程：
    1. classify_query - 查询分类
    2. 根据分类结果路由到相应的子图
    3. 如果是simple_retrieval，则调用factual子图
    4. 其他类型暂时返回简单响应
    """
    agent_workflow = StateGraph(state.MainGraphState)

    # 添加主图节点
    agent_workflow.add_node("classify_query", nodes.classify_query)
    agent_workflow.add_node("run_factual_subgraph", run_factual_subgraph)
    agent_workflow.add_node("handle_other_queries", handle_other_queries)
    
    # 设置入口点
    agent_workflow.set_entry_point("classify_query")
    
    # 添加条件路由
    agent_workflow.add_conditional_edges(
        "classify_query",
        route_based_on_query_type,
        {
            "simple_retrieval": "run_factual_subgraph",
            "other": "handle_other_queries"
        }
    )
    
    # 添加结束边
    agent_workflow.add_edge("run_factual_subgraph", END)
    agent_workflow.add_edge("handle_other_queries", END)

    # 编译图
    factual_test_app = agent_workflow.compile()
    return factual_test_app


def route_based_on_query_type(state: state.MainGraphState):
    """
    基于查询类型进行路由的条件函数。
    """
    query_type = state.get("query_type", "").strip().lower()
    
    if query_type == "simple_retrieval":
        return "simple_retrieval"
    else:
        return "other"


def run_factual_subgraph(state: state.MainGraphState):
    """
    运行事实检索子图的节点。
    将MainGraphState转换为FactualSubGraphState，调用factual workflow，然后将结果转换回来。
    """
    print("---RUNNING FACTUAL SUB-GRAPH---")
    
    # 准备factual子图的输入状态
    factual_input = {
        "question": state["question"],
        "enhanced_question": None,
        "documents": [],
        "graded_documents": None,
        "context": "",
        "generation": ""
    }
    
    try:
        # 导入并调用factual检索工作流
        from rag_pipeline.graph.workflows import factual_retrieval_workflow_app
        factual_result = factual_retrieval_workflow_app.invoke(factual_input)
        
        # 提取生成的答案
        final_answer = factual_result.get("generation", "无法生成答案")
        
        print(f"---FACTUAL SUB-GRAPH COMPLETED---")
        print(f"Final Answer: {final_answer[:100]}...")
        
        # 更新历史记录
        history = state.get("history", [])
        history.append("factual_subgraph_completed")
        
        return {
            "final_answer": final_answer,
            "history": history
        }
        
    except Exception as e:
        print(f"---ERROR IN FACTUAL SUB-GRAPH: {str(e)}---")
        error_msg = f"事实检索过程中出现错误：{str(e)}"
        
        return {
            "final_answer": error_msg,
            "error": error_msg,
            "history": state.get("history", []) + ["factual_subgraph_error"]
        }


def handle_other_queries(state: state.MainGraphState):
    """
    处理非simple_retrieval类型查询的占位符节点。
    """
    print("---HANDLING OTHER QUERY TYPES---")
    
    query_type = state.get("query_type", "unknown")
    
    message = f"检测到查询类型：{query_type}。此测试Agent目前仅支持simple_retrieval类型的查询。"
    
    # 更新历史记录
    history = state.get("history", [])
    history.append(f"handled_query_type_{query_type}")
    
    return {
        "final_answer": message,
        "history": history
    }
