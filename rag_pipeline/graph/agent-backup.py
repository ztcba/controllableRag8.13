# 这是整个应用的顶层组装逻辑。将其放在一个单独的文件中，可以让我们清晰地看到整个 Plan-and-Execute 流程的宏观结构，而不被节点的具体实现细节所干扰。
# create_agent 函数

# rag_pipeline/graph/agent.py
from langgraph.graph import END, StateGraph

# Import the main state and all the nodes/conditions
from rag_pipeline.graph import state
from rag_pipeline.graph import nodes

def create_agent():
    """
    Builds the main plan-and-execute agent graph.
    """
    agent_workflow = StateGraph(state.PlanExecute)

    # Add all the nodes to the graph
    agent_workflow.add_node("anonymize_question", nodes.anonymize_queries)
    agent_workflow.add_node("planner", nodes.plan_step)
    agent_workflow.add_node("de_anonymize_plan", nodes.deanonymize_queries)
    agent_workflow.add_node("break_down_plan", nodes.break_down_plan_step)
    agent_workflow.add_node("task_handler", nodes.run_task_handler_chain)
    
    # Nodes that run the sub-workflows
    agent_workflow.add_node("retrieve_chunks", nodes.run_qualitative_chunks_retrieval_workflow)
    agent_workflow.add_node("retrieve_summaries", nodes.run_qualitative_summaries_retrieval_workflow)
    agent_workflow.add_node("retrieve_book_quotes", nodes.run_qualitative_book_quotes_retrieval_workflow)
    agent_workflow.add_node("answer", nodes.run_qualtative_answer_workflow)
    
    agent_workflow.add_node("replan", nodes.replan_step)
    agent_workflow.add_node("get_final_answer", nodes.run_qualtative_answer_workflow_for_final_answer)

    # Define the graph's control flow (edges)
    agent_workflow.set_entry_point("anonymize_question")

    agent_workflow.add_edge("anonymize_question", "planner")
    agent_workflow.add_edge("planner", "de_anonymize_plan")
    agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")
    agent_workflow.add_edge("break_down_plan", "task_handler")

    agent_workflow.add_conditional_edges(
        "task_handler", 
        nodes.retrieve_or_answer, 
        {
            "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
            "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
            "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
            "chosen_tool_is_answer": "answer"
        }
    )

    agent_workflow.add_edge("retrieve_chunks", "replan")
    agent_workflow.add_edge("retrieve_summaries", "replan")
    agent_workflow.add_edge("retrieve_book_quotes", "replan")
    agent_workflow.add_edge("answer", "replan")

    # agent_workflow.add_conditional_edges(
    #     "replan",
    #     nodes.can_be_answered,
    #     {
    #         "can_be_answered_already": "get_final_answer",
    #         "cannot_be_answered_yet": "task_handler" # Modified to go back to task_handler with the new plan
    #     }
    # )
    # In the original code, it went back to "break_down_plan". 
    # Going to "task_handler" seems more direct since the replanned steps should already be actionable.
    # Let's stick to the original logic for now to be safe.
    # 经过分析planner 和 replanner 扮演的是“策略规划者”的角色，它们负责制定高层次的行动方针
    # break_down_plan 扮演的是“任务分解器”或“运营官”的角色，它负责将高层策略翻译成具体、可执行的工具调用指令。
    agent_workflow.add_conditional_edges(
        "replan",
        nodes.can_be_answered,
        {
            "can_be_answered_already": "get_final_answer",
            "cannot_be_answered_yet": "break_down_plan" # Sticking to original logic
        }
    )

    agent_workflow.add_edge("get_final_answer", END)

    # Compile the graph into a runnable app
    plan_and_execute_app = agent_workflow.compile()
    return plan_and_execute_app