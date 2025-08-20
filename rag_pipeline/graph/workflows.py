# 创建可复用子图（如 create_qualitative_retrieval_..._workflow_app）的逻辑
# 迁移到 rag_pipeline/graph/workflows.py

# rag_pipeline/graph/workflows.py
from langgraph.graph import END, StateGraph

# Import the state and nodes required for these workflows
from rag_pipeline.graph import state
from rag_pipeline.graph import nodes

# --- Factual Retrieval Sub-Graph Workflow ---

def create_factual_retrieval_workflow_app():
    """
    Builds the factual retrieval sub-graph for processing single factual questions.
    This is the fundamental building block used by more complex sub-graphs.
    """
    workflow = StateGraph(state.FactualSubGraphState)
    
    # Define the nodes
    workflow.add_node("enhance_query", nodes.enhance_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("rerank_documents", nodes.rerank_documents)
    workflow.add_node("generate_answer", nodes.generate_answer)
    
    # Build the graph flow
    workflow.set_entry_point("enhance_query")
    workflow.add_edge("enhance_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "rerank_documents")
    workflow.add_edge("rerank_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# --- Analytical Sub-Graph Workflow ---

def create_analytical_sub_graph_workflow_app():
    """
    Builds the analytical sub-graph for processing complex analytical questions.
    Uses sub-query decomposition and factual sub-graph invocation.
    """
    workflow = StateGraph(state.AnalyticalSubGraphState)
    
    # Define the nodes
    workflow.add_node("generate_sub_queries", nodes.generate_sub_queries)
    workflow.add_node("process_sub_queries", nodes.process_sub_queries)
    workflow.add_node("synthesize_analytical_answer", nodes.synthesize_analytical_answer)
    
    # Build the graph flow
    workflow.set_entry_point("generate_sub_queries")
    workflow.add_edge("generate_sub_queries", "process_sub_queries")
    workflow.add_edge("process_sub_queries", "synthesize_analytical_answer")
    workflow.add_edge("synthesize_analytical_answer", END)
    
    return workflow.compile()

# --- Tool Use Sub-Graph Workflow ---

def create_tool_use_sub_graph_workflow_app():
    """
    Builds the tool use sub-graph for processing questions requiring external tools.
    Handles web search and external information integration.
    """
    workflow = StateGraph(state.ToolUseSubGraphState)
    
    # Define the nodes
    workflow.add_node("decide_tool_use", nodes.decide_tool_use)
    workflow.add_node("generate_search_query", nodes.generate_search_query)
    workflow.add_node("execute_web_search", nodes.execute_web_search)
    workflow.add_node("generate_tool_use_answer", nodes.generate_tool_use_answer)
    
    # Build the graph flow
    workflow.set_entry_point("decide_tool_use")
    
    # Add conditional routing based on tool decision
    def should_use_tool(state):
        return "use_tool" if state.get("needs_tool", False) else "no_tool_needed"
    
    workflow.add_conditional_edges(
        "decide_tool_use",
        should_use_tool,
        {
            "use_tool": "generate_search_query",
            "no_tool_needed": "generate_tool_use_answer"  # Generate answer without search
        }
    )
    
    workflow.add_edge("generate_search_query", "execute_web_search")
    workflow.add_edge("execute_web_search", "generate_tool_use_answer")
    workflow.add_edge("generate_tool_use_answer", END)
    
    return workflow.compile()

# --- Qualitative Retrieval Workflows ---

def create_qualitative_retrieval_book_chunks_workflow_app():
    """Builds the sub-graph for retrieving and filtering book chunks."""
    workflow = StateGraph(state.QualitativeRetrievalGraphState)

    # Define the nodes
    workflow.add_node("retrieve_chunks_context_per_question", nodes.retrieve_chunks_context_per_question)
    workflow.add_node("keep_only_relevant_content", nodes.keep_only_relevant_content)

    # Build the graph
    workflow.set_entry_point("retrieve_chunks_context_per_question")
    workflow.add_edge("retrieve_chunks_context_per_question", "keep_only_relevant_content")

    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        nodes.is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content"
        },
    )
    
    return workflow.compile()

def create_qualitative_retrieval_chapter_summaries_workflow_app():
    """Builds the sub-graph for retrieving and filtering chapter summaries."""
    workflow = StateGraph(state.QualitativeRetrievalGraphState)

    workflow.add_node("retrieve_summaries_context_per_question", nodes.retrieve_summaries_context_per_question)
    workflow.add_node("keep_only_relevant_content", nodes.keep_only_relevant_content)

    workflow.set_entry_point("retrieve_summaries_context_per_question")
    workflow.add_edge("retrieve_summaries_context_per_question", "keep_only_relevant_content")
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        nodes.is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content"
        },
    )
    
    return workflow.compile()

def create_qualitative_book_quotes_retrieval_workflow_app():
    """Builds the sub-graph for retrieving and filtering book quotes."""
    workflow = StateGraph(state.QualitativeRetrievalGraphState)

    workflow.add_node("retrieve_book_quotes_context_per_question", nodes.retrieve_book_quotes_context_per_question)
    workflow.add_node("keep_only_relevant_content", nodes.keep_only_relevant_content)

    workflow.set_entry_point("retrieve_book_quotes_context_per_question")
    workflow.add_edge("retrieve_book_quotes_context_per_question", "keep_only_relevant_content")
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        nodes.is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content"
        },
    )

    return workflow.compile()

# --- Qualitative Answer Workflow ---

def create_qualitative_answer_workflow_app():
    """Builds the sub-graph for generating a grounded answer from context."""
    workflow = StateGraph(state.QualitativeAnswerGraphState)

    workflow.add_node("answer_question_from_context", nodes.answer_question_from_context)

    workflow.set_entry_point("answer_question_from_context")
    workflow.add_conditional_edges(
        "answer_question_from_context",
        nodes.is_answer_grounded_on_context,
        {
            "hallucination": "answer_question_from_context",
            "grounded on context": END
        }
    )

    return workflow.compile()

# --- Compile and export the apps for easy import ---
# This makes them singletons, so they are compiled only once when the module is imported.
# 我们在文件底部直接调用了这些创建函数，并将编译好的 app 实例导出。
# 这样做的好处是，其他模块（比如 nodes.py）可以直接导入这些已经编译好的、可立即使用的子图，避免了每次调用节点时都重新编译，提高了效率。

# New adaptive sub-graphs
factual_retrieval_workflow_app = create_factual_retrieval_workflow_app()
analytical_sub_graph_workflow_app = create_analytical_sub_graph_workflow_app()
tool_use_sub_graph_workflow_app = create_tool_use_sub_graph_workflow_app()

# Original workflows (keeping for backward compatibility)
qualitative_chunks_retrieval_workflow_app = create_qualitative_retrieval_book_chunks_workflow_app()
qualitative_summaries_retrieval_workflow_app = create_qualitative_retrieval_chapter_summaries_workflow_app()
qualitative_book_quotes_retrieval_workflow_app = create_qualitative_book_quotes_retrieval_workflow_app()
qualitative_answer_workflow_app = create_qualitative_answer_workflow_app()