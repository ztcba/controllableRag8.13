# 创建可复用子图（如 create_qualitative_retrieval_..._workflow_app）的逻辑
# 迁移到 src/rag_pipeline/graph/workflows.py

# src/rag_pipeline/graph/workflows.py
from langgraph.graph import END, StateGraph

# Import the state and nodes required for these workflows
from src.rag_pipeline.graph import state
from src.rag_pipeline.graph import nodes

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
qualitative_chunks_retrieval_workflow_app = create_qualitative_retrieval_book_chunks_workflow_app()
qualitative_summaries_retrieval_workflow_app = create_qualitative_retrieval_chapter_summaries_workflow_app()
qualitative_book_quotes_retrieval_workflow_app = create_qualitative_book_quotes_retrieval_workflow_app()
qualitative_answer_workflow_app = create_qualitative_answer_workflow_app()