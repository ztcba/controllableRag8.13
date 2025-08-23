from typing import List, TypedDict, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

# --- Sub-Graph State Definitions ---

class FactualSubGraphState(TypedDict):
    """
    State for the factual retrieval and re-ranking sub-graph.
    Manages the process from query enhancement to final answer generation for this path.
    """
    question: str
    """The original or sub-query question to be answered."""
    
    enhanced_question: Optional[str]
    """An enhanced version of the question, optimized for retrieval."""

    documents: List[Document]
    """List of retrieved documents after re-ranking and filtering."""

    graded_documents: Optional[List[Tuple[Document, float]]]
    """List of all retrieved documents with their relevance scores for inspection."""

    context: str
    """The consolidated context string, synthesized from 'documents'."""

    generation: str
    """The final answer generated for this path."""


class AnalyticalSubGraphState(TypedDict):
    """
    State for the analytical sub-graph using sub-query decomposition.
    """
    question: str
    """The original complex question."""

    sub_queries: List[str]
    """The list of decomposed sub-questions."""
    
    sub_query_results: Dict[str, str]
    """A dictionary holding the answer for each sub-question."""

    generation: str
    """The final synthesized answer based on sub-query results."""


class ContextualSubGraphState(TypedDict):
    """
    State for the contextual sub-graph that incorporates user profile.
    (Placeholder for future implementation)
    """
    question: str
    user_profile: Dict[str, Any]
    contextualized_question: str
    documents: List[Document]
    generation: str


class ToolUseSubGraphState(TypedDict):
    """
    State for the sub-graph that handles external tool calls.
    Manages the process from tool decision to final answer generation.
    """
    question: str
    """The original question requiring external tool use."""
    
    needs_tool: bool
    """Whether external tools are needed to answer the question."""
    
    search_query: str
    """The optimized search query for external search engines."""
    
    search_results: List[Dict[str, str]]
    """List of search results with title, snippet, and url."""
    
    generation: str
    """The final answer generated based on search results."""


# --- Main Graph State Definition ---

class MainGraphState(TypedDict):
    """
    Represents the top-level state of the main agent graph.
    It holds only the essential information shared across all sub-graphs and for routing.
    """
    # --- Inputs & Core Data ---
    question: str
    """The original, unmodified user question."""

    user_profile: Optional[Dict[str, Any]]
    """Extracted user background, passed down to relevant sub-graphs."""

    # --- Routing & Control ---
    query_type: str
    """The high-level classification of the query, used for routing to the correct sub-graph."""

    # --- Final Output ---
    final_answer: str
    """The final, generated response to be presented to the user."""

    # --- Global Tracking ---
    error: Optional[str]
    """For logging critical errors that halt the entire process."""

    history: List[str]
    """A high-level log of which sub-graphs or major nodes were executed."""
