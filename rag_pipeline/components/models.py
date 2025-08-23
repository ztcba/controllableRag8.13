# 逐一找到 functions_for_pipeline.py 中所有的 pydantic.BaseModel 子类，
# 剪切 它们的定义，然后 粘贴 到 src/rag_pipeline/components/models.py 文件中
# rag_pipeline/components/models.py
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import List, Dict

class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(description="The relevant content from the retrieved documents that is relevant to the query.")

class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")

class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")

class is_grounded_on_facts(BaseModel):
    """
    Output schema for the rewritten question.
    """
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(description="binary result of whether the question can be fully answered or not")
    explanation: str = Field(description="An explanation of why the question can be fully answered or not.")

class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(description="Whether the distilled content is grounded on the original context.")
    explanation: str = Field(description="An explanation of why the distilled content is or is not grounded on the original context.")

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class TaskHandlerOutput(BaseModel):
    """Output schema for the task handler."""
    query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
    curr_context: str = Field(description="The context to be based on in order to answer the query.")
    tool: str = Field(description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context.")

class AnonymizeQuestion(BaseModel):
    """Anonymized question and mapping."""
    anonymized_question : str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.") # 保持 dict 不变
    explanation: str = Field(description="Explanation of the action.")

class DeAnonymizePlan(BaseModel):
    """Possible results of the action."""
    plan: List[str] = Field(description="Plan to follow in future. with all the variables replaced with the mapped words.") # 保持 List 不变

class CanBeAnsweredAlready(BaseModel):
    """Possible results of the action."""
    can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")


# --- 问题分类 ---
class QueryClassification(BaseModel):
    """
    Output schema for the query classification chain.
    Ensures the LLM's output is a single, valid category name.
    """
    query_type: str = Field(
        description="The classification of the user's query. Must be one of: 'simple_retrieval', 'comparison', 'logical_reasoning', 'scenario_synthesis', 'tool_use', 'data_visualization'."
    )
# --- 问题分类 ---

# --- 事实型查询的结构化输出 ---
class EnhancedQuery(BaseModel):
    """
    Output schema for the query enhancement chain.
    """
    enhanced_question: str = Field(
        description="An improved, more specific version of the original query, optimized for vector store retrieval."
    )

class DocumentRelevanceScore(BaseModel):
    """
    Output schema for the document re-ranking chain.
    Provides a relevance score for a single document against a query.
    """
    score: float = Field(
        description="The relevance score of the document to the query, on a scale of 1 to 10."
    )
    explanation: str = Field(
        description="A brief explanation of why the document received this score."
    )

class GeneratedAnswer(BaseModel):
    """
    Output schema for the answer generation chain.
    """
    answer_based_on_content: str = Field(
        description="The final answer generated based on the provided context and question."
    )
# --- 事实型查询的结构化输出 ---

# --- 分析推理子图的结构化输出 ---
class SubQueries(BaseModel):
    """
    Output schema for the sub-query generation chain.
    Used in the analytical sub-graph to decompose complex questions.
    """
    sub_queries: List[str] = Field(
        description="A list of specific sub-questions that together can answer the original complex question. Each sub-query should be factual and independently answerable."
    )

class AnalyticalSynthesis(BaseModel):
    """
    Output schema for the analytical synthesis chain.
    Used to combine answers from multiple sub-queries into a coherent final answer.
    """
    answer_based_on_content: str = Field(
        description="A comprehensive, well-structured answer that synthesizes information from all sub-query results to address the original analytical question."
    )
# --- 分析推理子图的结构化输出 ---

# --- 工具使用子图的结构化输出 ---
class ToolDecision(BaseModel):
    """
    Output schema for tool decision making.
    """
    needs_external_tool: bool = Field(
        description="Whether the question requires external tools (like web search) to answer adequately."
    )
    reasoning: str = Field(
        description="Explanation of why external tools are or are not needed."
    )

class SearchQuery(BaseModel):
    """
    Output schema for search query generation.
    """
    search_query: str = Field(
        description="An optimized search query for finding relevant external information."
    )

class ToolUseAnswer(BaseModel):
    """
    Output schema for generating answers based on external tool results.
    """
    answer_based_on_content: str = Field(
        description="A comprehensive answer that combines information from search results with knowledge about the China Postdoctoral Science Foundation."
    )
# --- 工具使用子图的结构化输出 ---