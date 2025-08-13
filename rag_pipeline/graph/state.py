# 目标: 将所有用于 LangGraph 的 TypedDict 状态定义集中到一个文件中。
# 原因: Graph 的状态是数据在图中流动的“管道”。集中定义它们可以让我们清晰地了解整个系统的数据流结构。
# 迁移 TypedDict 类:找到 functions_for_pipeline.py 中所有的 TypedDict 子类，剪切它们的定义
# 为了清晰起见，我为 PlanExecute 中的 mapping 和 plan 字段添加了更具体的类型提示，并按逻辑对字段进行了分组。
# src/rag_pipeline/graph/state.py
from typing import List, TypedDict, Dict

class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of our qualitative retrieval sub-graph.
    """
    question: str
    context: str
    relevant_context: str

class QualitativeAnswerGraphState(TypedDict):
    """
    Represents the state of our qualitative answer sub-graph.
    """
    question: str
    context: str
    answer: str

class PlanExecute(TypedDict):
    """
    Represents the state of the main agent graph.
    """
    # Original input
    question: str

    # Planner state
    anonymized_question: str
    mapping: Dict
    plan: List[str]
    past_steps: List[str]
    
    # Execution state
    curr_state: str # To track current node for debugging
    query_to_retrieve_or_answer: str
    curr_context: str
    aggregated_context: str
    tool: str
    
    # Final output
    response: str