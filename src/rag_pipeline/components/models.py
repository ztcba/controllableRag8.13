# 逐一找到 functions_for_pipeline.py 中所有的 pydantic.BaseModel 子类，
# 剪切 它们的定义，然后 粘贴 到 src/rag_pipeline/components/models.py 文件中
# src/rag_pipeline/components/models.py
from langchain_core.pydantic_v1 import BaseModel, Field
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
    plan: List = Field(description="Plan to follow in future. with all the variables replaced with the mapped words.") # 保持 List 不变

class CanBeAnsweredAlready(BaseModel):
    """Possible results of the action."""
    can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")