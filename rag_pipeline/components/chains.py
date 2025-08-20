# 创建所有原子的、可执行的"链"（Runnable 对象），这些链将LLM、提示和输出解析器组合在一起。
# 这是最重要的部分。我们将 create_..._chain 函数移动到这里，它们是执行单个、具体任务（如"规划"、"回答问题"）的基本单元。
# 这使得这些单元可以在不同的工作流或 Agent 中被复用。
# 所有名为 create_..._chain 的函数
# rag_pipeline/components/chains.py

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import from our new modules
from rag_pipeline.components.llms import get_chat_model, get_planner_model
from rag_pipeline.components import models
from rag_pipeline.components import prompts

# Each function now creates a chain using imported components.

def create_keep_only_relevant_content_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.keep_only_relevant_content_prompt_template,
        input_variables=["query", "retrieved_documents"],
    )
    return prompt | llm.with_structured_output(models.KeepRelevantContent, method="function_calling")

def create_question_answer_from_context_cot_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )
    return prompt | llm.with_structured_output(models.QuestionAnswerFromContext, method="function_calling")

def create_is_relevant_content_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.is_relevant_content_prompt_template,
        input_variables=["query", "context"],
    )
    return prompt | llm.with_structured_output(models.Relevance, method="function_calling")

def create_is_grounded_on_facts_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )
    return prompt | llm.with_structured_output(models.is_grounded_on_facts, method="function_calling")

def create_can_be_answered_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.can_be_answered_prompt_template,
        input_variables=["question","context"],
    )
    return prompt | llm.with_structured_output(models.QuestionAnswer, method="function_calling")

def create_is_distilled_content_grounded_on_content_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.is_distilled_content_grounded_on_content_prompt_template,
        input_variables=["distilled_content", "original_context"],
    )
    return prompt | llm.with_structured_output(models.IsDistilledContentGroundedOnContent, method="function_calling")

def create_plan_chain():
    planner_llm = get_planner_model()
    prompt = PromptTemplate(
        template=prompts.planner_prompt, # Note: using the extracted prompt
        input_variables=["question"], 
    )
    # 使用 function_calling 方法而不是 json_schema 方法来避免 Pydantic v1 警告
    return prompt | planner_llm.with_structured_output(models.Plan, method="function_calling")

def create_break_down_plan_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.break_down_plan_prompt_template,
        input_variables=["plan"],
    )
    return prompt | llm.with_structured_output(models.Plan, method="function_calling")

def create_replanner_chain():
    replanner_llm = get_planner_model()
    prompt = PromptTemplate(
        template=prompts.replanner_prompt_template,
        input_variables=["question", "plan", "past_steps", "aggregated_context"],
    )
    return prompt | replanner_llm.with_structured_output(models.Plan, method="function_calling")

def create_task_handler_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.tasks_handler_prompt_template,
        input_variables=["curr_task", "aggregated_context", "last_tool", "past_steps", "question"],
    )
    return prompt | llm.with_structured_output(models.TaskHandlerOutput, method="function_calling")

def create_anonymize_question_chain():
    llm = get_chat_model()
    parser = JsonOutputParser(pydantic_object=models.AnonymizeQuestion)
    prompt = PromptTemplate(
        template=prompts.anonymize_question_prompt_template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def create_deanonymize_plan_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.de_anonymize_plan_prompt_template,
        input_variables=["plan", "mapping"],
    )
    return prompt | llm.with_structured_output(models.DeAnonymizePlan, method="function_calling")

def create_can_be_answered_already_chain():
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.can_be_answered_already_prompt_template,
        input_variables=["question","context"],
    )
    return prompt | llm.with_structured_output(models.CanBeAnsweredAlready, method="function_calling")

def create_classification_chain():
    """
    Factory function to create the query classification chain.

    This function encapsulates the logic for creating a chain that takes a user's
    question and classifies it into a predefined category using a structured output
    format.

    Returns:
        Runnable: A LangChain runnable object ready to be invoked.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.query_classifier_prompt_template,
        input_variables=["question"],
    )
    # Ensure structured output using the Pydantic model
    return prompt | llm.with_structured_output(
        models.QueryClassification, method="function_calling"
    )

def create_query_enhancement_chain():
    """
    Factory function for the query enhancement chain.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.query_enhancement_prompt_template,
        input_variables=["question"],
    )
    return prompt | llm.with_structured_output(
        models.EnhancedQuery, method="function_calling"
    )

def create_reranking_chain():
    """
    Factory function for the document re-ranking chain.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.document_reranking_prompt_template,
        input_variables=["question", "document"],
    )
    return prompt | llm.with_structured_output(
        models.DocumentRelevanceScore, method="function_calling"
    )

def create_generation_chain():
    """
    Factory function for the answer generation chain.
    Used in factual sub-graph to generate final answers.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.generation_prompt_template,
        input_variables=["question", "context"],
    )
    return prompt | llm.with_structured_output(
        models.GeneratedAnswer, method="function_calling"
    )

def create_sub_query_generation_chain():
    """
    Factory function for the sub-query generation chain.
    Used in analytical sub-graph to decompose complex questions.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.sub_query_generation_prompt_template,
        input_variables=["question"],
    )
    return prompt | llm.with_structured_output(
        models.SubQueries, method="function_calling"
    )

def create_analytical_synthesis_chain():
    """
    Factory function for the analytical synthesis chain.
    Used in analytical sub-graph to synthesize answers from sub-queries.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.analytical_synthesis_prompt_template,
        input_variables=["question", "sub_query_results"],
    )
    return prompt | llm.with_structured_output(
        models.AnalyticalSynthesis, method="function_calling"
    )

def create_tool_decision_chain():
    """
    Factory function for the tool decision chain.
    Used to determine if external tools are needed to answer a question.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.tool_decision_prompt_template,
        input_variables=["question"],
    )
    return prompt | llm.with_structured_output(
        models.ToolDecision, method="function_calling"
    )

def create_search_query_generation_chain():
    """
    Factory function for the search query generation chain.
    Used to create optimized search queries for external tools.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.search_query_generation_prompt_template,
        input_variables=["question"],
    )
    return prompt | llm.with_structured_output(
        models.SearchQuery, method="function_calling"
    )

def create_tool_use_answer_generation_chain():
    """
    Factory function for the tool use answer generation chain.
    Used to generate answers based on external search results.
    """
    llm = get_chat_model()
    prompt = PromptTemplate(
        template=prompts.tool_use_answer_generation_prompt_template,
        input_variables=["question", "search_results"],
    )
    return prompt | llm.with_structured_output(
        models.ToolUseAnswer, method="function_calling"
    )