# functions_for_pipeline.py
# 调整过taskhandler的提示词的版本，在一些任务上表现不足
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# Import LLM factory functions
from rag_pipeline.components.llms import get_chat_model, get_elite_model    # 【核心简化】

from retriever_factory import create_hybrid_retriever
# from langchain.retrievers.document_compressors import CrossEncoderRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from pprint import pprint
from rag_pipeline.components.rerankers import get_reranker_model
from graphstate import PlanExecute
from web_search import run_web_search
# embedding_model = get_embedding_model()
# retriever = create_hybrid_retriever()
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pprint import pprint
import os
# 画图用
import base64
import io
import matplotlib.pyplot as plt

from typing import List
from typing import Literal



### Helper functions for the notebook
from helper_functions import escape_quotes, text_wrap


# ========================================================================================
# 图状态
# ========================================================================================





# ========================================================================================
# 接到用户问题后计划
# ========================================================================================

class Plan(BaseModel):
        """Plan to follow in future"""

        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )



def create_plan_chain():
    

    planner_prompt =""" For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    """
    
    
#     """ # 角色：你是一位顶级的科研项目信息分析专家。

# # 任务：
# 为给定的用户问题 {question}，制定一个逻辑严谨、步骤清晰的**信息检索计划**。你的目标是将一个复杂的问题分解成一系列独立的、可执行的**信息获取子任务**。

# # 指示与准则：
# 1.  **识别核心实体**：精确识别问题中的所有关键信息点，例如：年份（如“2025年”）、机构名称（如“同济大学”）、文件名（如“项目指南”）、申请人身份（如“在职博士”、“高级职称”）、项目类型（如“青年科学基金”、“卓越研究群体”）等。
# 2.  **分解检索任务**：基于识别出的实体和问题逻辑（如“对比”、“申请条件”、“区别”、“是否可以”），将原问题拆解成多个独立的检索步骤。
# 3.  **明确信息目标**：每个步骤都必须清晰地说明需要“检索”或“查找”的**具体信息内容**。
#     *   **反例 (不要这样做)**: “研究申请条件。”
#     *   **正例 (请这样做)**: “查找2025年国家自然科学基金青年科学基金项目对申请人身份（如在职博士研究生）的具体要求。”
# 4.  **体现逻辑依赖**：如果后续步骤需要利用前面步骤的检索结果，请在计划中清晰地体现出这种顺序。
# 5.  **绝对专注检索**：你的输出**只能包含**信息检索的步骤。**严禁包含**任何关于“如何整合信息”、“进行逻辑判断”、“比较差异”或“生成最终答案”的步骤。整个计划的终点是“获取了所有必要的信息片段”，后续的加工由其他节点完成。

# # 输出格式：
# 请以有序列表的形式输出检索计划。

# ---
# # 示例：
# ## 用户问题：
# 对比2024年和2025年的国家自然科学基金项目指南，关于经费包干制的项目类型，2025年新增了哪两类？
# ## 你的输出：
# 1. 查找《2024年国家自然科学基金项目指南》中，关于“经费包干制”所覆盖的项目类型列表。
# 2. 查找《2025年国家自然科学基金项目指南》中，关于“经费包干制”所覆盖的项目类型列表。

#     """

    planner_prompt = PromptTemplate(
        template=planner_prompt,
        input_variables=["question"], 
        )

    planner_llm = get_chat_model()

    planner = planner_prompt | planner_llm.with_structured_output(Plan)
    return planner

# Initialize the planner chain
planner = create_plan_chain()

def plan_step(state: PlanExecute):
    """
    Plans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "planner"
    print("Planning step")
    pprint("--------------------")
    plan = planner.invoke({"question": state['question']})
    state["plan"] = plan.steps
    print(f'plan: {state["plan"]}')
    return state

# ========================================================================================
# 分解plan
# ========================================================================================



def create_break_down_plan_chain():

    break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
    you need to go through the plan refine it according to this:
    1. every step has to be able to be executed by either:
        i. retrieving relevant information from a vector store of book chunks
        ii. answering a question from a given context.
    2. every step should contain all the information needed to execute it.

    output the refined plan
    """

    break_down_plan_prompt = PromptTemplate(
        template=break_down_plan_prompt_template,
        input_variables=["plan"],
    )

    break_down_plan_llm = get_elite_model()

    break_down_plan_chain = break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)

    return break_down_plan_chain

break_down_plan_chain = create_break_down_plan_chain()

def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    refined_plan = break_down_plan_chain.invoke({"plan": state["plan"]}) # state["plan"]会填充提示词的{plan}
    state["plan"] = refined_plan.steps
    print(f'break_down_plan_step为: {state["plan"]}')
    return state

# ========================================================================================
# task handler
# ========================================================================================




def create_task_handler_chain():

    tasks_handler_prompt_template ="""You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
    You have the following tools at your disposal:
    Tool A: a tool that retrieves relevant information from a vector store of the document library(e.g., "基金项目指南","申请注意事项") based on a given query.
    - use Tool A when you think the current task should search for information in the document library.
    Tool B: a tool that answers a question from a given context.
    - use Tool B ONLY when the current task can be answered by the aggregated context {aggregated_context}

    You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
    You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
    if you decide to use Tools A, output the query to be used for the tool and also output the relevant tool.
    if you decide to use Tool B, output the question to be used for the tool, the context, and also that the tool to be used is Tool B.

    """
    class TaskHandlerOutput(BaseModel):
        """Output schema for the task handler."""
        query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
        curr_context: str = Field(description="The context to be based on in order to answer the query.")
        tool: str = Field(description="The tool to be used should be either retrieve_chunks or answer_from_context.")


    task_handler_prompt = PromptTemplate(
        template=tasks_handler_prompt_template,
        input_variables=["curr_task", "aggregated_context", "last_tool", "past_steps", "question"]
    )

    task_handler_llm = get_chat_model()
    task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput)
    return task_handler_chain

task_handler_chain = create_task_handler_chain()


def run_task_handler_chain(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    state["curr_state"] = "task_handler"
    print("the current plan is:")

    pprint("--------------------") 

    if not state['past_steps']:
        state["past_steps"] = []

    curr_task = state["plan"][0]
    print("当前任务是:", curr_task)
    inputs = {"curr_task": curr_task,
               "aggregated_context": state["aggregated_context"],
                "last_tool": state["tool"],
                "past_steps": state["past_steps"],
                "question": state["question"]}
    
    output = task_handler_chain.invoke(inputs)
  
    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    if output.tool == "retrieve_chunks":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="retrieve_chunks"
        print("decided to use retrieve tool with query:", output.query)
       
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="answer"
        print("decided to use answer from context tool with question:", output.query)



    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state  

# ========================================================================================
# 根据工具导流
# ========================================================================================

def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    state["curr_state"] = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")

# ========================================================================================
# 重新计划步骤
# ========================================================================================


def create_replanner_chain():

    replanner_prompt_template =""" 你的核心目标是回答用户的问题：{question}

你已经执行了以下步骤：
{past_steps}

根据已经检索到的信息，你现在掌握了如下上下文：
{aggregated_context}

---
现在，请你扮演决策者的角色，严格按照以下逻辑进行判断和规划：

1.  **首先判断**：基于当前的“聚合上下文”，是否已经包含了足够的信息来完整、准确地回答用户的“原始问题”？

2.  **然后规划**：
    -   **如果信息足够**：那么你的新计划应该是唯一的、明确的“回答”步骤。请只输出一个步骤，例如：["根据已聚合的上下文，综合整理并生成最终答案。"]
    -   **如果信息不足**：请分析还缺少哪些关键信息，并制定出下一步**简洁有效**的检索计划。**不要**猜测具体的表名（如表1、表2），而是应该围绕核心概念进行规划。例如：["检索关于面上资助评审标准的具体评分细则。"]

你之前的计划是：{plan}
请根据以上决策，更新你的计划。只输出下一步需要执行的步骤。

    """

    replanner_prompt = PromptTemplate(
        template=replanner_prompt_template,
        input_variables=["question", "plan", "past_steps", "aggregated_context"],
        # partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
    )

    replanner_llm = get_elite_model()

    replanner = replanner_prompt | replanner_llm.with_structured_output(Plan)
    return replanner

replanner = create_replanner_chain()

def replan_step(state: PlanExecute):
    """
    Replans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "replan"
    print("Replanning step")
    pprint("--------------------")
    inputs = {"question": state["question"], "plan": state["plan"], "past_steps": state["past_steps"], "aggregated_context": state["aggregated_context"]}
    plan = replanner.invoke(inputs)
    state["plan"] = plan.steps
    return state

# ========================================================================================
# 检索、回答、上网查询、sqltotxt等工具
# ========================================================================================


# 检索工具
from typing import TypedDict, List
from pprint import pprint



def run_qualitative_chunks_retrieval_workflow(state: PlanExecute):
    """
    (已集成) 运行定性检索工作流，采用“小块检索，大块重排”策略，
    并使用自定义的 AiHubMixReranker。
    
    Args:
        state: The current state of the plan execution. 
               (使用 dict 以便通用，实际应为 PlanExecute 类型)
        
    Returns:
        The state with the updated aggregated context.
    """
    
    # 维持状态更新逻辑 1: 更新当前状态
    state["curr_state"] = "retrieve_and_rerank"

    
    question = state["query_to_retrieve_or_answer"]
    
    # --- 流程开始 ---
    
    # 步骤一：检索 (Retrieve)
    # 使用现有的混合检索器，它现在会在“子块”上进行检索
    retriever = create_hybrid_retriever(bm25_k=6, vector_k=6, hybrid_weights=[0.7, 0.3]) # 召回更多候选以供精排
    print("用户的原始问题:",state["question"])
    print(f"1. 🔍 专门用于检索的query: '{question}'")
    child_chunks = retriever.invoke(question)
    print(f"   ✅ Retrieved {len(child_chunks)} child chunks.")
    print(f"   🔍 检索到的子块内容: {[chunk.page_content for chunk in child_chunks]}")

    # 步骤二：扩展 (Expand)
    # 从子块的元数据中提取出父块，并去重
    parent_chunks_map = {}
    for chunk in child_chunks:
        # 假设父块内容存储在 'parent_content' metadata 字段中
        parent_content = chunk.metadata.get("parent_content")
        if parent_content:
            # 使用内容作为key，自动去重
            parent_chunks_map[parent_content] = Document(page_content=parent_content, metadata=chunk.metadata)
            
    unique_parent_chunks = list(parent_chunks_map.values())
    print(f"检索到的父块内容: {[chunk.page_content for chunk in unique_parent_chunks]}")


    if not unique_parent_chunks:
        # 如果没有检索到任何内容，直接返回
        print("   ⚠️ No unique parent chunks found. Skipping reranking.")
        state["aggregated_context"] = ""
        pprint("--------------------")
        return state

    # ========================= 【核心替换部分】 =========================
    
    # 步骤三：重排 (Rerank)
    # 使用新的 AiHubMixReranker 对父块进行重排序

    
    # 1. 使用您的工厂函数实例化重排器，并设置返回 top 3 的文档
    reranker = get_reranker_model(top_n=5) 
    
    # 2. 需要用transform_documents方法。

    reranked_docs = reranker.transform_documents(
        documents=unique_parent_chunks,
        query=question
    )
    print(f"   ✅ Reranked and selected top {len(reranked_docs)} parent chunks via API.")
    print(f"   重排后的父块内容: {[doc.page_content for doc in reranked_docs]}")
    
    # ====================================================================

    # 步骤四：聚合 (Aggregate)
    # 将重排后的高质量父块内容聚合为最终上下文
    relevant_context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    
    # --- 流程结束 ---

    # 维持状态更新逻辑 2: 更新聚合上下文
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
        
    state["aggregated_context"] += relevant_context
    
    print("4. ✨ Final context generated and aggregated.")
    pprint("--------------------")
    return state

# ========================================================================================
# 回答flow
# ========================================================================================


def create_question_answer_from_context_cot_chain():
    class QuestionAnswerFromContext(BaseModel):
        answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")

    question_answer_from_context_llm = get_chat_model()


    question_answer_cot_prompt_template ="""
    You are an expert AI assistant. Your mission is to provide a final, high-quality answer to the user's question based *exclusively* on the provided context.

    **Instructions:**

    1.  Identify the User's Persona: Pay close attention to the user's specific role, identity, or situation described in the question (e.g., "an in-service doctoral candidate" ).
    1.  **Review the Context**: Carefully read the entire provided context to understand all available information.
    2.  **Identify Relevant Facts**: Pinpoint the specific sentences or data points within the context that directly address the user's question.
    3.  **Synthesize the Answer**: Combine the relevant facts into a clear, concise, and comprehensive answer.
    4.  **Strictly Adhere to Context**: DO NOT use any information outside of the provided text. If the context does not contain the answer, you must state that the information is not available.

    ---
    **Provided Context:**
    {context}
    ---
    **User's Question:**
    {question}
    ---

    Now, generate the final answer based on these instructions.
    """

    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )
    question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain

question_answer_from_context_cot_chain = create_question_answer_from_context_cot_chain()

def answer_question_from_context(state):
    """
    Answers a question from a given context.

    Args:
        question: The query question.
        context: The context to answer the question from.
        chain: The LLMChain instance.

    Returns:
        The answer to the question from the context.
    """
    question = state["question"]
    context = state["aggregated_context"] if "aggregated_context" in state else state["context"]

    input_data = {
    "question": question,
    "context": context
}
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f'answer before checking hallucination: {answer}')
    return {"answer": answer, "context": context, "question": question}

def create_is_grounded_on_facts_chain():
    class is_grounded_on_facts(BaseModel):
        """
        Output schema for the rewritten question.
        """
        grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    is_grounded_on_facts_llm = get_chat_model()
    is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
    you don't mind if it doesn't make sense, as long as it is grounded in the context.
    output a json containing the answer to the question, and appart from the json format don't output any additional text.

    """
    is_grounded_on_facts_prompt = PromptTemplate(
        template=is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )
    is_grounded_on_facts_chain = is_grounded_on_facts_prompt | is_grounded_on_facts_llm.with_structured_output(is_grounded_on_facts)
    return is_grounded_on_facts_chain
is_grounded_on_facts_chain = create_is_grounded_on_facts_chain()
def is_answer_grounded_on_context(state):
    """Determines if the answer to the question is grounded in the facts.
    
    Args:
        state: A dictionary containing the context and answer.
    """
    print("Checking if the answer is grounded in the facts...")
    context = state["context"]
    answer = state["answer"]
    
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"

def run_qualtative_answer_workflow(state):
    """
    Run the qualitative answer workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "answer"
    print("Running the qualitative answer workflow...")
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["answer"]
    return state

def create_qualitative_answer_workflow_app():
    class QualitativeAnswerGraphState(TypedDict):
        """
        Represents the state of our graph.

        """

        question: str
        context: str
        answer: str

    qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

    # Define the nodes

    qualitative_answer_workflow.add_node("answer_question_from_context",answer_question_from_context)

    # Build the graph
    qualitative_answer_workflow.set_entry_point("answer_question_from_context")

    qualitative_answer_workflow.add_conditional_edges(
    "answer_question_from_context",is_answer_grounded_on_context ,{"hallucination":"answer_question_from_context", "grounded on context":END}

    )

    qualitative_answer_workflow_app = qualitative_answer_workflow.compile()
    return qualitative_answer_workflow_app

qualitative_answer_workflow_app = create_qualitative_answer_workflow_app()
# ========================================================================================
# 最终解答
# ========================================================================================



def run_qualtative_answer_workflow_for_final_answer(state):
    """
    Run the qualitative answer workflow for the final answer.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated response.
    """
    state["curr_state"] = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, value in output.items():
            pass  
        pprint("--------------------")
    state["response"] = value
    return state

# ========================================================================================
# 能不能回答
# ========================================================================================
def create_can_be_answered_already_chain():
    class CanBeAnsweredAlready(BaseModel):
        """Possible results of the action."""
        can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")

    can_be_answered_already_prompt_template = """You receive a query: {question} and a context: {context}.
    You need to determine if the question can be answered relying only the given context.
    The only infomation you have and can rely on is the context you received. 
    you have no prior knowledge of the question or the context.
    if you think the question can be answered based on the context, output 'true', otherwise output 'false'.
    """

    can_be_answered_already_prompt = PromptTemplate(
        template=can_be_answered_already_prompt_template,
        input_variables=["question","context"],
    )

    can_be_answered_already_llm = get_chat_model()
    can_be_answered_already_chain = can_be_answered_already_prompt | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)
    return can_be_answered_already_chain
can_be_answered_already_chain = create_can_be_answered_already_chain()
def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.
    Args:
        state: The current state of the plan execution.
    Returns:
        whether the original question can be answered or not.
    """
    state["curr_state"] = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    output = can_be_answered_already_chain.invoke(inputs)
    if output.can_be_answered == True:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        print("the aggregated context is:")
        print(text_wrap(state["aggregated_context"]))
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"
    

# ========================================================================================
# 路由节点
# ========================================================================================


# --- Pydantic 模型定义 ---

class RouteQuery(BaseModel):
    """用于路由用户问题的Pydantic模型。"""
    rout_choice: Literal["rout_is_retrieve", "rout_is_websearch", "rout_is_draw"] = Field(
        ...,
        description="根据用户问题的类型选择最合适的路由路径。"
    )

# --- 提示词定义 ---

router_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """你是一个善于将用户问题路由到正确工具的专家。根据用户的问题，决定下一步应该采取的行动。

你有三个选项：

1.  `rout_is_retrieve`: 如果问题可以从一个特定的、已有的知识库或文档中查找答案，请选择此项。
    例如：“根据《2025中国博士后科学基金资助指南》，面上资助中，自然科学领域的资助标准是多少？” 
    这类问题通常包含对特定文档或内部知识的引用。这类问题包含“基金”、“指南”等关键词。

2.  `rout_is_websearch`: 从某个网站上获得信息，请选择此项。除此之外，一定不要选择此项
    例如：“《Nature》期刊最新一期的论文标题列表？” 
3.  `rout_is_draw`: 如果问题明确要求生成图表、绘图或任何形式的可视化，请选择此项。
    例如：“2025中国博士后科学基金‘面上资助’项目专家评审时各项指标的权重分布如何？请画出饼图。”
    这类问题包含“画图”、“图表”、“可视化”、“饼图”、“柱状图”等关键词。
"""),
        ("human", "根据以下用户问题进行路由决策：\n\n问题: {question}")
    ]
)

# --- 路由节点函数 ---

def router(state: PlanExecute) -> PlanExecute:
    """
    分析用户的 question，并决定下一步的路由。
    更新 state 中的 'rout' 字段。
    """
    print("---(节点: router) 开始路由决策---")
    
    question = state["question"]
    if not question:
        raise ValueError("状态(state)中的'question'字段不能为空")

    print(f"---(节点: router) 接收到问题: {question}---")

    # 获取LLM模型
    llm = get_chat_model()
    
    # 将LLM与Pydantic模型绑定，以获得结构化输出
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # 创建完整的处理链
    chain = router_prompt_template | structured_llm
    
    # 执行调用链
    result = chain.invoke({"question": question})
    
    # 获取决策结果并更新状态
    decision = result.rout_choice
    state["rout"] = decision
    
    print(f"---(节点: router) 路由决策完成: {decision}---")
    
    return state

def rout(state: PlanExecute):
    return state['rout']

# ========================================================================================
# 联网搜索 --已经在头部引入
# ========================================================================================


# ========================================================================================
# 画图节点
# ========================================================================================

# 定义单个数据项的结构
# 定义单个数据项的结构 (用于 LLM 结构化输出)
class ChartDataItem(BaseModel):
    label: str = Field(description="图表的标签名，例如'真实性'")
    value: float = Field(description="该标签对应的数值，例如 30")

# 定义期望的最终输出结构，它是一个包含多个数据项的列表
class ChartData(BaseModel):
    data: List[ChartDataItem] = Field(description="一个包含所有图表数据项的列表")


def draw_chart(state: PlanExecute) -> PlanExecute:
    """
    一个 LangGraph 节点，功能如下：
    1. 使用 LLM 从 state['aggregated_context'] 的自然语言文本中提取结构化数据。
    2. 使用提取的数据通过 Matplotlib 生成图表。
    3. 将生成的图表编码为 Base64 字符串并存入 state['chart_image']。
    """
    print("---(节点: draw) 开始执行绘图---")
    question = state.get("question", "")
    final_answer_text = state.get("response", "")


    if not final_answer_text:
        print("---(节点: draw) 最终答案文本为空，无法绘图。---")
        # 此时 response 字段已经有内容了，我们不应该覆盖它
        # 只需要更新 chart_image 字段即可
        state['chart_image'] = "" 
        return state

    # --- 步骤 1: 使用 LLM 提取结构化数据 ---
    try:
        # 建议使用性能更强的模型以保证 JSON 输出的稳定性，例如 gpt-4
        # 请确保您已设置 OPENAI_API_KEY 环境变量
        llm = get_chat_model()
        parser = JsonOutputParser(pydantic_object=ChartData)

        extraction_prompt = ChatPromptTemplate.from_template(
            "你的任务是从给定的上下文中提取用于绘图的数据。\n"
            "严格遵循用户的格式化指令：\n{format_instructions}\n"
            "--- 上下文开始 ---\n{context}\n--- 上下文结束 ---\n"
            "根据以上上下文，提取用于回答用户问题 '{question}' 所需的各项指标及其数值。"
        )

        extraction_chain = extraction_prompt | llm | parser
        
        print("---(节点: draw) 调用 LLM 提取数据... ---")
        extracted_result = extraction_chain.invoke({
            "context": aggregated_context,
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        })
        
        chart_data_list = extracted_result.get('data', [])
        if not chart_data_list:
            raise ValueError("LLM 未能从上下文中提取到任何有效数据。")

        labels = [item['label'] for item in chart_data_list]
        sizes = [item['value'] for item in chart_data_list]
        print(f"---(节点: draw) 成功提取数据: {labels}, {sizes} ---")

    except Exception as e:
        print(f"---(节点: draw) LLM 提取数据失败: {e} ---")
        state['response'] = "抱歉，我无法从提供的文本中准确地解析出绘图所需的数据。"
        state['chart_image'] = ""
        return state

    # --- 步骤 2: 使用 Matplotlib 生成图表 ---
    try:
        print("---(节点: draw) 开始使用 Matplotlib 生成图表... ---")
        # 设置 Matplotlib 以支持中文显示 (确保您的系统已安装 SimHei 字体)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax.axis('equal')  # 确保饼图是正圆形

        # 尝试从问题中生成一个简洁的标题
        title = question.split("？")[0].split("?")[0]
        plt.title(title, fontsize=16, pad=20)
        
        # 将图表保存到内存中的字节流
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # 将图像字节流编码为 Base64 字符串
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        print("---(节点: draw) 图表生成并编码成功。---")
        
        # --- 步骤 3: 更新状态 ---
        state["chart_image"] = image_base64
        state["response"] = "好的，我已经根据检索到的信息为您生成了图表。"

    except Exception as e:
        print(f"---(节点: draw) Matplotlib 生成图表时出错: {e} ---")
        state["chart_image"] = ""
        state["response"] = f"抱歉，在生成图表时遇到了一个内部错误: {e}"
    finally:
        plt.close('all') # 清理所有 plt figure，防止内存泄漏

    return state


# ========================================================================================
# 新增条件边函数，在replanner处
# ========================================================================================
def decide_final_step(state: PlanExecute):
    """
    在生成最终答案后，检查是否还需要绘图。
    """
    if state.get("rout") == "rout_is_draw":
        print("---(决策: decide_final_step) 原始意图是画图，路由到 draw 节点---")
        return "go_to_draw"
    else:
        print("---(决策: decide_final_step) 任务完成，路由到 END---")
        return "end"
# ========================================================================================
# 最终agent
# ========================================================================================

def create_agent():
    
    agent_workflow = StateGraph(PlanExecute)

    agent_workflow.add_node("router", router) 
    agent_workflow.add_node("web_search", run_web_search)
    agent_workflow.add_node("draw", draw_chart)
    # Add the plan node
    agent_workflow.add_node("planner", plan_step)

    # Add the break down plan node

    agent_workflow.add_node("break_down_plan", break_down_plan_step)

    # Add the qualitative chunks retrieval node
    agent_workflow.add_node("retrieve", run_qualitative_chunks_retrieval_workflow)

    # Add the qualitative answer node
    agent_workflow.add_node("answer", run_qualtative_answer_workflow)

    # Add the task handler node
    agent_workflow.add_node("task_handler", run_task_handler_chain)

    # Add a replan node
    agent_workflow.add_node("replan", replan_step)

    # Add answer from context node
    agent_workflow.add_node("get_final_answer", run_qualtative_answer_workflow_for_final_answer)

    # Set the entry point
    agent_workflow.set_entry_point("router")

    # 边的定义
    agent_workflow.add_conditional_edges(
    "router",
    rout,
    {
        "rout_is_retrieve": "planner",     
        "rout_is_websearch": "web_search",  
        "rout_is_draw": "planner"              
    }
    )



    agent_workflow.add_edge("planner", "break_down_plan")

    # From break_down_plan we go to task handler
    agent_workflow.add_edge("break_down_plan", "task_handler")

    # From task handler we go to either retrieve or answer
    agent_workflow.add_conditional_edges("task_handler", retrieve_or_answer, {"chosen_tool_is_retrieve_chunks": "retrieve",  "chosen_tool_is_answer": "answer"})

    # After retrieving we go to replan
    agent_workflow.add_edge("retrieve", "replan")


    # After answering we go to replan
    agent_workflow.add_edge("answer", "replan")

    # After replanning we check if the question can be answered, if yes we go to get_final_answer, if not we go to task_handler
    agent_workflow.add_conditional_edges("replan",can_be_answered, {"can_be_answered_already": "get_final_answer", "cannot_be_answered_yet": "break_down_plan"})

    agent_workflow.add_conditional_edges(
        "get_final_answer",
        decide_final_step, # <--- 使用新的决策函数
        {
            "go_to_draw": "draw", # 如果需要画图，就去 draw 节点
            "end": END            # 否则，直接结束
        }
    )

    # 在 web_search 执行完毕后，直接结束流程
    agent_workflow.add_edge("web_search", END)

    # 在 draw 执行完毕后，直接结束流程
    agent_workflow.add_edge("draw", END)

    plan_and_execute_app = agent_workflow.compile()

    return plan_and_execute_app

