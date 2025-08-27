# functions_for_pipeline.py
from langchain_openai import ChatOpenAI 
# from langchain_groq import ChatGroq
from langchain.vectorstores import  FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# Import LLM factory functions
from rag_pipeline.components.llms import get_chat_model, get_embedding_model    # 【核心简化】

from retriever_factory import create_hybrid_retriever
# from langchain.retrievers.document_compressors import CrossEncoderRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from pprint import pprint
from rag_pipeline.components.rerankers import get_reranker_model
# embedding_model = get_embedding_model()
# retriever = create_hybrid_retriever()
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pprint import pprint
import os
from typing_extensions import TypedDict
from typing import List



### Helper functions for the notebook
from helper_functions import escape_quotes, text_wrap


# ========================================================================================
# 图状态
# ========================================================================================

class PlanExecute(TypedDict):
    curr_state: str # current state of the plan execution
    question: str # 用户提的问题
    query_to_retrieve_or_answer: str 
    plan: List[str] # 分解的计划
    past_steps: List[str] # 过去的步骤
    curr_context: str #
    aggregated_context: str # 聚合的检索结果
    tool: str # 使用何种工具(检索或回答)
    response: str



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

    break_down_plan_llm = get_chat_model()

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
    refined_plan = break_down_plan_chain.invoke({"plan": state["plan"]})
    state["plan"] = refined_plan.steps
    return state

# ========================================================================================
# task handler
# ========================================================================================




def create_task_handler_chain():

    tasks_handler_prompt_template ="""You are a master task dispatcher. Your goal is to select the perfect tool to execute the current task: "{curr_task}". You must also formulate the precise input (query) for that tool.

    You have access to the initial user question "{question}" and the work done so far "{past_steps}" for context.you also receive the last tool used {last_tool}

    Here are your available tools:
    - **`retrieve_chunks`**: Use this to search for factual information, definitions, rules, or specific data within the document library (e.g., "2025年度国家自然科学基金项目指南"，"2025年度国家自然科学基金项目指南","同济大学国家自然科学基金2025年申请注意事项"). This is your primary tool for information extraction from provided documents.
    - **`answer_from_context`**: Use this ONLY when the plan explicitly states to synthesize, compare, or reason based on information already gathered in `aggregated_context`. The query should be a direct question to be answered from the context.
    
    Based on the current task, select the best tool and generate the most effective query.
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
    print(state["plan"])
    pprint("--------------------") 

    if not state['past_steps']:
        state["past_steps"] = []

    curr_task = state["plan"][0]

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
       
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="answer_from_context"

    elif output.tool == "code_interpreter":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="code_interpreter"
    
    elif output.tool == "web_search":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="web_search"

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
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    # elif state["tool"] == "retrieve_quotes":
    #     return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")

# ========================================================================================
# 重新计划步骤
# ========================================================================================


def create_replanner_chain():

    replanner_prompt_template =""" For the given objective, come up with a simple step by step plan of how to figure out the answer. 
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

    Your objective was this:
    {question}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    You already have the following context:
    {aggregated_context}

    Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
    Do not return previously done steps as part of the plan.

    the format is json so escape quotes and new lines.

    """

    replanner_prompt = PromptTemplate(
        template=replanner_prompt_template,
        input_variables=["question", "plan", "past_steps", "aggregated_context"],
        # partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
    )

    replanner_llm = get_chat_model()

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



def run_qualitative_chunks_retrieval_workflow(state: dict) -> dict:
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
    print("🚀 Running the integrated retrieval workflow (Retrieve -> Expand -> Rerank)...")
    
    question = state["query_to_retrieve_or_answer"]
    
    # --- 流程开始 ---
    
    # 步骤一：检索 (Retrieve)
    # 使用现有的混合检索器，它现在会在“子块”上进行检索
    retriever = create_hybrid_retriever(bm25_k=20, vector_k=20) # 召回更多候选以供精排
    print(f"1. 🔍 Retrieving child chunks for query: '{question}'")
    child_chunks = retriever.invoke(question)
    print(f"   ✅ Retrieved {len(child_chunks)} child chunks.")

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
    print(f"2. 🧱 Expanding to {len(unique_parent_chunks)} unique parent chunks.")

    if not unique_parent_chunks:
        # 如果没有检索到任何内容，直接返回
        print("   ⚠️ No unique parent chunks found. Skipping reranking.")
        state["aggregated_context"] = ""
        pprint("--------------------")
        return state

    # ========================= 【核心替换部分】 =========================
    
    # 步骤三：重排 (Rerank)
    # 使用新的 AiHubMixReranker 对父块进行重排序
    print("3. ⚖️ Reranking parent chunks using AiHubMix Reranker...")
    
    # 1. 使用您的工厂函数实例化重排器，并设置返回 top 3 的文档
    reranker = get_reranker_model(top_n=3) 
    
    # 2. 需要用transform_documents方法。

    reranked_docs = reranker.transform_documents(
        documents=unique_parent_chunks,
        query=question
    )
    print(f"   ✅ Reranked and selected top {len(reranked_docs)} parent chunks via API.")
    
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
    You need to determine if the question can be fully answered relying only the given context.
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
# 最终agent
# ========================================================================================

def create_agent():
    
    agent_workflow = StateGraph(PlanExecute)


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
    agent_workflow.set_entry_point("planner")

    # 边的定义

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

    # After getting the final answer we end
    agent_workflow.add_edge("get_final_answer", END)


    plan_and_execute_app = agent_workflow.compile()

    return plan_and_execute_app