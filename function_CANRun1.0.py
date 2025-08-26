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
    curr_state: str
    question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    curr_context: str
    aggregated_context: str
    tool: str
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
    tasks_handler_prompt_template = """You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
    You have the following tools at your disposal:
    Tool A: a tool that retrieves relevant information from a vector store based on a given query.
    - use Tool A when you think the current task should search for information in the vector store.
    
    you also receive the last tool used {last_tool}


    You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
    You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
    if you decide to use Tools A, output the query to be used for the tool and also output the relevant tool.
    if you decide to use Tool D, output the question to be used for the tool, the context, and also that the tool to be used is Tool D.

    """

    class TaskHandlerOutput(BaseModel):
        """Output schema for the task handler."""
        query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
        curr_context: str = Field(description="The context to be based on in order to answer the query.")
        tool: str = Field(description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context.")


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
# 检索
# ========================================================================================



from typing import TypedDict, List
from pprint import pprint



def run_qualitative_chunks_retrieval_workflow(state: PlanExecute):
    """
    运行定性检索工作流，现已简化为使用一个结合了检索和内容压缩的检索器。
    
    Args:
        state: The current state of the plan execution.
        
    Returns:
        The state with the updated aggregated context.
    """
    
    # 维持状态更新逻辑 1: 更新当前状态
    state["curr_state"] = "retrieve_chunks"
    print("Running the unified chunk retrieval...")
    
    question = state["query_to_retrieve_or_answer"]
    
    # 使用优化的检索器（这里我们先假设它能正确返回一个检索器实例）
    retriever = create_hybrid_retriever()
    
    print(f"Retriever type: {type(retriever)}") # 加上这句日志，方便调试

    # --- 修改开始 ---
    # 直接、清晰地调用 invoke 方法
    try:
        print("🔍 Invoking retriever...")
        # EnsembleRetriever 和其他标准检索器都有 .invoke() 方法
        docs = retriever.invoke(question) 
        print(f"✅ Retrieved {len(docs)} documents.")

        # 注意：如果您的 `create_hybrid_retriever` 返回的是我们第一段代码中的
        # 预处理检索器，其 invoke 的返回格式可能是字典列表。
        # 如果是标准的 EnsembleRetriever，返回的是 Document 列表。
        # 这里需要做一个判断来兼容两种情况。
        if docs and isinstance(docs[0], dict):
            print("   (转换从字典格式到 Document 对象)")
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=result.get('content', ''),
                    metadata=result.get('metadata', {})
                ) for result in docs
            ]

    except Exception as e:
        print(f"❌ 检索失败: {e}")
        # 在这里可以添加备用逻辑，或者直接返回空结果
        docs = []
    # --- 修改结束 ---

    # 聚合文档内容
    relevant_context = " ".join(doc.page_content for doc in docs)
    
    # 维持状态更新逻辑 2: 更新聚合上下文
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
        
    state["aggregated_context"] += relevant_context
    
    pprint("--------------------")
    return state

# ========================================================================================
# 回答flow
# ========================================================================================


def create_question_answer_from_context_cot_chain():
    class QuestionAnswerFromContext(BaseModel):
        answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")

    question_answer_from_context_llm = get_chat_model()


    question_answer_cot_prompt_template = """ 
    Examples of Chain-of-Thought Reasoning

    Example 1

    Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
    Question: Who is the tallest person?
    Reasoning Chain:
    The context tells us Mary is taller than Jane
    It also says Jane is shorter than Tom
    And Tom is the same height as David
    So the order from tallest to shortest is: Mary, Tom/David, Jane
    Therefore, Mary must be the tallest person

    Example 2
    Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
    A third spell created a bright light at the end of the caster's wand.
    Question: Based on the context, if Harry cast these spells, what could he do?
    Reasoning Chain:
    The context describes three different magic spells
    The first spell allows turning a person into an animal temporarily
    The second spell can levitate or float objects
    The third spell creates a bright light
    If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
    So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
    Instructions.

    Example 3 
    Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
    Question: Why did Harry receive a broomstick for his birthday?
    Reasoning Chain:
    The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
    However, the context does not provide any information about why he received that specific present or who gave it to him.
    There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
    Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

    For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
    just like in the previous examples.
    Context
    {context}
    Question
    {question}
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


    agent_workflow.add_edge("planner", "break_down_plan")


    # From break_down_plan we go to task handler
    agent_workflow.add_edge("break_down_plan", "task_handler")

    # From task handler we go to either retrieve or answer
    agent_workflow.add_conditional_edges("task_handler", retrieve_or_answer, {"chosen_tool_is_retrieve_chunks": "retrieve", "chosen_tool_is_retrieve_summaries": "retrieve", "chosen_tool_is_answer": "answer"})

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