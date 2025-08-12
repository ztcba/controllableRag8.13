# 将所有作为 LangGraph 节点的函数（即接收 state 并返回更新后的 state 的函数）集中到一个文件中。
# 原因: 节点是图的基本执行单元。将它们放在一起可以让我们专注于每个节点的具体业务逻辑，而不用关心它们是如何连接的。
# 一个节点，一个功能: 每个函数代表图中的一个明确步骤
# 依赖注入: 节点函数不应再依赖于全局变量（如 planner 或 task_handler_chain）。
# 相反，它们应该在函数内部通过调用 components.chains 中的工厂函数（如 create_plan_chain()）来创建所需的链实例。这使得节点更加独立和可测试。
# 清晰的 State: 每个节点函数的 state 参数都应该用我们之前在 state.py 中定义的相应 TypedDict 进行类型注解，这能极大地提高代码的可读性和健壮性。
# 我们将把 创建链的实例 和 调用链 的逻辑都放在节点函数内部。

# src/rag_pipeline/graph/nodes.py
from pprint import pprint

# LangChain/LangGraph imports
# (We might not need these directly in nodes.py, but it's good practice to have them if needed)

# Local application imports
from src.rag_pipeline.components import chains
from src.rag_pipeline.components.retrievers import (
    chunks_query_retriever,
    chapter_summaries_query_retriever,
    book_quotes_query_retriever,
)
from src.rag_pipeline.graph import state  # Import the state definitions
from src.rag_pipeline.utils.helpers import escape_quotes, text_wrap

# We will also need to import the compiled sub-workflows later.
# For now, we define the nodes that will be part of them.
# We'll import the compiled apps inside the nodes that run them.
from src.rag_pipeline.graph import workflows

# 注意：检查循环导入: 请注意，nodes.py 中的 run_*_workflow 函数会从 workflows.py 导入 ..._app。而 workflows.py 会从 nodes.py 导入节点。这会造成循环导入！
# 解决方案: 将 from src.rag_pipeline.graph.workflows import ..._app 这几行导入语句移动到需要它们的函数内部，而不是放在文件顶部。这是一种延迟导入，可以有效避免循环依赖问题。

# --- Nodes for Qualitative Sub-workflows子图节点 ---

# Node for sub-graph: retrieve_chunks_context_per_question
def retrieve_chunks_context_per_question(state: state.QualitativeRetrievalGraphState):
    """
    Retrieves relevant chunks for a given question.
    """
    print("Retrieving relevant chunks...")
    question = state["question"]
    docs = chunks_query_retriever.get_relevant_documents(question)
    context = " ".join(doc.page_content for doc in docs)
    context = escape_quotes(context)
    return {"context": context, "question": question}

# Node for sub-graph: retrieve_summaries_context_per_question
def retrieve_summaries_context_per_question(state: state.QualitativeRetrievalGraphState):
    """
    Retrieves relevant chapter summaries for a given question.
    """
    print("Retrieving relevant chapter summaries...")
    question = state["question"]
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(question)
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )
    context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}

# Node for sub-graph: retrieve_book_quotes_context_per_question
def retrieve_book_quotes_context_per_question(state: state.QualitativeRetrievalGraphState):
    """
    Retrieves relevant book quotes for a given question.
    """
    print("Retrieving relevant book quotes...")
    question = state["question"]
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(question)
    book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_qoutes_context = escape_quotes(book_qoutes)
    return {"context": book_qoutes_context, "question": question}

# Node for sub-graph: keep_only_relevant_content
def keep_only_relevant_content(state: state.QualitativeRetrievalGraphState):
    """
    Filters the context to keep only content relevant to the question.
    """
    # Instantiate the chain inside the node
    chain = chains.create_keep_only_relevant_content_chain()
    
    question = state["question"]
    context = state["context"]
    input_data = {"query": question, "retrieved_documents": context}
    
    print("keeping only the relevant content...")
    pprint("--------------------")
    output = chain.invoke(input_data)
    
    relevant_content = "".join(output.relevant_content)
    relevant_content = escape_quotes(relevant_content)
    
    # This node returns all necessary fields for the state
    return {"relevant_context": relevant_content, "context": context, "question": question}

# Conditional edge for sub-graph: is_distilled_content_grounded_on_content
def is_distilled_content_grounded_on_content(state: state.QualitativeRetrievalGraphState):
    """
    Checks if the distilled content is grounded in the original context.
    """
    pprint("--------------------")
    print("Determining if the distilled content is grounded on the original context...")
    
    # Instantiate the chain inside the conditional function
    chain = chains.create_is_distilled_content_grounded_on_content_chain()
    
    distilled_content = state["relevant_context"]
    original_context = state["context"]
    input_data = {"distilled_content": distilled_content, "original_context": original_context}
    
    output = chain.invoke(input_data)
    grounded = output.grounded
    
    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"

# Node for sub-graph: answer_question_from_context
def answer_question_from_context(state: state.QualitativeAnswerGraphState):
    """
    Generates an answer to a question based on the provided context.
    """
    # Instantiate the chain inside the node
    chain = chains.create_question_answer_from_context_cot_chain()
    
    question = state["question"]
    # Handle the case for the final answer node where aggregated_context is used
    context = state.get("aggregated_context") or state["context"]
    
    input_data = {"question": question, "context": context}
    print("Answering the question from the retrieved context...")
    
    output = chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f'answer before checking hallucination: {answer}')
    
    return {"answer": answer, "context": context, "question": question}

# Conditional edge for sub-graph: is_answer_grounded_on_context
def is_answer_grounded_on_context(state: state.QualitativeAnswerGraphState):
    """
    Checks if the generated answer is grounded in the provided context.
    """
    print("Checking if the answer is grounded in the facts...")
    
    # Instantiate the chain
    chain = chains.create_is_grounded_on_facts_chain()
    
    context = state["context"]
    answer = state["answer"]
    
    result = chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts
    
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"
    

# --- 主图节点Nodes for the Main Agent (PlanExecute) ---
# 构成顶层 Plan-and-Execute Agent 的节点

def anonymize_queries(state: state.PlanExecute):
    """
    Anonymizes the main question by replacing named entities with variables.
    """
    state["curr_state"] = "anonymize_question"
    print("Anonymizing question")
    pprint("--------------------")
    
    # Instantiate chain
    anonymize_chain = chains.create_anonymize_question_chain()
    
    input_values = {"question": state['question']}
    anonymized_question_output = anonymize_chain.invoke(input_values)
    
    state["anonymized_question"] = anonymized_question_output["anonymized_question"]
    state["mapping"] = anonymized_question_output["mapping"]
    
    print(f'anonymized_question: {state["anonymized_question"]}')
    pprint("--------------------")
    return state

def plan_step(state: state.PlanExecute):
    """
    Generates an initial plan to answer the anonymized question.
    """
    state["curr_state"] = "planner"
    print("Planning step")
    pprint("--------------------")
    
    # Instantiate chain
    planner_chain = chains.create_plan_chain()
    
    plan_output = planner_chain.invoke({"question": state['anonymized_question']})
    state["plan"] = plan_output.steps
    
    print(f'plan: {state["plan"]}')
    return state
    
def deanonymize_queries(state: state.PlanExecute):
    """
    De-anonymizes the generated plan using the stored mapping.
    """
    state["curr_state"] = "de_anonymize_plan"
    print("De-anonymizing plan")
    pprint("--------------------")
    
    # Instantiate chain
    deanonymize_chain = chains.create_deanonymize_plan_chain()
    
    deanonimzed_plan_output = deanonymize_chain.invoke({"plan": state["plan"], "mapping": state["mapping"]})
    state["plan"] = deanonimzed_plan_output.plan
    
    print(f'de-anonimized_plan: {state["plan"]}')
    return state

def break_down_plan_step(state: state.PlanExecute):
    """
    Refines the plan into actionable steps for the tools.
    Breaks down the plan steps into retrievable or answerable tasks.

    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    
    # Instantiate chain
    break_down_chain = chains.create_break_down_plan_chain()
    
    # Note: The original chain expected a dict, but the prompt implies passing the plan list directly.
    # Let's assume the chain can handle `state["plan"]`. We might need to adjust this if the chain fails.
    # Let's wrap it in a dictionary to be safe, as per original code's prompt.
    refined_plan_output = break_down_chain.invoke({"plan": state["plan"]})
    state["plan"] = refined_plan_output.steps
    return state

def run_task_handler_chain(state: state.PlanExecute):
    """
    Decides which tool to use for the current step in the plan.
    """
    state["curr_state"] = "task_handler"
    print("the current plan is:")
    print(state["plan"])
    pprint("--------------------") 

    if 'past_steps' not in state or not state['past_steps']:
        state["past_steps"] = []
    
    if 'tool' not in state:
        state['tool'] = "" # Initialize tool if not present

    curr_task = state["plan"][0]

    # Instantiate chain
    task_handler = chains.create_task_handler_chain()
    
    inputs = {
        "curr_task": curr_task,
        "aggregated_context": state.get("aggregated_context", ""),
        "last_tool": state["tool"],
        "past_steps": state["past_steps"],
        "question": state["question"]
    }
    
    output = task_handler.invoke(inputs)
  
    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    state["query_to_retrieve_or_answer"] = output.query
    state["tool"] = output.tool
    
    if output.tool == "answer_from_context":
        state["curr_context"] = output.curr_context
    
    # The tool name from the LLM needs to be mapped to our tool names
    if "retrieve_chunks" in output.tool:
        state["tool"] = "retrieve_chunks"
    elif "retrieve_summaries" in output.tool:
        state["tool"] = "retrieve_summaries"
    elif "retrieve_quotes" in output.tool:
        state["tool"] = "retrieve_quotes"
    elif "answer_from_context" in output.tool:
        state["tool"] = "answer"
    else:
        # Fallback or error, let's stick to the output tool name
        print(f"Warning: Unknown tool '{output.tool}' received from task handler.")

    return state  

# Conditional edge: retrieve_or_answer
def retrieve_or_answer(state: state.PlanExecute):
    """
    Routes to the correct tool-running node based on the task handler's decision.
    """
    state["curr_state"] = "decide_tool"
    print(f"deciding which tool to use: {state['tool']}")
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state["tool"] == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        # This is a critical failure point. It means the LLM gave a tool name
        # that we don't have a path for.
        raise ValueError(f"Invalid tool '{state['tool']}' in state. Cannot route.")

def run_qualitative_chunks_retrieval_workflow(state: state.PlanExecute):
    """
    Node that executes the chunks retrieval sub-workflow.
    """
    state["curr_state"] = "retrieve_chunks"
    print("Running the qualitative chunks retrieval workflow...")
    
    # We will compile the workflow app in workflows.py and import it
    from src.rag_pipeline.graph.workflows import qualitative_chunks_retrieval_workflow_app
    
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question, "context": "", "relevant_context": ""}
    
    # The sub-workflow returns its final state.
    sub_workflow_output = qualitative_chunks_retrieval_workflow_app.invoke(inputs)
    
    if "aggregated_context" not in state or not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += sub_workflow_output['relevant_context']
    return state

# ... Similar functions for summaries and quotes ...
def run_qualitative_summaries_retrieval_workflow(state: state.PlanExecute):
    state["curr_state"] = "retrieve_summaries"
    print("Running the qualitative summaries retrieval workflow...")
    from src.rag_pipeline.graph.workflows import qualitative_summaries_retrieval_workflow_app
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question, "context": "", "relevant_context": ""}
    sub_workflow_output = qualitative_summaries_retrieval_workflow_app.invoke(inputs)
    if "aggregated_context" not in state or not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += sub_workflow_output['relevant_context']
    return state

def run_qualitative_book_quotes_retrieval_workflow(state: state.PlanExecute):
    state["curr_state"] = "retrieve_book_quotes"
    print("Running the qualitative book quotes retrieval workflow...")
    from src.rag_pipeline.graph.workflows import qualitative_book_quotes_retrieval_workflow_app
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question, "context": "", "relevant_context": ""}
    sub_workflow_output = qualitative_book_quotes_retrieval_workflow_app.invoke(inputs)
    if "aggregated_context" not in state or not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += sub_workflow_output['relevant_context']
    return state

def run_qualtative_answer_workflow(state: state.PlanExecute):
    """
    Node that executes the qualitative answer sub-workflow for intermediate steps.
    """
    state["curr_state"] = "answer"
    print("Running the qualitative answer workflow...")
    from src.rag_pipeline.graph.workflows import qualitative_answer_workflow_app
    
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context, "answer": ""}
    
    sub_workflow_output = qualitative_answer_workflow_app.invoke(inputs)
    
    if "aggregated_context" not in state or not state["aggregated_context"]:
        state["aggregated_context"] = ""
    # Append the intermediate answer to the aggregated context
    state["aggregated_context"] += "\n" + sub_workflow_output["answer"]
    return state

def replan_step(state: state.PlanExecute):
    """
    Replans the next steps based on the current progress.
    """
    state["curr_state"] = "replan"
    print("Replanning step")
    pprint("--------------------")
    
    # Instantiate chain
    replanner_chain = chains.create_replanner_chain()
    
    inputs = {
        "question": state["question"],
        "plan": state["plan"], # The remaining plan
        "past_steps": state["past_steps"],
        "aggregated_context": state["aggregated_context"]
    }
    
    plan_output = replanner_chain.invoke(inputs)
    state["plan"] = plan_output.steps
    return state

# Conditional edge: can_be_answered
def can_be_answered(state: state.PlanExecute):
    """
    Checks if the original question can now be answered with the aggregated context.
    """
    state["curr_state"] = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    
    # If the plan is empty, we must be finished.
    if not state["plan"]:
        print("Plan is empty. Proceeding to final answer.")
        return "can_be_answered_already"
        
    # Instantiate chain
    can_be_answered_chain = chains.create_can_be_answered_already_chain()
    
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    
    output = can_be_answered_chain.invoke(inputs)
    
    if output.can_be_answered:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"

def run_qualtative_answer_workflow_for_final_answer(state: state.PlanExecute):
    """
    Node that generates the final answer to the original question.
    """
    state["curr_state"] = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    from src.rag_pipeline.graph.workflows import qualitative_answer_workflow_app
    
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context, "answer": ""}
    
    # We invoke the sub-workflow to get a grounded answer
    final_output = qualitative_answer_workflow_app.invoke(inputs)
    
    print("Final Answer:")
    pprint(final_output["answer"])
    
    state["response"] = final_output["answer"]
    return state