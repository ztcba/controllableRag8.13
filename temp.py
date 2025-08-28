
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