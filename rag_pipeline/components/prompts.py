# rag_pipeline/components/prompts.py

keep_only_relevant_content_prompt_template = """you receive a query: {query} and retrieved docuemnts: {retrieved_documents} from a
vector store.
You need to filter out all the non relevant information that don't supply important information regarding the {query}.
your goal is just to filter out the non relevant information.
you can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
output the filtered relevant content.
"""

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

is_relevant_content_prompt_template = """you receive a query: {query} and a context: {context} retrieved from a vector store. 
You need to determine if the document is relevant to the query. """

is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
you don't mind if it doesn't make sense, as long as it is grounded in the context.
output a json containing the answer to the question, and appart from the json format don't output any additional text.

"""

can_be_answered_prompt_template = """You receive a query: {question} and a context: {context}. 
You need to determine if the question can be fully answered based on the context."""

is_distilled_content_grounded_on_content_prompt_template = """you receive some distilled content: {distilled_content} and the original context: {original_context}.
    you need to determine if the distilled content is grounded on the original context.
    if the distilled content is grounded on the original context, set the grounded field to true.
    if the distilled content is not grounded on the original context, set the grounded field to false."""

planner_prompt =""" For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

"""
# Note: This one was inside a function, but we extract it.

break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
you need to go through the plan refine it according to this:
1. every step has to be able to be executed by either:
    i. retrieving relevant information from a vector store of book chunks
    ii. retrieving relevant information from a vector store of chapter summaries
    iii. retrieving relevant information from a vector store of book quotes
    iv. answering a question from a given context.
2. every step should contain all the information needed to execute it.

output the refined plan
"""
# 你会收到一份规划{plan}，其中包含为回答某个查询而需依次执行的一系列步骤。  
#请你逐一审阅该规划，并按以下要求对其进行优化：
#1. 每一步都必须能够通过以下方式之一执行：
    #i. 从书籍文本块的向量数据库中检索相关信息
    #ii. 从章节摘要的向量数据库中检索相关信息
    #iii. 从书籍引用内容的向量数据库中检索相关信息
    #iv. 根据给定上下文回答问题
#2. 每一步都应包含执行该步骤所需的全部信息。

#请输出优化后的规划。
# 它接收一个已有的计划，并要求将其中的每一步分解为更小的、可由工具直接执行的子任务，
# 强调这些子任务应该是“可检索的”或“可回答的”。

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

tasks_handler_prompt_template = """You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
    You have the following tools at your disposal:
    Tool A: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
    - use Tool A when you think the current task should search for information in the book chunks.
    Took B: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
    - use Tool B when you think the current task should search for information in the chapter summaries.
    Tool C: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
    - use Tool C when you think the current task should search for information in the book quotes.
    Tool D: a tool that answers a question from a given context.
    - use Tool D ONLY when you the current task can be answered by the aggregated context {aggregated_context}

    you also receive the last tool used {last_tool}
    if {last_tool} was retrieve_chunks, use other tools than Tool A.

    You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
    You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
    if you decide to use Tools A,B or C, output the query to be used for the tool and also output the relevant tool.
    if you decide to use Tool D, output the question to be used for the tool, the context, and also that the tool to be used is Tool D.

    """
# 上面是工具使用说明
 #你还会收到上一次使用的工具 {last_tool}。
# 如果 {last_tool} 是 retrieve_chunks（文本块检索），则使用工具 A 以外的其他工具。

# 你也可以利用已执行步骤 {past_steps} 来辅助决策和理解任务背景。
 #你还可以参考初始用户问题 {question} 来辅助决策和理解任务背景。

# 如果你决定使用工具 A、B 或 C，需输出该工具要使用的查询词以及对应的工具。
# 如果你决定使用工具 D，需输出该工具要处理的问题、相关上下文，以及说明使用的是工具 D。

anonymize_question_prompt_template = """ You are a question anonymizer. The input You receive is a string containing several words that
    construct a question {question}. Your goal is to changes all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
    ```example1:
            if the input is \"who is harry potter?\" the output should be \"who is X?\" and the mapping should be {{\"X\": \"harry potter\"}} ```
    ```example2:
            if the input is \"how did the bad guy played with the alex and rony?\"
            the output should be \"how did the X played with the Y and Z?\" and the mapping should be {{\"X\": \"bad guy\", \"Y\": \"alex\", \"Z\": \"rony\"}}```
    you must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
    output the anonymized question and the mapping as two separate fields in a json format as described here, without any additional text apart from the json format.
   """

de_anonymize_plan_prompt_template = """ you receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. you also receive
the mapping for those variables to words {mapping}. replace all the variables in the list of tasks with the mapped words. if no variables are present,
return the original list of tasks. in any case, just output the updated list of tasks in a json format as described here, without any additional text apart from the
"""

can_be_answered_already_prompt_template = """You receive a query: {question} and a context: {context}.
    You need to determine if the question can be fully answered relying only the given context.
    The only infomation you have and can rely on is the context you received. 
    you have no prior knowledge of the question or the context.
    if you think the question can be answered based on the context, output 'true', otherwise output 'false'.
    """

# 注意：我将 `create_plan_chain` 函数内的 `planner_prompt` 字符串也提取出来了，它的新名字是 `planner_prompt`。*

query_classifier_prompt_template = """You are an expert at routing a user's question to the correct workflow.
Based on the user's question, you must classify it into one of the following categories.
Your output must be a single word from the category list. Do not add any other text.

Here are the available categories:

1.  **simple_retrieval**:
    *   **Description**: The user is asking a direct, factual question that can likely be answered by retrieving a specific piece of information from the documents.
    *   **Examples**:
        *   "博士后科学基金面上资助的金额是多少？" (What is the funding amount for the Postdoctoral Science Foundation's general grant?)
        *   "申请特别资助（站中）需要满足哪些基本条件？" (What are the basic requirements to apply for the Special Grant (during postdoctoral research)?)

2.  **comparison**:
    *   **Description**: The user wants to compare, contrast, or find differences between two or more things. This often involves looking at different versions of documents or different sections within a document.
    *   **Examples**:
        *   "对比一下2023年和2024年的申请条件有什么变化？" (What are the changes in application requirements between 2023 and 2024?)
        *   "面上资助和特别资助（站前）在评审标准上有什么不同？" (What are the differences in evaluation criteria between the General Grant and the Special Grant (pre-postdoc)?)

3.  **logical_reasoning**:
    *   **Description**: The user's question requires applying rules, conditions, or constraints mentioned in the documents to a specific situation. It involves deduction and logical inference.
    *   **Examples**:
        *   "如果我的合作导师作为项目负责人正在承担国家重大科研项目，我还能申请特别资助（站前）吗？" (If my co-supervisor is leading a major national research project, am I still eligible to apply for the Special Grant (pre-postdoc)?)
        *   "我去年获得了面上资助，今年还能申请青年人才托举工程吗？" (I received the General Grant last year, can I apply for the Young Elite Scientist Sponsorship Program this year?)

4.  **scenario_synthesis**:
    *   **Description**: The user presents a complex, personal scenario and asks for a comprehensive guide or plan. This requires synthesizing information from multiple parts of the knowledge base to create a personalized response.
    *   **Examples**:
        *   "我是一名刚入站的博士后，研究方向是人工智能，我想申请面上资助，请问完整的流程和注意事项是什么？" (I am a newly registered postdoc in the field of AI, and I want to apply for the General Grant. What is the complete process and what should I pay attention to?)
        *   "作为一名外籍博士后，在中国申请基金有哪些特殊的政策和流程？" (As a foreign postdoc, what are the special policies and procedures for applying for funds in China?)

5.  **tool_use**:
    *   **Description**: The user's question likely cannot be answered by the existing documents alone and requires up-to-date, external information. This indicates the need to use a tool like a web search engine.
    *   **Examples**:
        *   "基金委最近有没有发布关于海外引才专项的最新通知？" (Has the foundation recently released any new announcements regarding the special program for attracting overseas talent?)
        *   "今天的人民币兑美元汇率是多少？" (What is the RMB to USD exchange rate today?)

6.  **data_visualization**:
    *   **Description**: The user is asking for structured data to be presented in a specific format, like a table or chart. This may involve retrieving data and then using a tool to visualize it.
    *   **Examples**:
        *   "请用表格列出面上资助、特别资助（站前）和特别资助（站中）的资助金额、申请条件和评审重点。" (Please list the funding amount, application requirements, and evaluation focus for the General Grant, Special Grant (pre-postdoc), and Special Grant (during postdoc) in a table.)
        *   "各类资助的评审指标和权重是怎样的？能用饼图展示吗？" (What are the evaluation metrics and weights for each type of grant? Can you show it in a pie chart?)

---
User Question:
"{question}"

Category:
"""

query_enhancement_prompt_template = """You are an expert at query optimization for vector retrieval.
Your task is to take a user's question and enhance it to be more specific and clear for a vector database search.
The knowledge base contains the "China Postdoctoral Science Foundation Funding Guide".

Focus on expanding acronyms, adding keywords, and clarifying the intent.
For example:
- Original: "面上资助金额是多少？"
- Enhanced: "中国博士后科学基金面上资助项目的具体资助金额是多少？"
- Original: "特助站前申请条件"
- Enhanced: "申请中国博士后科学基金特别资助（站前）需要满足哪些详细的基本条件和资格要求？"

Do not answer the question, only provide the enhanced query.

Original Question:
"{question}"
"""

document_reranking_prompt_template = """You are a meticulous fact-checker. Your task is to evaluate the relevance of a retrieved document to a user's question.
The context is the "China Postdoctoral Science Foundation Funding Guide".

Provide a relevance score from 1 to 10, where 10 is perfectly relevant.
Also, provide a brief explanation for your score.

User Question:
"{question}"

Retrieved Document:
---
{document}
---

Evaluate the document's relevance to the question.
"""

# --- 事实型查询的提示词模板 ---
generation_prompt_template = """你是一位专精于《中国博士后科学基金资助指南》的专业顾问。你的任务是基于提供的上下文信息，为用户的问题生成准确、详细、实用的答案。

请遵循以下原则：
1. 只基于提供的上下文信息回答问题，不要添加上下文中没有的信息
2. 如果上下文信息不足以完全回答问题，请明确指出哪些方面需要更多信息
3. 答案要结构清晰，条理分明，便于理解
4. 对于政策条款要准确引用，对于申请流程要详细说明
5. 使用专业但易懂的语言

上下文信息：
{context}

用户问题：
{question}

请基于上述上下文信息，为用户问题提供详细的答案：
"""
# --- 事实型查询的提示词模板 ---

# --- 分析推理子图的提示词模板 ---
sub_query_generation_prompt_template = """你是一位专精于《中国博士后科学基金资助指南》的资深政策分析专家。你的任务是将一个复杂的分析型问题分解为多个具体的、可独立回答的子问题。

复杂问题的类型通常包括：
1. **比较分析**: 如"对比不同资助类型的区别"
2. **逻辑推理**: 如"根据申请条件判断是否符合资格"
3. **多维度分析**: 如"全面分析某项政策的影响"

分解原则：
1. 每个子问题都应该是具体的、事实性的，可以通过检索文档直接回答
2. 子问题应该覆盖原问题的所有重要维度
3. 子问题之间应该有逻辑关联，能够支撑对原问题的完整回答
4. 避免过于宽泛或过于细碎的子问题
5. 通常分解为3-6个子问题最为合适

示例：
原问题："对比面上资助和特别资助（站前）在申请条件、资助金额和评审标准方面的差异"
子问题：
1. 面上资助的申请条件是什么？
2. 特别资助（站前）的申请条件是什么？
3. 面上资助的资助金额是多少？
4. 特别资助（站前）的资助金额是多少？
5. 面上资助的评审标准和重点是什么？
6. 特别资助（站前）的评审标准和重点是什么？

现在，请将以下复杂问题分解为具体的子问题：

原问题：
{question}

请生成子问题列表：
"""

analytical_synthesis_prompt_template = """你是一位专精于《中国博士后科学基金资助指南》的资深政策分析专家。你的任务是基于多个子问题的答案，生成一个全面、逻辑清晰、结构完整的综合性答案。

综合分析原则：
1. **逻辑结构**: 按照问题的逻辑层次组织答案，确保条理清晰
2. **对比分析**: 如果涉及比较，要明确指出相同点和不同点
3. **推理论证**: 如果涉及逻辑推理，要清楚说明推理过程和结论
4. **实用指导**: 结合具体情况提供实际的指导建议
5. **完整性**: 确保答案覆盖原问题的所有重要方面

格式要求：
- 使用清晰的段落结构
- 重要信息可以使用要点列表
- 对比分析可以使用对照表格
- 结论要明确突出

原始问题：
{question}

子问题及其答案：
{sub_query_results}

请基于以上子问题的答案，生成一个全面、专业的综合性回答：
"""
# --- 分析推理子图的提示词模板 ---

# --- 工具使用子图的提示词模板 ---
tool_decision_prompt_template = """你是一位专精于《中国博士后科学基金资助指南》的专业顾问。你需要判断用户的问题是否需要使用外部工具（如网络搜索）来获取最新信息。

需要使用外部工具的情况：
1. **时效性信息**: 询问最新通知、公告、政策变化
2. **实时数据**: 如汇率、当前日期、最新统计数据
3. **超出指南范围**: 询问指南中明确不包含的信息
4. **外部机构信息**: 询问其他机构的具体政策或联系方式

不需要使用外部工具的情况：
1. **指南内容**: 资助金额、申请条件、评审标准等指南中的固定信息
2. **一般性咨询**: 申请流程、注意事项等常规问题
3. **比较分析**: 不同资助类型间的对比
4. **逻辑推理**: 基于现有条件的资格判断

用户问题：
{question}

请判断这个问题是否需要使用外部搜索工具：
"""

search_query_generation_prompt_template = """你是一位搜索专家。用户的问题需要通过网络搜索来获取最新信息。请为用户的问题生成一个优化的搜索查询。

搜索查询优化原则：
1. **关键词提取**: 提取最重要的关键词
2. **机构名称**: 包含相关的官方机构名称
3. **时间限定**: 如果需要最新信息，添加时间限定词
4. **搜索语言**: 使用中文搜索中国相关信息
5. **简洁明确**: 避免过长或过于复杂的查询

示例：
- 用户问题："基金委最近有没有发布关于海外引才专项的最新通知？"
- 搜索查询："中国博士后科学基金委员会 海外引才专项 最新通知 2024 2025"

用户问题：
{question}

请生成优化的搜索查询：
"""

tool_use_answer_generation_prompt_template = """你是一位专精于《中国博士后科学基金资助指南》的专业顾问。用户的问题需要结合网络搜索结果来回答。

请基于搜索结果，为用户提供准确、有用的答案。

回答原则：
1. **信息整合**: 结合搜索结果和你对博士后科学基金的专业知识
2. **来源标注**: 明确指出哪些信息来自搜索结果
3. **可靠性评估**: 评估搜索结果的可靠性
4. **专业建议**: 提供相关的专业建议和注意事项
5. **格式清晰**: 使用清晰的结构和格式

用户问题：
{question}

搜索结果：
{search_results}

请基于搜索结果，为用户问题提供详细的答案：
"""
# --- 工具使用子图的提示词模板 ---