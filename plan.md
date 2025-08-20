
---

### **项目需求与改造方案总结**

#### **一、 核心目标**

我们的核心目标是将一个基于“哈利波特”小说的问答项目，**彻底改造**为一个专业的、面向**《中国博士后科学基金资助指南》**的智能问答助手。这个助手必须超越简单的“一问一答”，具备处理复杂查询的多种高级能力。

#### **二、 六大核心能力需求**

我们把这些高级能力具体化为六个明确的需求点：

1.  **精准信息检索 (Accurate Retrieval)**: Agent 的基础。能根据直接问题，从指南中精确定位答案。
    *   *对应路径*: `simple_retrieval`

2.  **跨文档比较分析 (Cross-Document Comparison)**: 能比较不同年份指南的条款差异，或不同资助类型的异同。
    *   *对应路径*: `comparison` (利用 Analytical Strategy 实现)

3.  **复杂逻辑推理 (Complex Logical Reasoning)**: 能根据指南中的规则和用户提供的个人情况，进行逻辑判断。例如，“我正在承担XX项目，还能申请YY资助吗？”
    *   *对应路径*: `logical_reasoning` (利用 Analytical Strategy 实现)

4.  **场景化信息综合 (Scenario-based Synthesis)**: 能根据用户的完整个人背景，生成一份个性化的、完整的申请流程和注意事项指南。
    *   *对应路径*: `scenario_synthesis` (利用 Contextual Strategy 实现)

5.  **外部工具调用 (Tool Use)**: 当指南中没有答案时（如查询最新通知），能调用搜索引擎等外部工具。
    *   *对应路径*: `tool_use`

6.  **数据处理与可视化 (Data Visualization)**: 能从指南中提取结构化数据（如资助金额、评审权重），并以表格或图表形式呈现。
    *   *对应路径*: `data_visualization`

#### **三、 核心改造方案：基于 LangGraph 的分层自适应架构**

为了实现上述所有能力，我们摒弃了原项目简单的 Plan-and-Execute 模式，采用了更先进、更模块化的 **分层自适应架构**，其核心是 LangGraph。

1.  **分层状态管理 (Hierarchical State)**:
    *   我们定义了一个轻量级的**主图状态 `MainGraphState`**，只负责传递全局信息（如原始问题、用户画像、最终答案）。
    *   我们为每一种核心处理策略（Factual, Analytical, Contextual 等）都定义了**独立的子图状态**（如 `FactualSubGraphState`, `AnalyticalSubGraphState`），它们各自管理内部复杂的中间数据，实现了高度的模块化和解耦。

2.  **自适应路由 (Adaptive Routing)**:
    *   整个工作流的入口是一个**查询分类节点 `classify_query`**。它像一个智能调度中心，分析用户问题的意图，并将其路由到最合适的子图进行处理。


3.  ** 模块化子图 (Modular Sub-graphs) 的详细设计**

“模块化子图”的核心思想是，将每一种复杂的处理策略（如 `Factual`, `Analytical`）封装成一个独立的、自包含的、可复用的 LangGraph 工作流。它就像一个功能强大的“黑盒”函数：给它输入，它就返回结果，而主图无需关心其内部复杂的实现细节。

下面，我们结合 adaptive_retrieval.ipynb 的方法，详细描述每个子图的设计：

#### **A. 事实检索子图 (`Factual Sub-graph`)**

*   **目的**: 处理直接、单一的事实性问题。这是所有更复杂流程的**基础构建块 (Fundamental Building Block)**。
*   **输入**: 一个 `question` (字符串)。
*   **输出**: 一个 `generation` (最终答案字符串)。
*   **内部状态**: `FactualSubGraphState`，管理着从查询增强到答案生成的所有中间数据。
*   **内部工作流 (Workflow)**:
    1.  **入口节点: `enhance_query`**: 接收 `question`，利用 LLM 将其改写为对向量数据库更友好的 `enhanced_question`。
    2.  **节点: `retrieve_documents`**: 使用 `enhanced_question` 从向量库中召回一批候选文档。
    3.  **节点: `rerank_documents`**: 这是一个质量控制关卡。它使用 LLM 逐一评估每个文档与问题的相关性，并剔除低分文档，确保进入下一步的都是高质量信息。
    4.  **出口节点: `generate_answer`**: 将所有通过筛选的文档内容合并为上下文，然后调用 LLM，基于此上下文生成最终的、有理有据的答案。

#### **B. 分析推理子图 (`Analytical Sub-graph`)**

*   **目的**: 处理需要“比较”、“分析”或“推理”的复杂问题。例如，“对比A和B的区别”或“根据规则X，在Y场景下会如何？”。
*   **输入**: 一个 `question` (字符串)。
*   **输出**: 一个 `generation` (最终的综合性答案字符串)。
*   **内部状态**: `AnalyticalSubGraphState`，核心是管理子问题列表 (`sub_queries`) 和它们的答案 (`sub_query_results`)。
*   **内部工作流 (Workflow)**:
    1.  **入口节点: `generate_sub_queries`**: 这是此策略的**核心**。它接收复杂问题，并调用 LLM 将其分解为一系列更小、更具体、可以被独立回答的事实性子问题。
    2.  **核心节点: `process_sub_queries`**: 这个节点体现了**模块化复用**的精髓。它会遍历 `sub_queries` 列表，然后**为每一个子问题，完整地调用一次上面定义好的 `Factual Sub-graph`**。它将子问题的答案收集起来，存入 `sub_query_results` 字典中。
    3.  **出口节点: `synthesize_analytical_answer`**: 它接收原始的复杂问题和所有子问题的答案，然后调用一个专门的 LLM Chain，将这些零散的信息片段“编织”成一段逻辑连贯、条理清晰的最终报告。

#### **C. 场景综合子图 (`Contextual Sub-graph`)**

*   **目的**: 处理带有用户个人背景的“场景题”，提供个性化指导。
*   **输入**: 一个 `question` (字符串) 和一个 `user_profile` (字典)。
*   **输出**: 一个 `generation` (个性化的行动指南)。
*   **内部状态**: `ContextualSubGraphState`。
*   **内部工作流 (Workflow)**:
    1.  **入口节点: `contextualize_query`**: 将用户的 `question` 和 `user_profile` 融合，生成一个高度个性化的、用于检索的查询。
    2.  **复用与调整**: 接下来，它可以**直接复用 `Factual Sub-graph` 中的 `retrieve_documents` 和 `rerank_documents` 节点**。我们甚至可以为 `rerank_documents` 节点提供一个带有用户背景的、定制化的 Prompt，使其在排序时不仅考虑相关性，还考虑“对该用户的重要性”。
    3.  **出口节点: `synthesize_scenario_guide`**: 一个专门的生成节点，其 Prompt 指导 LLM 根据上下文和用户背景，生成步骤清晰、考虑周全的行动指南。

---

### ** 我们的开发范式 (Node-Chain-Prompt-Model Paradigm)**

为了实现上述优雅、可靠的子图，我们必须严格遵守一套开发纪律。这套范式确保了我们代码的每一部分都有单一、明确的职责，使得整个系统清晰、可维护、易于测试。

我把它总结为 **“四层分离”** 原则：

1.  **表现层 (Node - 节点)**:
    *   **文件**: nodes.py
    *   **职责**: **只做编排 (Orchestration)**。节点函数是图的“骨架”，它的代码应该极其简洁。其唯一的任务是：
        1.  从 `chains` 模块的工厂函数中获取一个“工具”（Chain）。
        2.  调用这个“工具”，并把 `state` 中的数据作为输入传给它。
        3.  将“工具”返回的结果，更新回 `state` 中。
    *   **禁止**: 节点内部绝不能出现 Prompt 字符串，不能直接实例化 LLM，也不能有复杂的业务逻辑。

2.  **逻辑层 (Chain - 链)**:
    *   **文件**: chains.py
    *   **职责**: **封装工具 (Encapsulation)**。此文件中的每个 `create_..._chain` 函数都是一个**工厂**，负责组装并返回一个完整的、可执行的 LangChain Runnable（我们的“工具”）。它将 Prompt、LLM 和 Output Parser 粘合在一起。
    *   **命名**: 必须以 `create_` 作为前缀。

3.  **指令层 (Prompt - 提示)**:
    *   **文件**: prompts.py
    *   **职责**: **定义意图 (Intention)**。此文件只包含多行字符串常量，即我们的 Prompt Template。它将所有对 LLM 的指令（“你是一个XX专家，请你做YY事情...”）与代码逻辑完全分离，便于集中管理和优化。

4.  **结构层 (Model - 模型)**:
    *   **文件**: models.py
    *   **职责**: **定义数据契约 (Data Contract)**。此文件只包含 Pydantic `BaseModel` 的定义。我们为每一个期望从 LLM 获得的、有特定结构的输出都定义一个模型。这通过 LangChain 的 `.with_structured_output()` 方法，强制 LLM 返回可靠、类型安全的 JSON，彻底消除了不稳定的字符串解析。

通过严格遵守这“四层分离”的范式，我们构建的每一个功能（无论是节点还是子图）都将是健壮、清晰且高度模块化的。