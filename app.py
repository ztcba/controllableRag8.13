import streamlit as st
import base64
from typing import Dict, Any
import io

# 从您的 agent 文件中导入必要的内容
# 请确保 agent_creator.py 和 app.py 在同一个目录下
from function_CANRun2 import create_agent, PlanExecute
from graphstate import PlanExecute 

# --- 1. Agent 加载与缓存 ---
# 使用 @st.cache_resource 装饰器可以确保 Agent 只被创建一次，
# 即使页面刷新或用户多次点击按钮，也能复用同一个 Agent 实例，
# 极大地提高了应用的响应速度和效率。
@st.cache_resource
def load_agent():
    """
    加载并返回编译好的 LangGraph Agent。
    """
    print("--- 正在创建和编译 Agent (此消息应只在应用首次启动时出现) ---")
    agent = create_agent()
    return agent

# --- 2. Streamlit 界面布局 ---
st.set_page_config(page_title="智能 Agent 助理", layout="wide")
st.title("🤖 智能 Agent 助理")
st.markdown("该 Agent 可以进行网络搜索、生成图表或执行复杂的 RAG 任务。")

# 加载 Agent
agent = load_agent()

# 初始化会话状态，用于存储历史记录等 (可选，但推荐)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 用户输入 ---
user_question = st.chat_input("请输入您的问题...")

if user_question:
    # 将用户问题添加到聊天记录中
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # --- 4. 调用 Agent 并处理输入 ---
    with st.chat_message("assistant"):
        # 使用 st.spinner 提供一个加载动画，提升用户体验
        with st.spinner("Agent 正在思考中..."):
            
            # 准备 Agent 的初始状态输入
            # 这是关键一步：必须提供一个符合 PlanExecute 结构的字典
            # 我们现在提供一个完整的字典，包含 `graphstate.py` 中定义的所有字段，
            # 并为它们设置了合理的初始值。
            initial_state = {
                "question": user_question,
                "curr_state": "",
                "query_to_retrieve_or_answer": "",
                "plan": [],
                "past_steps": [],
                "curr_context": "",
                "aggregated_context": """"# 2025年中国博士后科学基金“面上资助”项目评审指标及分值说明
2025年中国博士后科学基金“面上资助”项目评审包含三项核心指标，各指标的具体评价内容与对应分值如下：
1. **学术绩效**：作为评审的基础指标，主要评价项目申报人已取得的科研成果，该指标分值设定为30分。
2. **创新能力**：作为评审的核心指标，重点从研究内容的创新性、选题的自主性、学科交叉情况三个维度进行评价，该指标分值设定为60分，在三项指标中占比最高，是衡量项目学术价值与研究潜力的关键依据。
3. **研究基础和条件保障**：主要评价项目申报人的研究基础及可依托的科研平台情况，为项目顺利开展的可行性提供参考，该指标分值设定为10分。""",
                "tool": "",
                "response": "",
                "rout": "",  # 'rout' 的初始值为空，将由 router 节点决定
                "chart_image": ""
            }

            # 调用 Agent 的 invoke 方法
            # final_state 将是 Agent 执行到 END 节点时的最终状态
            final_state = agent.invoke(initial_state)

            # --- 5. 智能地展示结果 ---
            # 这是应用的核心逻辑：根据 final_state 的内容决定如何展示
            
            # 优先检查是否生成了图表
            if final_state and final_state.get("chart_image"):
                st.subheader("为您生成的图表：")
                
                # 首先，将 base64 字符串解码为 bytes
                image_bytes = base64.b64decode(final_state["chart_image"])
                # 然后，将 bytes 包装成一个 BytesIO 对象传递给 st.image
                st.image(io.BytesIO(image_bytes))
                # -------------------------
                # 如果有伴随的文字回复，也一并显示
                if final_state.get("response"):
                    st.markdown(final_state["response"])

            # 其次，检查是否有来自 RAG 流程的最终回答
            # (通常最终答案会放在 'response' 字段)
            elif final_state and final_state.get("response"):
                st.markdown(final_state["response"])

            # 再次，检查是否只有网络搜索结果
            # (web_search 节点只填充 'aggregated_context')
            elif final_state and final_state.get("aggregated_context"):
                st.subheader("网络搜索结果：")
                st.markdown(final_state["aggregated_context"])
            
            # 最后，如果出现意外情况，展示一个错误信息
            else:
                st.error("抱歉，Agent 执行完毕，但未能找到可供展示的结果。")
                # 为了方便调试，可以展示最终状态的原始数据
                st.json(final_state)

            # 将 Agent 的最终状态（或一个整理后的回复）也存入历史记录
            # 这里我们简单地将整个 final_state 存起来，实际应用中可以只存关键信息
            st.session_state.messages.append({"role": "assistant", "content": final_state})

# (可选) 展示历史聊天记录
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         # 这里需要更复杂的逻辑来重新渲染历史结果
#         st.write(message["content"])