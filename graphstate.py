from typing_extensions import TypedDict
from typing import List
from typing import Literal

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
    rout: Literal["rout_is_retrieve", "rout_is_websearch", "rout_is_draw", ""]
    # 增加一个字段来存储 base64 编码的图表图片
    chart_image: str