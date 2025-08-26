# 测试 create_agent 函数
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_create_agent():
    """测试 create_agent 函数是否能正常工作"""
    
    try:
        print("🔧 开始测试 create_agent 函数...")
        
        # 导入函数
        from function_temp import create_agent
        print("✅ 成功导入 create_agent 函数")
        
        # 创建 agent
        print("🚀 创建 agent...")
        agent = create_agent()
        print(f"✅ 成功创建 agent，类型: {type(agent)}")
        
        # 测试一个简单的查询
        test_query = "2025中国博士后科学基金“面上资助”项目专家评审时各项指标的权重分布如何？"
        print(f"🔍 测试查询: {test_query}")
        
        # 创建初始状态
        initial_state = {
            "curr_state": "",
            "question": test_query,
            "query_to_retrieve_or_answer": "",
            "plan": [],
            "past_steps": [],
            "curr_context": "",
            "aggregated_context": "",
            "tool": "",
            "response": ""
        }
        
        print("📊 初始状态已创建")
        print("🏃‍♂️ 开始执行 agent...")
        
        # 执行 agent
        result = None
        step_count = 0
        max_steps = 10  # 限制最大步数，避免无限循环
        
        for output in agent.stream(initial_state):
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            for key, value in output.items():
                print(f"节点: {key}")
                if isinstance(value, dict):
                    current_state = value.get('curr_state', 'unknown')
                    print(f"当前状态: {current_state}")
                    
                    # 显示一些关键信息
                    if 'plan' in value and value['plan']:
                        print(f"计划步骤: {value['plan'][:2]}...")  # 只显示前两个步骤
                    if 'aggregated_context' in value and value['aggregated_context']:
                        context_preview = value['aggregated_context'][:200] + "..." if len(value['aggregated_context']) > 200 else value['aggregated_context']
                        print(f"聚合上下文: {context_preview}")
                    if 'response' in value and value['response']:
                        print(f"响应: {value['response']}")
                        result = value['response']
                        
            # 安全退出机制
            if step_count >= max_steps:
                print(f"\n⚠️ 达到最大步数限制 ({max_steps})，停止执行")
                break
                
        if result:
            print(f"\n🎉 Agent 执行完成！")
            print(f"最终结果: {result}")
        else:
            print(f"\n⚠️ Agent 执行完成但没有得到最终结果")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试 create_agent 函数")
    print("=" * 60)
    
    success = test_create_agent()
    
    if success:
        print("\n✅ 测试通过！create_agent 函数工作正常")
    else:
        print("\n❌ 测试失败！需要检查问题")
