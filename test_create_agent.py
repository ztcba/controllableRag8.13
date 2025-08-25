#!/usr/bin/env python3
"""
测试 create_agent 函数返回的 agent 是否能正常工作
"""

import sys
import os
from pathlib import Path
from pprint import pprint

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_setup():
    """测试环境配置"""
    print("=== 测试环境配置 ===")
    
    # 1. 测试环境变量加载
    print("\n1. 测试环境变量...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            print(f"   ✓ OPENAI_API_KEY 已加载: {openai_key[:10]}...")
        else:
            print("   ✗ OPENAI_API_KEY 未找到")
            
        llm_provider = os.getenv('LLM_PROVIDER')
        print(f"   ✓ LLM_PROVIDER: {llm_provider}")
        
        return True
    except Exception as e:
        print(f"   ✗ 环境变量加载失败: {e}")
        return False

def test_vector_stores():
    """测试向量数据库是否存在"""
    print("\n2. 测试向量数据库...")
    
    chunks_path = project_root / "chunks_vector_store"
    summaries_path = project_root / "chapter_summaries_vector_store"
    
    if chunks_path.exists() and (chunks_path / "index.faiss").exists():
        print(f"   ✓ chunks_vector_store 存在")
    else:
        print(f"   ✗ chunks_vector_store 不存在或不完整")
        return False
        
    if summaries_path.exists() and (summaries_path / "index.faiss").exists():
        print(f"   ✓ chapter_summaries_vector_store 存在")
    else:
        print(f"   ✗ chapter_summaries_vector_store 不存在或不完整")
        return False
    
    return True

def test_imports():
    """测试导入依赖"""
    print("\n3. 测试导入依赖...")
    
    try:
        # 测试基本导入
        from functions_for_pipeline import create_agent
        print("   ✓ create_agent 导入成功")
        
        # 测试 LangChain 导入
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        print("   ✓ LangChain 组件导入成功")
        
        # 测试 LangGraph 导入
        from langgraph.graph import StateGraph, END
        print("   ✓ LangGraph 组件导入成功")
        
        return True
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False

def test_create_agent():
    """测试 create_agent 函数"""
    print("\n4. 测试 create_agent 函数...")
    
    try:
        from functions_for_pipeline import create_agent
        
        # 创建 agent
        print("   正在创建 agent...")
        agent = create_agent()
        
        if agent is None:
            print("   ✗ agent 创建失败 - 返回 None")
            return False
            
        print(f"   ✓ agent 创建成功，类型: {type(agent)}")
        
        # 检查 agent 是否有必要的方法
        if hasattr(agent, 'invoke'):
            print("   ✓ agent 具有 invoke 方法")
        else:
            print("   ✗ agent 缺少 invoke 方法")
            return False
            
        if hasattr(agent, 'stream'):
            print("   ✓ agent 具有 stream 方法")
        else:
            print("   ⚠ agent 缺少 stream 方法（可能正常）")
        
        return agent
    except Exception as e:
        print(f"   ✗ create_agent 失败: {e}")
        import traceback
        print("   详细错误信息:")
        print("   " + "\n   ".join(traceback.format_exc().splitlines()))
        return False

def test_agent_simple_question(agent):
    """测试 agent 处理简单问题"""
    print("\n5. 测试 agent 处理简单问题...")
    
    try:
        # 简单的测试问题
        test_question = "中国生育保险参保情况如何？"
        
        print(f"   测试问题: {test_question}")
        print("   正在运行 agent...")
        
        # 创建输入状态
        initial_state = {
            "question": test_question,
            "curr_state": "",
            "query_to_retrieve_or_answer": "",
            "plan": [],
            "past_steps": [],
            "curr_context": "",
            "aggregated_context": "",
            "tool": "",
            "response": ""
        }
        
        # 调用 agent
        result = agent.invoke(initial_state)
        
        print("   ✓ agent 成功处理问题")
        print(f"   最终状态: {result.get('curr_state', 'unknown')}")
        
        if 'response' in result and result['response']:
            print(f"   回答: {result['response'][:200]}...")
            return True
        else:
            print("   ⚠ agent 运行完成但没有生成回答")
            return False
            
    except Exception as e:
        print(f"   ✗ agent 处理问题失败: {e}")
        import traceback
        print("   详细错误信息:")
        print("   " + "\n   ".join(traceback.format_exc().splitlines()))
        return False

def main():
    """主测试函数"""
    print("开始测试 create_agent 函数...")
    
    # 环境设置测试
    if not test_environment_setup():
        print("\n❌ 环境配置测试失败，无法继续")
        return
    
    # 向量数据库测试  
    if not test_vector_stores():
        print("\n❌ 向量数据库测试失败，无法继续")
        return
    
    # 导入测试
    if not test_imports():
        print("\n❌ 导入测试失败，无法继续")
        return
    
    # create_agent 测试
    agent = test_create_agent()
    if not agent:
        print("\n❌ create_agent 测试失败")
        return
    
    # 简单问题测试
    if test_agent_simple_question(agent):
        print("\n✅ 所有测试通过！create_agent 函数返回的 agent 工作正常")
    else:
        print("\n⚠ agent 创建成功但处理问题时出现问题")

if __name__ == "__main__":
    main()
