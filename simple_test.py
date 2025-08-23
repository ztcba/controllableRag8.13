#!/usr/bin/env python3
"""
简单的结构化输出测试脚本
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载.env文件
load_dotenv()

# 定义简单的测试模型
class SimpleTest(BaseModel):
    answer: bool = Field(description="True or False")
    explanation: str = Field(description="Brief explanation")

def main():
    print("🧪 测试模型结构化输出支持...")
    
    # 从.env获取配置
    api_key = os.getenv('XIAOCASEAI_API_KEY')
    api_base = os.getenv('LLM_BASE_URL') 
    model_name = os.getenv('CHAT_MODEL', 'gpt-4')
    
    if not api_key:
        print("❌ 未找到 XIAOCASEAI_API_KEY")
        return
    
    print(f"📱 API Base: {api_base}")
    print(f"📱 Model: {model_name}")
    
    # 创建LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model_name=model_name,
        temperature=0
    )
    
    # 简单prompt
    prompt = PromptTemplate(
        template="Is 2+2 equal to 4? Answer with true/false and explain briefly.",
        input_variables=[]
    )


    chain = prompt | llm.with_structured_output(SimpleTest)
            
    result = chain.invoke({})
            
    if result is None:
        print(f"❌ {name}: 返回 None")
    else:
        print(f"✅ 成功!")
        print(f"   答案: {result.answer}")
        print(f"   解释: {result.explanation}")
                


if __name__ == "__main__":
    main()
