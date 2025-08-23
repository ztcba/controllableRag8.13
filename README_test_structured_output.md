# 结构化输出测试脚本使用说明

## 概述
这个脚本用于测试你的模型提供商是否支持 LangChain 的结构化输出功能。

## 使用方法

1. **确保环境配置正确**：
   ```bash
   # 确保.env文件包含以下配置
   OPENAI_API_KEY=your_api_key
   OPENAI_API_BASE=your_api_base_url  # 可选
   MODEL_NAME=gpt-4  # 可选，默认gpt-4
   TEMPERATURE=0.0   # 可选，默认0.0
   MAX_TOKENS=2000   # 可选，默认2000
   ```

2. **运行测试脚本**：
   ```bash
   python test_structured_output.py
   ```

## 测试内容

脚本会测试以下几个方面：

### 1. 基础测试
- **function_calling方法**: 使用OpenAI的function calling功能
- **json_mode方法**: 使用JSON模式
- **默认方法**: 让LangChain自动选择最佳方法

### 2. 测试模型
- **简单模型**: 包含布尔值、浮点数、字符串字段
- **事实检查模型**: 专门用于检查答案是否基于事实

### 3. 实际场景测试
- 模拟真实的事实依据检查场景
- 测试中文内容处理能力

## 期望输出

### 成功的情况
```
✅ 测试成功! 类型: <class '__main__.TestModel'>
📄 结果: TestModel(result=True, confidence=1.0, explanation='2+2 equals 4 is mathematically correct')
```

### 失败的情况
```
❌ 返回 None - function_calling 不支持
❌ 错误: Failed to parse TestModel from completion
```

## 根据测试结果调整代码

### 如果function_calling可用
```python
chain = prompt | llm.with_structured_output(YourModel, method="function_calling")
```

### 如果只有json_mode可用
```python
chain = prompt | llm.with_structured_output(YourModel, method="json_mode")
```

### 如果都不可用，使用JsonOutputParser
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=YourModel)
prompt_with_instructions = PromptTemplate(
    template=your_template + "\n{format_instructions}",
    input_variables=your_variables,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt_with_instructions | llm | parser
```

## 常见问题

1. **ModuleNotFoundError**: 确保安装了所需依赖
   ```bash
   pip install langchain langchain-openai pydantic python-dotenv
   ```

2. **API错误**: 检查.env文件中的API配置是否正确

3. **返回None**: 说明你的模型提供商不支持structured_output，需要使用JsonOutputParser
