🧪 测试 简单模型 - 默认方法:
  ✅ 成功! 类型: <class '__main__.TestModel'>
  📄 结果: result=True confidence=1.0 explanation='2+2 equals 4 by basic arithmetic addition.'

============================================================
🧪 测试 2: 事实依据检查模型

🧪 测试 事实检查模型 - function_calling方法:
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持
  ❌ 返回 None - function_calling 不支持

🧪 测试 事实检查模型 - json_mode方法:
  ❌ 错误: Failed to parse GroundedFactsTest from completion {"question": "Is 2+2 equal to 4?", "answer": true, "explanation": "Yes, 2 plus 2 equals 4 in sandard arithmetic."}. Got: 2 validation errors for GroundedFactsTest
tandard arithmetic."}. Got: 2 validation errors for GroundedFactsTest
grounded_on_facts
  Field required [type=missing, input_value={'question': 'Is 2+2 equa...n standard arithmetic.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/missing
reasoning
  Field required [type=missing, input_value={'question': 'Is 2+2 equa...n standard arithmetic.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/missing
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE

🧪 测试 事实检查模型 - 默认方法:
  ✅ 成功! 类型: <class '__main__.GroundedFactsTest'>
  📄 结果: grounded_on_facts=True reasoning='According to basic arithmetic, the sum of 2 and 2 is 4. This is a well-established mathematical fact.'       

============================================================
🧪 测试 3: 实际应用场景

🎯 测试实际场景 - 事实依据检查:

  📝 测试方法: function_calling
    ❌ function_calling: 返回 None

  📝 测试方法: json_mode
    ❌ json_mode: 错误 - <html>
<head><title>504 Gateway Time-out</title></head>
<body>
<center><h1>504 Gateway Time-out</h1></center>  
<hr><center>nginx</center>
</body>
</html>

  📝 测试方法: default
    ✅ default: 成功!
    📄 grounded_on_facts: True
    📄 reasoning: 答案明确指出申请补贴的最后一天是2025年8月31日，这与上下文中“补贴政策的申请截止日期是2025年8月31日”完全一致，因此答案与事实相符。

============================================================
📊 测试结果总结:
============================================================
  simple_function_calling  : ❌ 不支持
  simple_json_mode         : ❌ 不支持
  simple_default           : ✅ 支持
  facts_function_calling   : ❌ 不支持
  facts_json_mode          : ❌ 不支持
  facts_default            : ✅ 支持

🎯 推荐的方法:
  ❌ 该模型/API不支持结构化输出
  💡 建议: 使用 JsonOutputParser 手动解析JSON输出

💡 解决方案建议:
  ✅ 以下方法可用: simple_default, facts_default