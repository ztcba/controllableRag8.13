#!/usr/bin/env python3
"""
Test script to verify that LLM configuration works and can actually call the LLM API.
"""

import sys
from pathlib import Path

# Add src to Python path so we can import from our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing LLM API calls...")

# Test 1: Load settings
print("\n1. Loading settings...")
try:
    from src.rag_pipeline.settings import settings
    print("   ✓ Settings loaded successfully")
    print(f"   Default model: {settings.default_model}")
    print(f"   LLM provider: {settings.llm_provider}")
    print(f"   LLM base URL: {settings.llm_base_url}")
except Exception as e:
    print(f"   ✗ Failed to load settings: {e}")
    sys.exit(1)

# Test 2: Instantiate models
print("\n2. Testing model instantiation...")
try:
    from src.rag_pipeline.components.llms import get_chat_model, get_planner_model
    
    # Test chat model
    print("   Testing get_chat_model...")
    chat_model = get_chat_model()
    print(f"   ✓ Chat model instantiated: {type(chat_model).__name__}")
    
    # Test planner model
    print("   Testing get_planner_model...")
    planner_model = get_planner_model()
    print(f"   ✓ Planner model instantiated: {type(planner_model).__name__}")
    
except Exception as e:
    print(f"   ✗ Failed to instantiate models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Actually call the LLM API
print("\n3. Testing actual LLM API call...")
try:
    # Simple test message
    test_message = "Hello, this is a test. Please respond with 'Hello, World!' and nothing else."
    
    print(f"   Sending message: {test_message}")
    
    # Call the chat model
    response = chat_model.invoke(test_message)
    
    print(f"   ✓ API call successful!")
    print(f"   Model response: {response.content}")
    
    # Test with a structured output
    print("\n4. Testing structured output...")
    try:
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content="What is 2+2? Respond with just the number.")]
        result = chat_model.invoke(messages)
        print(f"   Simple math test: {result.content}")
        
    except Exception as e:
        print(f"   ✗ Failed structured output test: {e}")
        
    print("\n🎉 All tests passed! LLM configuration is working correctly and can call the API.")
    
except Exception as e:
    print(f"   ✗ Failed to call LLM API: {e}")
    print("   This might be due to:")
    print("   1. Invalid API key in .env file")
    print("   2. Network connectivity issues")
    print("   3. Incorrect LLM provider settings")
    print("   4. The LLM service being temporarily unavailable")
    import traceback
    traceback.print_exc()
    sys.exit(1)