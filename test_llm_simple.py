#!/usr/bin/env python3
"""
Simple test script to verify that LLM configuration and model instantiation works correctly.
"""

import sys
from pathlib import Path

# Add src to Python path so we can import from our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing LLM configuration...")

# Test 1: Check if .env file exists
print("\n1. Checking .env file...")
env_file = src_path / ".env"
if env_file.exists():
    print("   ✓ .env file exists")
    # Show some contents
    with open(env_file, 'r') as f:
        lines = f.readlines()
        for line in lines[:5]:  # Show first 5 lines
            if line.strip() and not line.startswith('#'):
                print(f"   Content: {line.strip()}")
else:
    print("   ✗ .env file not found")

# Test 2: Try to load settings
print("\n2. Loading settings...")
try:
    from src.rag_pipeline.settings import settings
    print("   ✓ Settings loaded successfully")
    print(f"   Default model: {settings.default_model}")
    print(f"   LLM provider: {settings.llm_provider}")
    print(f"   LLM base URL: {settings.llm_base_url}")
except Exception as e:
    print(f"   ✗ Failed to load settings: {e}")
    sys.exit(1)

# Test 3: Try to instantiate models
print("\n3. Testing model instantiation...")
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
    
    print("\n🎉 All tests passed! LLM configuration is working correctly.")
    
except Exception as e:
    print(f"   ✗ Failed to instantiate models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)