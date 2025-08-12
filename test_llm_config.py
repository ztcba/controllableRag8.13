#!/usr/bin/env python3
"""
Test script to verify that LLM configuration and model instantiation works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to Python path so we can import from our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_settings_loading():
    """Test that settings can be loaded correctly from .env file."""
    print("Testing settings loading...")
    try:
        from src.rag_pipeline.settings import settings
        print(f"  - default_model: {settings.default_model}")
        print(f"  - llm_provider: {settings.llm_provider}")
        print(f"  - llm_base_url: {settings.llm_base_url}")
        print(f"  - default_temperature: {settings.default_temperature}")
        print("  ✓ Settings loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load settings: {e}")
        return False

def test_model_instantiation():
    """Test that LLM models can be instantiated."""
    print("\nTesting LLM model instantiation...")
    try:
        from src.rag_pipeline.components.llms import get_chat_model, get_planner_model
        
        # Test get_chat_model
        print("  - Testing get_chat_model...")
        chat_model = get_chat_model()
        print(f"    ✓ Chat model instantiated: {type(chat_model).__name__}")
        
        # Test get_planner_model
        print("  - Testing get_planner_model...")
        planner_model = get_planner_model()
        print(f"    ✓ Planner model instantiated: {type(planner_model).__name__}")
        
        # Check if both models are the same type (they should be in the default case)
        if type(chat_model) == type(planner_model):
            print(f"    ✓ Both models are of the same type: {type(chat_model).__name__}")
        else:
            print(f"    ! Models are of different types: {type(chat_model).__name__} vs {type(planner_model).__name__}")
            
        return True
    except Exception as e:
        print(f"  ✗ Failed to instantiate models: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_file():
    """Test that .env file exists and has basic content."""
    print("\nTesting .env file...")
    try:
        env_path = Path(__file__).parent / "src" / ".env"
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("  - .env file content:")
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        print(f"    {line}")
                print("  ✓ .env file exists and was read successfully")
                return True
        else:
            print("  ✗ .env file not found")
            return False
    except Exception as e:
        print(f"  ✗ Failed to read .env file: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("LLM Configuration Test Script")
    print("=" * 50)
    
    tests = [
        test_env_file,
        test_settings_loading,
        test_model_instantiation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLM configuration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())