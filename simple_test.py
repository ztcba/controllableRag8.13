import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("=== LLM Configuration Test ===")

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
    
    chat_model = get_chat_model()
    print(f"   ✓ Chat model instantiated: {type(chat_model).__name__}")
    
    planner_model = get_planner_model()
    print(f"   ✓ Planner model instantiated: {type(planner_model).__name__}")
    
    print("\n🎉 All configuration tests passed!")
    print("You can now use the LLM in your application.")
    
except Exception as e:
    print(f"   ✗ Failed to instantiate models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)