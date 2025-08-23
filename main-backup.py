# 将导入 create_agent 函数，编译 Agent，然后用一个示例问题来调用它
# main.py
import uuid
from pprint import pprint
from rag_pipeline.graph.agent import create_agent
from rag_pipeline.utils.helpers import text_wrap

def run_agent():
    """
    Initializes and runs the RAG agent with a sample question.
    """
    # 1. Create the agent
    # The agent is compiled when this function is called.
    agent_app = create_agent()
    print("Agent created successfully.")

    # 2. Define the inputs for the agent
    # The input must match the structure of the PlanExecute state
    question = "What are the main arguments of the book 'The Black Swan' by Nassim Nicholas Taleb, and how does the author criticize traditional risk management models?"
    
    initial_state = {
        "question": question,
        "anonymized_question": "",
        "mapping": {},
        "plan": [],
        "past_steps": [],
        "curr_state": "",
        "query_to_retrieve_or_answer": "",
        "curr_context": "",
        "aggregated_context": "",
        "tool": "",
        "response": ""
    }
    
    # 3. Define the configuration for the run
    # The `thread_id` is essential for LangGraph to manage the state
    # of a conversation or run. Each run should have a unique ID.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n=============================================")
    print(f"Running agent with question:\n{text_wrap(question)}")
    print("=============================================\n")
    
    # 4. Stream the agent's execution
    # Streaming allows us to see the output of each node as it runs.
    for event in agent_app.stream(initial_state, config=config):
        pprint(event)
        print("\n---\n")

    # 5. Get the final response
    # The final state is available after the stream is complete.
    final_state = agent_app.get_state(config)
    final_response = final_state.values()[0]['response']
    
    print("\n=============================================")
    print("Agent execution finished.")
    print(f"Final Answer:\n{text_wrap(final_response)}")
    print("=============================================\n")


if __name__ == "__main__":
    # This makes the script executable.
    run_agent()