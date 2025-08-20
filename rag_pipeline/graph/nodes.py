# rag_pipeline/graph/nodes.py
from pprint import pprint
from typing import List
from langchain_core.documents import Document

# Local application imports
from rag_pipeline.components import chains
from rag_pipeline.components.retrievers import (
    chunks_query_retriever,
    # ... other retrievers if needed ...
)
from rag_pipeline.graph.state import (
    MainGraphState, 
    FactualSubGraphState,
    AnalyticalSubGraphState,
    ToolUseSubGraphState
)
from rag_pipeline.utils.helpers import escape_quotes
from rag_pipeline.graph.workflows import factual_retrieval_workflow_app
from rag_pipeline.utils.web_search import create_web_search_tool

# --- Main Graph Nodes ---

def classify_query(state: MainGraphState):
    """
    Node that orchestrates the query classification.
    It's part of the main graph and receives the main state.
    """
    print("---CLASSIFYING QUERY---")
    state["curr_state"] = "classifying_query"
    
    classification_chain = chains.create_classification_chain()
    result = classification_chain.invoke({"question": state["question"]})
    query_type = result.query_type.strip().lower()
    
    print(f"---QUERY CLASSIFIED AS: {query_type}---")
    
    return {"query_type": query_type}

# --- Nodes for Factual Retrieval Sub-Graph ---

def enhance_query(state: FactualSubGraphState):
    """
    Node to enhance the user's query for better retrieval.
    Receives the FactualSubGraphState.
    """
    print("---ENHANCING QUERY---")
    
    question = state["question"]
    enhancement_chain = chains.create_query_enhancement_chain()
    result = enhancement_chain.invoke({"question": question})
    enhanced_question = result.enhanced_question
    
    print(f"---ENHANCED QUERY: {enhanced_question}---")
    
    return {"enhanced_question": enhanced_question}

def retrieve_documents(state: FactualSubGraphState):
    """
    Node to retrieve documents from the vector store.
    Receives the FactualSubGraphState.
    """
    print("---RETRIEVING DOCUMENTS---")
    
    query_to_use = state.get("enhanced_question") or state["question"]
    retriever = chunks_query_retriever 
    documents = retriever.get_relevant_documents(query_to_use)
    
    print(f"---RETRIEVED {len(documents)} DOCUMENTS---")
    
    return {"documents": documents}

def rerank_documents(state: FactualSubGraphState):
    """
    Node to re-rank retrieved documents for relevance.
    Receives the FactualSubGraphState.
    """
    print("---RERANKING DOCUMENTS---")
    
    question = state.get("enhanced_question") or state["question"]
    documents = state["documents"]
    
    reranking_chain = chains.create_reranking_chain()
    
    graded_documents = []
    for doc in documents:
        result = reranking_chain.invoke({"question": question, "document": doc.page_content})
        graded_documents.append((doc, result.score))
        print(f"  - Doc: ...{doc.page_content[:50].strip()}... | Score: {result.score}")

    graded_documents.sort(key=lambda x: x[1], reverse=True)
    
    relevance_threshold = 5.0
    final_documents = [doc for doc, score in graded_documents if score >= relevance_threshold]
    
    print(f"---FILTERED TO {len(final_documents)} DOCUMENTS (Threshold: >={relevance_threshold})---")
    
    return {
        "documents": final_documents,
        "graded_documents": graded_documents
    }

def generate_answer(state: FactualSubGraphState):
    """
    Node to generate the final answer based on the retrieved context.
    Receives the FactualSubGraphState.
    """
    print("---GENERATING ANSWER---")
    
    question = state["question"]
    documents = state["documents"]
    
    # Create context string
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    generation_chain = chains.create_generation_chain()
    result = generation_chain.invoke({"question": question, "context": context})
    
    answer = result.answer_based_on_content
    print(f"---FINAL ANSWER GENERATED---")
    
    return {"generation": answer, "context": context}

# --- Nodes for Analytical Sub-Graph ---

def generate_sub_queries(state: AnalyticalSubGraphState):
    """
    Node to decompose a complex question into sub-questions.
    Receives the AnalyticalSubGraphState.
    """
    print("---GENERATING SUB-QUERIES---")
    
    generation_chain = chains.create_sub_query_generation_chain()
    result = generation_chain.invoke({"question": state["question"]})
    
    sub_queries = result.sub_queries
    print(f"---DECOMPOSED INTO {len(sub_queries)} SUB-QUERIES---")
    
    return {"sub_queries": sub_queries}

# Placeholder for the node that will execute the factual sub-graph for each sub-query
def process_sub_queries(state: AnalyticalSubGraphState):
    """
    This node orchestrates running the factual sub-graph for each sub-query.
    It embodies the modular reuse principle by invoking the complete factual retrieval workflow
    for each decomposed sub-question.
    """
    print("---PROCESSING SUB-QUERIES---")
    
    sub_queries = state["sub_queries"]
    sub_query_results = {}
    
    print(f"Processing {len(sub_queries)} sub-queries...")
    
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"  Processing sub-query {i}/{len(sub_queries)}: {sub_query[:60]}...")
        
        # Create input state for the factual sub-graph
        factual_input = {
            "question": sub_query,
            "enhanced_question": None,
            "documents": [],
            "graded_documents": None,
            "context": "",
            "generation": ""
        }
        
        try:
            # Invoke the factual retrieval workflow for this sub-query
            factual_result = factual_retrieval_workflow_app.invoke(factual_input)
            
            # Extract the generated answer
            answer = factual_result.get("generation", "No answer generated")
            sub_query_results[sub_query] = answer
            
            print(f"    ✓ Sub-query {i} processed successfully")
            
        except Exception as e:
            print(f"    ✗ Error processing sub-query {i}: {str(e)}")
            sub_query_results[sub_query] = f"Error processing query: {str(e)}"
    
    print(f"---COMPLETED PROCESSING {len(sub_query_results)} SUB-QUERIES---")
    
    return {"sub_query_results": sub_query_results}

def synthesize_analytical_answer(state: AnalyticalSubGraphState):
    """
    Node to synthesize the final answer from the results of the sub-queries.
    Receives the AnalyticalSubGraphState.
    """
    print("---SYNTHESIZING ANALYTICAL ANSWER---")
    
    synthesis_chain = chains.create_analytical_synthesis_chain()
    result = synthesis_chain.invoke({
        "question": state["question"],
        "sub_query_results": state["sub_query_results"]
    })
    
    answer = result.answer_based_on_content
    print("---SYNTHESIS COMPLETE---")
    
    return {"generation": answer}

# --- Nodes for Tool Use Sub-Graph ---

def decide_tool_use(state: ToolUseSubGraphState):
    """
    Node to determine if external tools are needed to answer the question.
    Receives the ToolUseSubGraphState.
    """
    print("---DECIDING TOOL USE---")
    
    question = state["question"]
    decision_chain = chains.create_tool_decision_chain()
    result = decision_chain.invoke({"question": question})
    
    needs_tool = result.needs_external_tool
    print(f"---TOOL DECISION: {'NEEDED' if needs_tool else 'NOT NEEDED'} - {result.reasoning}---")
    
    return {"needs_tool": needs_tool}

def generate_search_query(state: ToolUseSubGraphState):
    """
    Node to generate an optimized search query for external tools.
    Receives the ToolUseSubGraphState.
    """
    print("---GENERATING SEARCH QUERY---")
    
    question = state["question"]
    query_chain = chains.create_search_query_generation_chain()
    result = query_chain.invoke({"question": question})
    
    search_query = result.search_query
    print(f"---SEARCH QUERY GENERATED: {search_query}---")
    
    return {"search_query": search_query}

def execute_web_search(state: ToolUseSubGraphState):
    """
    Node to execute web search using external tools.
    Receives the ToolUseSubGraphState.
    """
    print("---EXECUTING WEB SEARCH---")
    
    search_query = state["search_query"]
    search_tool = create_web_search_tool()
    
    try:
        search_results = search_tool.search(search_query)
        print(f"---SEARCH COMPLETED: {len(search_results)} results found---")
        
        return {"search_results": search_results}
        
    except Exception as e:
        print(f"---SEARCH FAILED: {str(e)}---")
        error_result = [{
            'title': '搜索失败',
            'snippet': f'无法获取外部信息：{str(e)}',
            'url': ''
        }]
        return {"search_results": error_result}

def generate_tool_use_answer(state: ToolUseSubGraphState):
    """
    Node to generate the final answer based on search results.
    Receives the ToolUseSubGraphState.
    """
    print("---GENERATING TOOL USE ANSWER---")
    
    question = state["question"]
    search_results = state["search_results"]
    
    # Format search results for prompt
    formatted_results = []
    for i, result in enumerate(search_results, 1):
        formatted_results.append(f"{i}. 标题: {result['title']}\n   内容: {result['snippet']}\n   来源: {result['url']}")
    
    search_results_text = "\n\n".join(formatted_results)
    
    answer_chain = chains.create_tool_use_answer_generation_chain()
    result = answer_chain.invoke({
        "question": question, 
        "search_results": search_results_text
    })
    
    answer = result.answer_based_on_content
    print("---TOOL USE ANSWER GENERATED---")
    
    return {"generation": answer}