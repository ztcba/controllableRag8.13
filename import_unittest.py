# import unittest.py

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from functions_for_pipeline import create_agent, END
from typing_extensions import TypedDict
from typing import List

# test_functions_for_pipeline.py


class TestCreateAgent(unittest.TestCase):
    """Test cases for the create_agent function in functions_for_pipeline.py"""

    @patch('functions_for_pipeline.StateGraph')
    def test_agent_creation(self, mock_state_graph):
        """Test that the agent can be created successfully"""
        # Setup mock
        mock_instance = MagicMock()
        mock_state_graph.return_value = mock_instance
        mock_compiled = MagicMock()
        mock_instance.compile.return_value = mock_compiled

        # Call the function
        result = create_agent()
        
        # Verify the function returned the compiled graph
        self.assertEqual(result, mock_compiled)
        
        # Verify StateGraph was initialized with PlanExecute
        mock_state_graph.assert_called_once()
        
        # Verify the entry point was set to "planner"
        mock_instance.set_entry_point.assert_called_with("planner")
        
        # Verify that all expected nodes were added
        expected_nodes = [
            "planner", "break_down_plan", "retrieve_chunks", "retrieve_summaries",
            "answer", "task_handler", "replan", "get_final_answer"
        ]
        
        # Count node additions
        node_calls = mock_instance.add_node.call_count
        self.assertEqual(node_calls, len(expected_nodes), 
                         f"Expected {len(expected_nodes)} nodes, but {node_calls} were added")
        
        # Verify the graph was compiled
        mock_instance.compile.assert_called_once()

    def test_agent_integration(self):
        """Integration test to check if the agent can be created without errors"""
        try:
            agent = create_agent()
            self.assertIsNotNone(agent)
            # Check the agent has expected methods of a compiled graph
            self.assertTrue(hasattr(agent, 'invoke'))
            self.assertTrue(hasattr(agent, 'get_graph'))
        except Exception as e:
            self.fail(f"create_agent() raised an exception: {e}")

    @patch('functions_for_pipeline.run_qualitative_chunks_retrieval_workflow')
    @patch('functions_for_pipeline.run_qualitative_summaries_retrieval_workflow')
    @patch('functions_for_pipeline.run_qualtative_answer_workflow')
    @patch('functions_for_pipeline.run_qualtative_answer_workflow_for_final_answer')
    @patch('functions_for_pipeline.plan_step')
    @patch('functions_for_pipeline.break_down_plan_step')
    @patch('functions_for_pipeline.run_task_handler_chain')
    @patch('functions_for_pipeline.replan_step')
    @patch('functions_for_pipeline.retrieve_or_answer')
    @patch('functions_for_pipeline.can_be_answered')
    def test_graph_structure(self, mock_can_be_answered, mock_retrieve_or_answer, *args):
        """Test the structure of the created graph"""
        # 配置模拟对象
        mock_retrieve_or_answer.__name__ = 'retrieve_or_answer'
        mock_can_be_answered.__name__ = 'can_be_answered'
        mock_retrieve_or_answer.return_value = "chosen_tool_is_retrieve_chunks"
        mock_can_be_answered.return_value = "can_be_answered_already"
        
        agent = create_agent()
        graph = agent.get_graph()
        
        # 不能直接访问entry_point，改为验证图的边和节点
        edges = graph.edges
        nodes = graph.nodes
        
        # 验证重要节点是否存在
        expected_nodes = [
            "planner", "break_down_plan", "retrieve_chunks", "retrieve_summaries", 
            "answer", "task_handler", "replan", "get_final_answer"
        ]
        for node in expected_nodes:
            self.assertIn(node, nodes, f"Node {node} missing from graph")
        
        # 验证关键边是否存在
        expected_edges = [
            ("planner", "break_down_plan"),
            ("break_down_plan", "task_handler"),
            ("retrieve_chunks", "replan"),
            ("retrieve_summaries", "replan"),
            ("answer", "replan")
        ]
        for edge in expected_edges:
            self.assertIn(edge, edges, f"Edge {edge} missing from graph")
        
        # 检查END节点的连接
        end_connections = [edge for edge in edges if edge[1] == END]
        self.assertTrue(len(end_connections) > 0, "No connections to END node found")

if __name__ == '__main__':
    unittest.main()