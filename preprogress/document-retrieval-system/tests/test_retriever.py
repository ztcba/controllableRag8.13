from rag_pipeline.components.retriever_factory import create_retriever
import unittest

class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.vector_store_path = "path/to/vector_store"  # Update with actual path
        self.embedding_model = "your_embedding_model"  # Update with actual model
        self.reorder = True  # Set to True or False based on test case
        self.num_docs = 5  # Number of documents to return

    def test_create_retriever(self):
        retriever = create_retriever(
            vector_store_path=self.vector_store_path,
            embedding_model=self.embedding_model,
            reorder=self.reorder,
            num_docs=self.num_docs
        )
        self.assertIsNotNone(retriever)
        self.assertTrue(hasattr(retriever, 'retrieve'))  # Check if retrieve method exists

    def test_retriever_functionality(self):
        retriever = create_retriever(
            vector_store_path=self.vector_store_path,
            embedding_model=self.embedding_model,
            reorder=self.reorder,
            num_docs=self.num_docs
        )
        query = "Sample query to test retrieval"
        results = retriever.retrieve(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # Ensure some results are returned

if __name__ == '__main__':
    unittest.main()