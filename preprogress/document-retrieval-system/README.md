# Document Retrieval System

This project implements a document processing and retrieval system that utilizes FAISS for vector storage and retrieval, combined with BM25 for enhanced search capabilities.

## Project Structure

- **rag_pipeline/**: Contains the core components of the system.
  - **components/**: Includes modules for document processing and retrieval.
    - **llms.py**: Functions to retrieve chat and embedding models.
    - **retriever_factory.py**: Creates a hybrid retriever combining BM25 and vector search.
    - **document_processor.py**: Processes documents for ingestion.
  - **utils/**: Utility functions for text processing.
    - **text_splitter.py**: Functions for splitting text while preserving structure.
  - **ingest.py**: Reads Markdown files from the `data` directory, processes them, and creates a FAISS vector database.
  - **settings.py**: Configuration settings for the project.

- **data/**: Directory for storing raw and processed data.
  - **raw/**: Contains raw data files.
  - **processed/**: Contains processed data files.

- **notebooks/**: Jupyter notebooks demonstrating usage of the system.

- **tests/**: Unit tests for the project components.
  - **test_ingest.py**: Tests for the ingest functionality.
  - **test_retriever.py**: Tests for the retriever functionality.

- **.env.example**: Example environment variable configuration.

- **.gitignore**: Specifies files and directories to ignore in version control.

- **requirements.txt**: Lists the required Python packages for the project.

- **setup.py**: Contains metadata and dependencies for packaging the project.

## Features

- **Document Ingestion**: Automatically processes Markdown files and creates a FAISS vector database.
- **Hybrid Retrieval**: Combines BM25 and vector search for efficient document retrieval.
- **Configurable**: Easily adjustable settings for models and parameters through `settings.py`.
- **Logging**: Utilizes `loguru` for logging important events and errors.

## Usage

1. Set up your environment by copying `.env.example` to `.env` and filling in the required variables.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run `ingest.py` to process documents and create the vector database.
4. Use `retriever_factory.py` to create a retriever instance for querying documents.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.