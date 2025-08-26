from setuptools import setup, find_packages

setup(
    name='document-retrieval-system',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A document processing and retrieval system using FAISS and BM25.',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'langchain_openai',
        'langchain_community',
        'faiss-cpu',
        'loguru',
        'numpy',
        'pandas',
        'scikit-learn',
        'markdown',
    ],
    entry_points={
        'console_scripts': [
            'ingest=rag_pipeline.ingest:main',
        ],
    },
)