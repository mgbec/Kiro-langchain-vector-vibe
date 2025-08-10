#!/usr/bin/env python3
"""
LangChain Vector Database Example
Creates a vector database using ChromaDB and demonstrates basic operations.
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

class VectorDatabase:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize the vector database with HuggingFace embeddings."""
        self.persist_directory = persist_directory
        
        # Initialize embeddings (using free HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize or load existing vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, texts, metadatas=None):
        """Add documents to the vector database."""
        # Create Document objects
        documents = [Document(page_content=text) for text in texts]
        
        # Add metadata if provided
        if metadatas:
            for doc, metadata in zip(documents, metadatas):
                doc.metadata = metadata
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"Added {len(split_docs)} document chunks to the vector database.")
        return len(split_docs)
    
    def similarity_search(self, query, k=4):
        """Search for similar documents."""
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query, k=4):
        """Search for similar documents with similarity scores."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def persist(self):
        """Persist the vector database to disk."""
        self.vectorstore.persist()
        print(f"Vector database persisted to {self.persist_directory}")


def main():
    # Initialize vector database
    vdb = VectorDatabase()
    
    # Sample documents to add
    sample_texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "Vector databases store high-dimensional vectors and enable similarity search.",
        "ChromaDB is an open-source embedding database that makes it easy to build LLM apps.",
        "Machine learning models can convert text into numerical representations called embeddings.",
        "Retrieval-augmented generation (RAG) combines information retrieval with text generation.",
        "Python is a popular programming language for data science and machine learning.",
        "Natural language processing involves analyzing and understanding human language.",
        "Embeddings capture semantic meaning and enable similarity comparisons between texts."
    ]
    
    # Sample metadata
    sample_metadata = [
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "vector_db_guide", "topic": "database"},
        {"source": "chromadb_docs", "topic": "database"},
        {"source": "ml_textbook", "topic": "embeddings"},
        {"source": "rag_paper", "topic": "generation"},
        {"source": "python_guide", "topic": "programming"},
        {"source": "nlp_textbook", "topic": "nlp"},
        {"source": "embeddings_guide", "topic": "embeddings"}
    ]
    
    # Add documents to the database
    print("Adding documents to vector database...")
    vdb.add_documents(sample_texts, sample_metadata)
    
    # Persist the database
    vdb.persist()
    
    # Example queries
    queries = [
        "What is LangChain?",
        "How do vector databases work?",
        "Tell me about embeddings"
    ]
    
    print("\n" + "="*50)
    print("SIMILARITY SEARCH EXAMPLES")
    print("="*50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Search with scores
        results = vdb.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Content: {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")
            print()


if __name__ == "__main__":
    main()