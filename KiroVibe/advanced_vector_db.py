#!/usr/bin/env python3
"""
Advanced Vector Database Example with LangChain
Demonstrates more advanced features like document loading, different embedding models, and querying.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document

load_dotenv()

class AdvancedVectorDatabase:
    def __init__(self, 
                 persist_directory="./advanced_chroma_db",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize advanced vector database with configurable embeddings."""
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize text splitter with better parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def load_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*.txt"):
        """Load documents from a directory."""
        loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=TextLoader)
        documents = loader.load()
        return documents
    
    def load_document_from_file(self, file_path: str):
        """Load a single document from file."""
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents
    
    def add_documents_from_files(self, file_paths: List[str]):
        """Add documents from multiple files."""
        all_documents = []
        for file_path in file_paths:
            try:
                docs = self.load_document_from_file(file_path)
                all_documents.extend(docs)
                print(f"Loaded document from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if all_documents:
            split_docs = self.text_splitter.split_documents(all_documents)
            self.vectorstore.add_documents(split_docs)
            print(f"Added {len(split_docs)} document chunks from {len(all_documents)} files.")
            return len(split_docs)
        return 0
    
    def add_text_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add text documents with optional metadata."""
        documents = [Document(page_content=text) for text in texts]
        
        if metadatas:
            for doc, metadata in zip(documents, metadatas):
                doc.metadata.update(metadata)
        
        split_docs = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(split_docs)
        return len(split_docs)
    
    def search_by_similarity(self, query: str, k: int = 5, filter_dict: Optional[dict] = None):
        """Search with optional metadata filtering."""
        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_scores_and_threshold(self, query: str, k: int = 5, score_threshold: float = 0.7):
        """Search with similarity score threshold."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Filter by score threshold (lower scores = more similar)
        filtered_results = [(doc, score) for doc, score in results if score <= score_threshold]
        return filtered_results
    
    def get_collection_stats(self):
        """Get statistics about the vector database."""
        collection = self.vectorstore._collection
        count = collection.count()
        return {"document_count": count}
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.vectorstore.delete_collection()
        print("Collection deleted.")
    
    def persist(self):
        """Persist the database."""
        self.vectorstore.persist()
        print(f"Database persisted to {self.persist_directory}")


def create_sample_documents():
    """Create sample documents for testing."""
    documents = {
        "python_basics.txt": """
Python is a high-level programming language known for its simplicity and readability.
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
Python has a vast ecosystem of libraries and frameworks that make it suitable for various applications.
""",
        "machine_learning.txt": """
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.
Popular Python libraries for machine learning include scikit-learn, TensorFlow, and PyTorch.
Supervised learning, unsupervised learning, and reinforcement learning are the main types of machine learning.
""",
        "web_development.txt": """
Web development involves creating websites and web applications.
Python frameworks like Django and Flask are popular choices for backend web development.
Frontend technologies include HTML, CSS, and JavaScript, while backend involves server-side programming.
"""
    }
    
    # Create sample files
    for filename, content in documents.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
    
    return list(documents.keys())


def main():
    print("Advanced Vector Database Example")
    print("=" * 40)
    
    # Create sample documents
    sample_files = create_sample_documents()
    print(f"Created sample files: {sample_files}")
    
    # Initialize advanced vector database
    vdb = AdvancedVectorDatabase()
    
    # Add documents from files
    print("\nAdding documents from files...")
    chunks_added = vdb.add_documents_from_files(sample_files)
    
    # Add some additional text documents
    additional_texts = [
        "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
        "Semantic search uses vector embeddings to find documents based on meaning rather than exact keyword matches.",
        "RAG (Retrieval-Augmented Generation) combines vector search with language models for better responses."
    ]
    
    additional_metadata = [
        {"category": "database", "difficulty": "intermediate"},
        {"category": "search", "difficulty": "advanced"},
        {"category": "ai", "difficulty": "advanced"}
    ]
    
    print("\nAdding additional text documents...")
    vdb.add_text_documents(additional_texts, additional_metadata)
    
    # Persist the database
    vdb.persist()
    
    # Get collection stats
    stats = vdb.get_collection_stats()
    print(f"\nDatabase stats: {stats}")
    
    # Example searches
    print("\n" + "="*50)
    print("ADVANCED SEARCH EXAMPLES")
    print("="*50)
    
    # Basic similarity search
    query1 = "What is Python used for?"
    print(f"\nQuery 1: '{query1}'")
    results1 = vdb.search_by_similarity(query1, k=3)
    for i, doc in enumerate(results1, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
    
    # Search with metadata filter
    query2 = "Tell me about advanced topics"
    print(f"\nQuery 2: '{query2}' (filtered by difficulty=advanced)")
    results2 = vdb.search_by_similarity(query2, k=3, filter_dict={"difficulty": "advanced"})
    for i, doc in enumerate(results2, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
    
    # Search with score threshold
    query3 = "machine learning libraries"
    print(f"\nQuery 3: '{query3}' (with score threshold)")
    results3 = vdb.search_with_scores_and_threshold(query3, k=5, score_threshold=0.8)
    for i, (doc, score) in enumerate(results3, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
    
    # Clean up sample files
    print("\nCleaning up sample files...")
    for filename in sample_files:
        try:
            os.remove(filename)
            print(f"Removed {filename}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()