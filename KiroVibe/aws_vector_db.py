#!/usr/bin/env python3
"""
AWS Vector Database Integration Examples
Demonstrates multiple approaches for storing vector databases in AWS.
"""

import os
import json
import boto3
from typing import List, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch, FAISS
from langchain.schema import Document

load_dotenv()

class AWSVectorDatabase:
    def __init__(self, 
                 aws_region="us-east-1",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize AWS Vector Database with configurable settings."""
        self.aws_region = aws_region
        self.embedding_model = embedding_model
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
    
    def create_opensearch_vectorstore(self, 
                                    opensearch_url: str,
                                    index_name: str = "vector-index",
                                    username: Optional[str] = None,
                                    password: Optional[str] = None):
        """Create OpenSearch vector store (AWS managed)."""
        
        # OpenSearch connection parameters
        connection_params = {
            "hosts": [opensearch_url],
            "http_auth": (username, password) if username and password else None,
            "use_ssl": True,
            "verify_certs": True,
            "ssl_show_warn": False
        }
        
        # Create OpenSearch vector store
        vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=self.embeddings,
            **connection_params
        )
        
        return vectorstore
    
    def create_faiss_vectorstore(self):
        """Create FAISS vector store for S3 storage."""
        # FAISS doesn't need initial setup, created when adding documents
        return None
    
    def upload_faiss_to_s3(self, 
                          vectorstore, 
                          bucket_name: str, 
                          s3_key_prefix: str = "vector-db/"):
        """Upload FAISS index to S3."""
        try:
            # Save FAISS index locally first
            local_path = "./temp_faiss_index"
            vectorstore.save_local(local_path)
            
            # Upload index files to S3
            index_file = f"{local_path}/index.faiss"
            pkl_file = f"{local_path}/index.pkl"
            
            # Upload index.faiss
            self.s3_client.upload_file(
                index_file, 
                bucket_name, 
                f"{s3_key_prefix}index.faiss"
            )
            
            # Upload index.pkl
            self.s3_client.upload_file(
                pkl_file, 
                bucket_name, 
                f"{s3_key_prefix}index.pkl"
            )
            
            print(f"FAISS index uploaded to s3://{bucket_name}/{s3_key_prefix}")
            
            # Clean up local files
            import shutil
            shutil.rmtree(local_path)
            
            return True
            
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return False
    
    def download_faiss_from_s3(self, 
                              bucket_name: str, 
                              s3_key_prefix: str = "vector-db/",
                              local_path: str = "./downloaded_faiss_index"):
        """Download FAISS index from S3."""
        try:
            # Create local directory
            os.makedirs(local_path, exist_ok=True)
            
            # Download index files
            self.s3_client.download_file(
                bucket_name, 
                f"{s3_key_prefix}index.faiss",
                f"{local_path}/index.faiss"
            )
            
            self.s3_client.download_file(
                bucket_name, 
                f"{s3_key_prefix}index.pkl",
                f"{local_path}/index.pkl"
            )
            
            # Load FAISS index
            vectorstore = FAISS.load_local(
                local_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print(f"FAISS index downloaded from s3://{bucket_name}/{s3_key_prefix}")
            return vectorstore
            
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return None
    
    def add_documents_to_vectorstore(self, 
                                   vectorstore, 
                                   texts: List[str], 
                                   metadatas: Optional[List[dict]] = None):
        """Add documents to any vector store."""
        documents = [Document(page_content=text) for text in texts]
        
        if metadatas:
            for doc, metadata in zip(documents, metadatas):
                doc.metadata.update(metadata)
        
        split_docs = self.text_splitter.split_documents(documents)
        
        if vectorstore is None:  # FAISS case
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        else:  # OpenSearch case
            vectorstore.add_documents(split_docs)
        
        return vectorstore, len(split_docs)


def demo_opensearch_integration():
    """Demo OpenSearch integration (requires AWS OpenSearch cluster)."""
    print("=== OpenSearch Vector Database Demo ===")
    
    # Note: You need to create an OpenSearch cluster in AWS first
    opensearch_url = os.getenv("OPENSEARCH_URL", "https://your-opensearch-domain.us-east-1.es.amazonaws.com")
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    
    if opensearch_url == "https://your-opensearch-domain.us-east-1.es.amazonaws.com":
        print("⚠️  OpenSearch URL not configured. Set OPENSEARCH_URL in .env file")
        print("   Create an OpenSearch cluster in AWS first")
        return
    
    aws_vdb = AWSVectorDatabase()
    
    try:
        # Create OpenSearch vector store
        vectorstore = aws_vdb.create_opensearch_vectorstore(
            opensearch_url=opensearch_url,
            index_name="langchain-demo",
            username=username,
            password=password
        )
        
        # Sample documents
        sample_texts = [
            "AWS OpenSearch is a managed search and analytics service.",
            "Vector databases enable semantic search capabilities.",
            "LangChain provides integrations with various vector stores."
        ]
        
        sample_metadata = [
            {"source": "aws_docs", "category": "service"},
            {"source": "vector_guide", "category": "database"},
            {"source": "langchain_docs", "category": "integration"}
        ]
        
        # Add documents
        vectorstore, count = aws_vdb.add_documents_to_vectorstore(
            vectorstore, sample_texts, sample_metadata
        )
        
        print(f"Added {count} documents to OpenSearch")
        
        # Search
        results = vectorstore.similarity_search("What is OpenSearch?", k=2)
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")
        
    except Exception as e:
        print(f"OpenSearch demo failed: {e}")
        print("Make sure your OpenSearch cluster is accessible and credentials are correct")


def demo_faiss_s3_integration():
    """Demo FAISS with S3 storage."""
    print("\n=== FAISS + S3 Vector Database Demo ===")
    
    bucket_name = os.getenv("S3_BUCKET_NAME", "your-vector-db-bucket")
    
    if bucket_name == "your-vector-db-bucket":
        print("⚠️  S3 bucket not configured. Set S3_BUCKET_NAME in .env file")
        print("   Create an S3 bucket first")
        return
    
    aws_vdb = AWSVectorDatabase()
    
    try:
        # Sample documents
        sample_texts = [
            "FAISS is a library for efficient similarity search and clustering.",
            "S3 provides scalable object storage for vector database backups.",
            "LangChain supports FAISS integration for local vector storage."
        ]
        
        sample_metadata = [
            {"source": "faiss_docs", "category": "library"},
            {"source": "aws_docs", "category": "storage"},
            {"source": "langchain_docs", "category": "integration"}
        ]
        
        # Create FAISS vector store
        vectorstore, count = aws_vdb.add_documents_to_vectorstore(
            None, sample_texts, sample_metadata
        )
        
        print(f"Created FAISS index with {count} documents")
        
        # Test search before upload
        results = vectorstore.similarity_search("What is FAISS?", k=2)
        print("\nSearch results before S3 upload:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
        
        # Upload to S3
        success = aws_vdb.upload_faiss_to_s3(vectorstore, bucket_name, "demo-vector-db/")
        
        if success:
            # Download from S3
            downloaded_vectorstore = aws_vdb.download_faiss_from_s3(
                bucket_name, "demo-vector-db/"
            )
            
            if downloaded_vectorstore:
                # Test search after download
                results = downloaded_vectorstore.similarity_search("What is S3?", k=2)
                print("\nSearch results after S3 download:")
                for i, doc in enumerate(results, 1):
                    print(f"{i}. {doc.page_content}")
        
    except Exception as e:
        print(f"FAISS + S3 demo failed: {e}")
        print("Make sure your AWS credentials are configured and S3 bucket exists")


def main():
    print("AWS Vector Database Integration Examples")
    print("=" * 50)
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        print("✅ AWS credentials found")
    except Exception as e:
        print("❌ AWS credentials not configured")
        print("   Configure with: aws configure")
        print("   Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return
    
    # Demo OpenSearch integration
    demo_opensearch_integration()
    
    # Demo FAISS + S3 integration
    demo_faiss_s3_integration()
    
    print("\n" + "=" * 50)
    print("Setup Instructions:")
    print("1. Configure AWS credentials: aws configure")
    print("2. Create S3 bucket for FAISS storage")
    print("3. (Optional) Create OpenSearch cluster for managed vector search")
    print("4. Update .env file with your AWS resource URLs")


if __name__ == "__main__":
    main()