#!/usr/bin/env python3
"""
AWS Bedrock Knowledge Base Integration
Uses AWS Bedrock's managed vector database service.
"""

import os
import boto3
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

class BedrockVectorDatabase:
    def __init__(self, aws_region="us-east-1"):
        """Initialize Bedrock Vector Database."""
        self.aws_region = aws_region
        
        # Initialize AWS clients
        self.bedrock_client = boto3.client('bedrock', region_name=aws_region)
        self.bedrock_agent_client = boto3.client('bedrock-agent', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Initialize Bedrock embeddings
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def create_knowledge_base(self, 
                            knowledge_base_name: str,
                            s3_bucket_name: str,
                            s3_key_prefix: str = "documents/",
                            description: str = "LangChain Vector Database"):
        """Create a Bedrock Knowledge Base."""
        try:
            # Create knowledge base
            response = self.bedrock_agent_client.create_knowledge_base(
                name=knowledge_base_name,
                description=description,
                roleArn=f"arn:aws:iam::{self._get_account_id()}:role/AmazonBedrockExecutionRoleForKnowledgeBase",
                knowledgeBaseConfiguration={
                    'type': 'VECTOR',
                    'vectorKnowledgeBaseConfiguration': {
                        'embeddingModelArn': f"arn:aws:bedrock:{self.aws_region}::foundation-model/amazon.titan-embed-text-v1"
                    }
                },
                storageConfiguration={
                    'type': 'OPENSEARCH_SERVERLESS',
                    'opensearchServerlessConfiguration': {
                        'collectionArn': f"arn:aws:aoss:{self.aws_region}:{self._get_account_id()}:collection/{knowledge_base_name.lower()}",
                        'vectorIndexName': 'vector-index',
                        'fieldMapping': {
                            'vectorField': 'vector',
                            'textField': 'text',
                            'metadataField': 'metadata'
                        }
                    }
                }
            )
            
            knowledge_base_id = response['knowledgeBase']['knowledgeBaseId']
            print(f"Created Knowledge Base: {knowledge_base_id}")
            
            # Create data source
            data_source_response = self.bedrock_agent_client.create_data_source(
                knowledgeBaseId=knowledge_base_id,
                name=f"{knowledge_base_name}-data-source",
                description="S3 data source for vector database",
                dataSourceConfiguration={
                    'type': 'S3',
                    's3Configuration': {
                        'bucketArn': f"arn:aws:s3:::{s3_bucket_name}",
                        'inclusionPrefixes': [s3_key_prefix]
                    }
                }
            )
            
            data_source_id = data_source_response['dataSource']['dataSourceId']
            print(f"Created Data Source: {data_source_id}")
            
            return knowledge_base_id, data_source_id
            
        except Exception as e:
            print(f"Error creating knowledge base: {e}")
            return None, None
    
    def upload_documents_to_s3(self, 
                              documents: List[str], 
                              bucket_name: str, 
                              key_prefix: str = "documents/"):
        """Upload documents to S3 for Bedrock ingestion."""
        try:
            uploaded_files = []
            
            for i, doc_content in enumerate(documents):
                # Create filename
                filename = f"document_{i+1}.txt"
                s3_key = f"{key_prefix}{filename}"
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=doc_content.encode('utf-8'),
                    ContentType='text/plain'
                )
                
                uploaded_files.append(s3_key)
                print(f"Uploaded: s3://{bucket_name}/{s3_key}")
            
            return uploaded_files
            
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return []
    
    def start_ingestion_job(self, knowledge_base_id: str, data_source_id: str):
        """Start ingestion job to process documents."""
        try:
            response = self.bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                description="Ingesting documents for vector search"
            )
            
            ingestion_job_id = response['ingestionJob']['ingestionJobId']
            print(f"Started ingestion job: {ingestion_job_id}")
            
            return ingestion_job_id
            
        except Exception as e:
            print(f"Error starting ingestion job: {e}")
            return None
    
    def query_knowledge_base(self, 
                           knowledge_base_id: str, 
                           query: str, 
                           max_results: int = 5):
        """Query the knowledge base."""
        try:
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            results = []
            for result in response['retrievalResults']:
                results.append({
                    'content': result['content']['text'],
                    'score': result['score'],
                    'location': result.get('location', {}),
                    'metadata': result.get('metadata', {})
                })
            
            return results
            
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []
    
    def _get_account_id(self):
        """Get AWS account ID."""
        sts_client = boto3.client('sts')
        return sts_client.get_caller_identity()['Account']


def demo_bedrock_knowledge_base():
    """Demo Bedrock Knowledge Base integration."""
    print("=== AWS Bedrock Knowledge Base Demo ===")
    
    # Configuration
    knowledge_base_name = "langchain-demo-kb"
    bucket_name = os.getenv("S3_BUCKET_NAME", "your-bedrock-bucket")
    
    if bucket_name == "your-bedrock-bucket":
        print("⚠️  S3 bucket not configured. Set S3_BUCKET_NAME in .env file")
        return
    
    bedrock_vdb = BedrockVectorDatabase()
    
    try:
        # Sample documents
        documents = [
            """
            Amazon Bedrock is a fully managed service that offers a choice of 
            high-performing foundation models (FMs) from leading AI companies 
            like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon 
            via a single API, along with a broad set of capabilities you need 
            to build generative AI applications with security, privacy, and 
            responsible AI.
            """,
            """
            Vector databases are specialized databases designed to store and 
            query high-dimensional vectors efficiently. They enable semantic 
            search, recommendation systems, and similarity matching at scale.
            """,
            """
            LangChain is a framework for developing applications powered by 
            language models. It provides integrations with various vector 
            databases and enables building RAG (Retrieval-Augmented Generation) 
            applications.
            """
        ]
        
        # Upload documents to S3
        print("Uploading documents to S3...")
        uploaded_files = bedrock_vdb.upload_documents_to_s3(
            documents, bucket_name, "langchain-demo/"
        )
        
        if not uploaded_files:
            print("Failed to upload documents")
            return
        
        # Create knowledge base (Note: This requires proper IAM roles)
        print("Creating knowledge base...")
        kb_id, ds_id = bedrock_vdb.create_knowledge_base(
            knowledge_base_name, bucket_name, "langchain-demo/"
        )
        
        if not kb_id:
            print("Failed to create knowledge base")
            print("Make sure you have the required IAM roles and permissions")
            return
        
        # Start ingestion job
        print("Starting ingestion job...")
        job_id = bedrock_vdb.start_ingestion_job(kb_id, ds_id)
        
        if job_id:
            print("Ingestion job started. Wait for completion before querying.")
            print(f"Knowledge Base ID: {kb_id}")
            print("You can query this knowledge base once ingestion is complete.")
            
            # Example query (uncomment after ingestion is complete)
            # print("Querying knowledge base...")
            # results = bedrock_vdb.query_knowledge_base(kb_id, "What is Amazon Bedrock?")
            # for i, result in enumerate(results, 1):
            #     print(f"{i}. Score: {result['score']:.4f}")
            #     print(f"   Content: {result['content'][:100]}...")
        
    except Exception as e:
        print(f"Bedrock demo failed: {e}")
        print("Make sure you have Bedrock access and proper IAM permissions")


def main():
    print("AWS Bedrock Knowledge Base Integration")
    print("=" * 50)
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        print("✅ AWS credentials found")
    except Exception as e:
        print("❌ AWS credentials not configured")
        return
    
    # Check Bedrock access
    try:
        bedrock_client = boto3.client('bedrock', region_name='us-east-1')
        bedrock_client.list_foundation_models()
        print("✅ Bedrock access confirmed")
    except Exception as e:
        print("❌ Bedrock access not available")
        print("   Make sure Bedrock is available in your region and you have permissions")
    
    demo_bedrock_knowledge_base()
    
    print("\n" + "=" * 50)
    print("Setup Requirements:")
    print("1. AWS credentials configured")
    print("2. S3 bucket for document storage")
    print("3. IAM role: AmazonBedrockExecutionRoleForKnowledgeBase")
    print("4. Bedrock model access enabled")
    print("5. OpenSearch Serverless collection (auto-created)")


if __name__ == "__main__":
    main()