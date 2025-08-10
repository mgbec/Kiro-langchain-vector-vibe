# LangChain Vector Database Examples

This project demonstrates how to create and use vector databases with LangChain, featuring both basic and advanced implementations.

## Features

- **Basic Vector Database**: Simple implementation with ChromaDB
- **Advanced Vector Database**: Enhanced features including file loading, metadata filtering, and score thresholds
- **HuggingFace Embeddings**: Uses free, local embedding models
- **Persistent Storage**: Saves vector databases to disk
- **Similarity Search**: Find semantically similar documents

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Create a `.env` file for API keys:
```bash
cp .env.example .env
# Edit .env with your API keys if needed
```

## Usage

### Basic Example

Run the basic vector database example:
```bash
python vector_db_example.py
```

This will:
- Create a vector database with sample documents
- Demonstrate similarity search with scores
- Persist the database to `./chroma_db/`

### Advanced Example

Run the advanced vector database example:
```bash
python advanced_vector_db.py
```

This demonstrates:
- Loading documents from files
- Metadata filtering
- Score thresholds
- Collection statistics

## Key Components

### VectorDatabase Class
- `add_documents()`: Add text documents with optional metadata
- `similarity_search()`: Find similar documents
- `similarity_search_with_score()`: Search with similarity scores
- `persist()`: Save database to disk

### AdvancedVectorDatabase Class
- `load_documents_from_directory()`: Load multiple files
- `search_by_similarity()`: Search with metadata filtering
- `search_with_scores_and_threshold()`: Filter by similarity threshold
- `get_collection_stats()`: Database statistics

## Embedding Models

The examples use HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model, which:
- Runs locally (no API key required)
- Provides good performance for most use cases
- Creates 384-dimensional embeddings

## Customization

You can easily customize:
- **Embedding Model**: Change the `model_name` parameter
- **Chunk Size**: Modify `chunk_size` in the text splitter
- **Database Location**: Set different `persist_directory`
- **Search Parameters**: Adjust `k` (number of results) and score thresholds

## File Structure

```
.
├── vector_db_example.py      # Basic implementation
├── advanced_vector_db.py     # Advanced features
├── requirements.txt          # Dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## AWS Integration

### Option 1: FAISS + S3 Storage
Store FAISS vector databases in S3 for scalable, cost-effective storage:

```bash
python aws_vector_db.py
```

Features:
- Local FAISS indexing with S3 backup
- Upload/download vector databases to/from S3
- Cost-effective for moderate scale

### Option 2: AWS OpenSearch
Use managed OpenSearch for production vector search:

```bash
# Requires OpenSearch cluster setup
python aws_vector_db.py
```

Features:
- Fully managed vector search
- Real-time indexing and search
- Built-in scaling and high availability

### Option 3: AWS Bedrock Knowledge Base
Use AWS Bedrock's managed vector database service:

```bash
python aws_bedrock_vector_db.py
```

Features:
- Fully managed by AWS
- Integrated with Bedrock foundation models
- Automatic document processing and chunking

### AWS Setup

1. **Quick Setup**: Run the setup script to create AWS resources:
```bash
python setup_aws_resources.py
```

2. **Manual Setup**:
   - Create S3 bucket for storage
   - (Optional) Create OpenSearch domain
   - (Optional) Set up Bedrock Knowledge Base with IAM roles

3. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and resource names
```

## Next Steps

- Integrate with your own documents
- Experiment with different embedding models
- Add more sophisticated metadata
- Implement retrieval-augmented generation (RAG)
- Scale to larger document collections with AWS services