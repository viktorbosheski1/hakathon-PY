# Compliance Questions AI Assistant

FastAPI application for embedding PDF documents into ChromaDB with metadata support.

## Features

- **ChromaDB Integration**: Connects to remote or local ChromaDB instance
- **PDF Document Processing**: Uploads and embeds PDF documents
- **Metadata Storage**: Stores document metadata (type, department, tags, etc.)
- **Semantic Search**: Search documents using natural language queries
- **Document Management**: List and delete documents

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**:
Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_openai_api_key
```

3. **Run the application**:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /health
```
Check service health and ChromaDB connection status.

### Upload PDF
```
POST /upload-pdf
```
Upload and embed a PDF document.

**Parameters**:
- `file`: PDF file (required)
- `document_type`: Type of document (optional, default: "compliance")
- `department`: Department name (optional)
- `tags`: Comma-separated tags (optional)

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@document.pdf" \
  -F "document_type=policy" \
  -F "department=Legal" \
  -F "tags=gdpr,privacy"
```

**Example using Python**:
```python
import requests

with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "document_type": "policy",
        "department": "Legal",
        "tags": "gdpr,privacy"
    }
    response = requests.post("http://localhost:8000/upload-pdf", files=files, data=data)
    print(response.json())
```

### Search Documents
```
GET /search?query=your_query&n_results=5
```
Search for relevant document chunks.

**Parameters**:
- `query`: Search query (required)
- `n_results`: Number of results (optional, default: 5)
- `document_type`: Filter by document type (optional)
- `department`: Filter by department (optional)

**Example**:
```bash
curl "http://localhost:8000/search?query=data%20protection&n_results=3"
```

### List Documents
```
GET /documents
```
List all documents in the database.

### Delete Document
```
DELETE /documents/{document_id}
```
Delete a document and all its chunks.

## How It Works

1. **PDF Upload**: When you upload a PDF, the system:
   - Saves the file temporarily
   - Extracts text from all pages using PyPDF2
   - Splits the text into chunks (1000 chars with 200 char overlap)
   - Generates embeddings for each chunk using OpenAI
   - Stores chunks, embeddings, and metadata in ChromaDB

2. **Search**: When you search:
   - Your query is embedded using the same OpenAI model
   - ChromaDB performs similarity search using vector distance
   - Returns the most relevant chunks with their metadata

3. **Metadata**: Each chunk stores:
   - Document ID (unique identifier for the document)
   - Filename
   - Document type
   - Department
   - Tags
   - Chunk index and total chunks
   - Page number

## Configuration

### ChromaDB Connection

The application supports both remote and local ChromaDB:

**Remote ChromaDB** (configured in `.env`):
```
USE_REMOTE_CHROMA=true
CHROMA_HOST=ec2-13-60-3-254.eu-north-1.compute.amazonaws.com
CHROMA_PORT=8000
```

**Local ChromaDB**:
```
USE_REMOTE_CHROMA=false
```
Data will be stored in `./chroma_db` directory.

## Interactive API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Requirements

- Python 3.8+
- OpenAI API key
- ChromaDB instance (remote or local)
