# Implementation Summary

## New Files Created

### 1. `postgres_manager.py`
PostgreSQL manager for Q&A vector similarity search:
- **PostgreSQLManager class**: Manages PostgreSQL connections and Q&A operations
- **search_qa method**: Searches PostgreSQL for similar Q&A pairs using vector embeddings
  - Parameters: `query` (str), `n_results` (int, default=1)
  - Returns: List of dictionaries with `query` (question), `answer`, and `similarity` score (0-1)
  - Uses pgvector extension if available, otherwise calculates similarity in Python
  - Similarity is normalized to 0-1 range using cosine similarity

### 2. `llm_helper.py`
LLM helper for Azure OpenAI operations:
- **LLMHelper class**: Manages Azure OpenAI API calls
- **read_prompt_from_file**: Reads prompt templates from disk
- **format_user_prompt**: Formats prompts with variables
- **call_llm**: Generic method to call Azure OpenAI with JSON response parsing
- **get_internal_documents_answer**: Gets LLM answer from internal documents
  - Reads system and user prompts from `./prompts/internal_docs_system.txt` and `./prompts/internal_docs_user.txt`
  - Input variables: question, answer_qa, relevant_document
  - Returns: JSON with `answer` and `score` (0-1)
- **get_combined_answer**: Gets combined LLM answer
  - Reads system and user prompts from `./prompts/combined_system.txt` and `./prompts/combined_user.txt`
  - Input variables: question, answer_qa, answer_internal_documents
  - Returns: JSON with `answer` and `score` (0-1)

### 3. Prompt Template Files (in `./prompts/` directory)
- `internal_docs_system.txt`: System prompt for internal documents analysis
- `internal_docs_user.txt`: User prompt template for internal documents (variables: question, answer_qa, relevant_document)
- `combined_system.txt`: System prompt for combining answers
- `combined_user.txt`: User prompt template for combining answers (variables: question, answer_qa, answer_internal_documents)

## Updated Files

### `main.py`
Added new endpoints:

1. **GET `/search_qa`**
   - Parameters: `query` (str), `n_results` (int, default=1)
   - Returns: Query, results count, and list of Q&A results with similarity scores
   - Searches PostgreSQL database for similar Q&A pairs

2. **GET `/answer-question`**
   - Parameter: `question` (str)
   - Returns 6 values:
     - `answer_qa`: Answer from Q&A database
     - `answer_internal_documents`: LLM-generated answer from internal documents
     - `answer_combined`: LLM-generated combined answer
     - `score_qa`: Similarity score from Q&A search (0-1)
     - `score_internal_documents`: Confidence score from LLM (0-1)
     - `score_combined`: Confidence score from LLM (0-1)
   
   **Workflow:**
   1. Search PostgreSQL Q&A database → get answer_qa and score_qa
   2. Search internal documents (ChromaDB) → get most relevant document
   3. Call LLM with question, answer_qa, and relevant document → get answer_internal_documents and score_internal_documents
   4. Call LLM with question, answer_qa, and answer_internal_documents → get answer_combined and score_combined
   5. Return all 6 values (wrapped in try-except, returns empty values on error)

Updated endpoint:
- **GET `/search_internal_documents`**: Now explicitly returns similarity scores in the response

### `requirements.txt`
Added new dependencies:
- `openai` - Azure OpenAI Python SDK
- `numpy` - For vector similarity calculations

## Configuration

### Environment Variables (`.env`)
Required variables (already present):
- `AZURE_ENDPOINT` - Azure OpenAI endpoint URL
- `AZURE_API_KEY` - Azure OpenAI API key
- `REASONING_MODEL_NAME` - Model name (e.g., gpt-4, gpt-5-mini)
- `REASONING_API_VERSION` - API version (e.g., 2024-12-01-preview)
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER_NAME`, `POSTGRES_PASSWORD` - PostgreSQL connection details
- `EMBEDDING_MODEL_NAME`, `EMBEDDING_API_VERSION` - Embedding model configuration

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Search Q&A Database
```bash
GET /search_qa?query=What%20is%20the%20policy%20on%20data%20retention?&n_results=1
```

Response:
```json
{
  "query": "What is the policy on data retention?",
  "results_count": 1,
  "results": [
    {
      "query": "What is our data retention policy?",
      "answer": "Data must be retained for 7 years...",
      "similarity": 0.92
    }
  ]
}
```

### Search Internal Documents
```bash
GET /search_internal_documents?query=compliance%20requirements&n_results=1
```

Response:
```json
{
  "query": "compliance requirements",
  "results_count": 1,
  "results": [
    {
      "id": "doc_id_chunk_0",
      "content": "The compliance requirements include...",
      "metadata": {...},
      "similarity": 0.87
    }
  ]
}
```

### Answer Question (Complete Workflow)
```bash
GET /answer-question?question=What%20are%20the%20data%20retention%20requirements?
```

Response:
```json
{
  "answer_qa": "Data must be retained for 7 years according to policy...",
  "answer_internal_documents": "Based on the internal policy document, data retention requirements specify...",
  "answer_combined": "Combining both sources, the comprehensive answer is...",
  "score_qa": 0.92,
  "score_internal_documents": 0.85,
  "score_combined": 0.89
}
```

## Features

1. **Semantic Similarity Scores**: All scores are normalized to 0-1 range for consistency
2. **Multi-source Answers**: Combines Q&A database, internal documents, and LLM reasoning
3. **Error Handling**: Comprehensive try-except blocks return empty values on errors
4. **Flexible Prompts**: Prompts stored in files for easy customization
5. **JSON Responses**: LLM returns structured JSON for reliable parsing
6. **PostgreSQL Vector Search**: Optimized with pgvector extension support

## Architecture

```
User Question
    ↓
1. Search PostgreSQL Q&A → answer_qa + score_qa
    ↓
2. Search ChromaDB Internal Docs → relevant_document
    ↓
3. LLM Call (Internal Docs) → answer_internal_documents + score_internal_documents
    ↓
4. LLM Call (Combined) → answer_combined + score_combined
    ↓
Return 6 values
```
