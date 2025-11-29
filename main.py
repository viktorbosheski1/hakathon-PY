from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
import requests

from database import ChromaDBManager
from pdf_processor import PDFProcessor
from ingest_doc import ingest_pdf, ingest_qa
from postgres_manager import PostgreSQLManager
from llm_helper import llm_helper

import tempfile
import os

app = FastAPI(title="Compliance Questions AI Assistant")

# Initialize managers
db_manager = ChromaDBManager()
pdf_processor = PDFProcessor()
pg_manager = PostgreSQLManager()


@app.get("/")
async def root():
    return {
        "message": "Compliance Questions AI Assistant API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "ingest_document": "/ingest-document",
            "search_internal_documents": "/search_internal_documents",
            "search_qa": "/search_qa",
            "store_new_qa": "/store_new_qa",
            "list_documents": "/documents",
            "answer_question": "/answer-question"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the service and ChromaDB connection are healthy"""
    health_status = db_manager.get_health_status()
    
    if health_status["status"] == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=health_status
        )
    
    return health_status


@app.post("/ingest-document")
async def ingest_doc(
    file: UploadFile = File(...),
    document_type: str = Form("internal-document"),
    department: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    source: Optional[str] = Form(None)
):
    """
    Upload and embed a document into ChromaDB or PostgreSQL
    
    Parameters:
    - file: file to upload (PDF for internal-document, Excel for questions-and-answers)
    - document_type: Type of document (internal-document or questions-and-answers)
    - department: Department associated with the document
    - tags: Comma-separated tags for the document
    - source: Source name for Q&A (optional, defaults to filename)
    """
    
    try:
        # Determine file extension
        file_ext = file.filename.lower().split('.')[-1]
        
        # Save uploaded file temporarily with original filename
        temp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(tmp_file_path, 'wb') as tmp_file:
            content = await file.read()
            tmp_file.write(content)

        # Strip whitespace and validate document_type
        document_type = document_type.strip() if document_type else "internal-document"
        print(f"Document type: '{document_type}', File extension: '{file_ext}'")

        if document_type == 'internal-document':
            # Validate PDF
            if file_ext != 'pdf':
                raise ValueError("Internal documents must be PDF files")
            
            print("-------TEST Ingesting internal document...")
            # Use ingest_pdf function
            result = await ingest_pdf(
                file_path=tmp_file_path,
                document_type=document_type,
                department=department,
                tags=tags
            )
        elif document_type == 'questions-and-answers':
            # Validate Excel
            if file_ext not in ['xlsx', 'xls']:
                raise ValueError("Q&A documents must be Excel files (.xlsx or .xls)")
            
            # Use ingest_qa function
            result = await ingest_qa(
                file_path=tmp_file_path,
                source=source or file.filename
            )
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
        
        # Update filename in result
        result["filename"] = file.filename
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.get("/search_internal_documents")
async def search_internal_documents(
    query: str,
    n_results: int = 5,
    document_type: Optional[str] = None,
    department: Optional[str] = None
):
    """
    Search for relevant document chunks based on a query
    
    Parameters:
    - query: Search query
    - n_results: Number of results to return (default: 1)
    - document_type: Filter by document type
    - department: Filter by department
    
    Returns:
    - query: The search query
    - results_count: Number of results
    - results: List of results with content, metadata, and similarity score
    """
    try:
        results = db_manager.search_documents(
            query=query,
            n_results=n_results,
            document_type=document_type,
            department=department
        )

        filtered_results = [result for result in results if result['similarity'] > 0.1]
        
        return {
            "query": query,
            "results_count": len(filtered_results),
            "results": filtered_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@app.get("/search_qa")
async def search_qa(
    query: str,
    n_results: int = 1
):
    """
    Search for relevant Q&A pairs from PostgreSQL based on a query
    
    Parameters:
    - query: Search query
    - n_results: Number of results to return (default: 1)
    
    Returns:
    - query: The search query
    - results_count: Number of results
    - results: List of results with query (question), answer, and similarity score
    """
    try:
        results = pg_manager.search_qa(
            query=query,
            n_results=n_results
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Q&A: {str(e)}")


@app.post("/store_new_qa")
async def store_new_qa(document_id: str = Form(...)):
    """
    Store Q&A pairs from question_documents_data table into documents table with embeddings
    
    Parameters:
    - document_id: The document ID to retrieve Q&A pairs from
    
    Returns:
    - status: Success or error status
    - message: Description of the operation
    - document_id: The document ID processed
    - pairs_stored: Number of Q&A pairs stored
    """
    try:
        result = pg_manager.store_qa_from_document(document_id=document_id)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing Q&A pairs: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all documents in the database"""
    try:
        documents = db_manager.list_all_documents()
        
        return {
            "total_documents": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the database"""
    try:
        chunks_deleted = db_manager.delete_document(document_id)
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted",
            "chunks_deleted": chunks_deleted
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.delete("/documents")
async def delete_all_documents():
    """Delete all documents from the database"""
    try:
        chunks_deleted = db_manager.delete_all_documents()
        
        return {
            "status": "success",
            "message": "All documents deleted",
            "chunks_deleted": chunks_deleted
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {str(e)}")


@app.get("/answer-question")
async def answer_question(question: str):
    """
    Answer a question using combined approach:
    1. Search Q&A database (PostgreSQL)
    2. Search internal documents (ChromaDB)
    3. Use LLM to generate answer from internal documents
    4. Use LLM to generate combined answer
    
    Parameters:
    - question: The question to answer
    
    Returns:
    - answer_qa: Answer from Q&A database
    - answer_internal_documents: Answer generated by LLM from internal documents
    - answer_combined: Combined answer from both sources
    - score_qa: Similarity score for Q&A answer (0-1)
    - score_internal_documents: Confidence score for internal documents answer (0-1)
    - score_combined: Confidence score for combined answer (0-1)
    """
    try:
        # Initialize default values
        answer_qa = ""
        answer_internal_documents = ""
        answer_combined = ""
        score_qa = 0.0
        score_internal_documents = 0.0
        score_combined = 0.0
        
        # Step 1: Search Q&A database
        try:
            qa_results = pg_manager.search_qa(query=question, n_results=1)
            if qa_results and len(qa_results) > 0:
                answer_qa = qa_results[0].get("answer", "")
                score_qa = qa_results[0].get("similarity", 0.0)
        except Exception as e:
            print(f"Error searching Q&A: {e}")
            answer_qa = ""
            score_qa = 0.0
        
        # Step 2: Search internal documents
        relevant_documents = []
        try:
            print(f"Searching internal documents for: {question}")
            internal_results = db_manager.search_documents(
                query=question,
                n_results=5
            )
            print(f"Internal documents search returned {len(internal_results) if internal_results else 0} results")
            if internal_results and len(internal_results) > 0:
                # relevant_documents = [result.get("content", "") for result in internal_results]
                relevant_documents = internal_results
                print(f"Found {len(relevant_documents)} relevant documents")
            else:
                print("No internal documents found")
        except Exception as e:
            print(f"Error searching internal documents: {e}")
            import traceback
            traceback.print_exc()
            relevant_documents = []
        
        # Step 3: Use LLM to generate answer from internal documents
        if llm_helper and relevant_documents:
            try:
                llm_response = llm_helper.get_internal_documents_answer(
                    question=question,
                    answer_qa=answer_qa,
                    relevant_documents=relevant_documents
                )
                answer_internal_documents = llm_response.get("answer", "")
                score_internal_documents = float(llm_response.get("score", 0.0))
            except Exception as e:
                print(f"Error getting internal documents answer from LLM: {e}")
                import traceback
                traceback.print_exc()
                answer_internal_documents = ""
                score_internal_documents = 0.0
        else:
            if not llm_helper:
                print("LLM helper not initialized")
            if not relevant_documents:
                print("No relevant documents found for LLM processing")
        
        # Step 4: Use LLM to generate combined answer
        if llm_helper and (answer_qa or answer_internal_documents):
            try:
                combined_response = llm_helper.get_combined_answer(
                    question=question,
                    answer_qa=answer_qa,
                    answer_internal_documents=answer_internal_documents
                )
                answer_combined = combined_response.get("answer", "")
                score_combined = float(combined_response.get("score", 0.0))
            except Exception as e:
                print(f"Error getting combined answer from LLM: {e}")
                answer_combined = ""
                score_combined = 0.0
        
        # Return all 6 values
        return {
            "question": question,
            "answer_qa": answer_qa,
            "score_qa": score_qa,

            "relevant_internal_documents": relevant_documents,
            "answer_internal_documents": answer_internal_documents,
            "score_internal_documents": score_internal_documents,

            "answer_combined": answer_combined,
            "score_combined": score_combined
        }
    
    except Exception as e:
        # Return empty values on error
        print(f"Error in answer_question endpoint: {e}")
        return {
            "answer_qa": "",
            "answer_internal_documents": "",
            "answer_combined": "",
            "score_qa": 0.0,
            "score_internal_documents": 0.0,
            "score_combined": 0.0
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
