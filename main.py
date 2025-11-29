from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import uuid

from database import ChromaDBManager
from pdf_processor import PDFProcessor
from ingest_doc import ingest_pdf, ingest_qa

import tempfile
import os

app = FastAPI(title="Compliance Questions AI Assistant")

# Initialize managers
db_manager = ChromaDBManager()
pdf_processor = PDFProcessor()


@app.get("/")
async def root():
    return {
        "message": "Compliance Questions AI Assistant API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "upload_pdf": "/upload-pdf",
            "search": "/search",
            "list_documents": "/documents"
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
        
        # Save uploaded file temporarily
        suffix = f'.{file_ext}'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Strip whitespace and validate document_type
        document_type = document_type.strip() if document_type else "internal-document"
        print(f"Document type: '{document_type}', File extension: '{file_ext}'")

        if document_type == 'internal-document':
            # Validate PDF
            if file_ext != 'pdf':
                raise ValueError("Internal documents must be PDF files")
            
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


@app.get("/search")
async def search_documents(
    query: str,
    n_results: int = 1,
    document_type: Optional[str] = None,
    department: Optional[str] = None
):
    """
    Search for relevant document chunks based on a query
    
    Parameters:
    - query: Search query
    - n_results: Number of results to return (default: 5)
    - document_type: Filter by document type
    - department: Filter by department
    """
    try:
        results = db_manager.search_documents(
            query=query,
            n_results=n_results,
            document_type=document_type,
            department=department
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
