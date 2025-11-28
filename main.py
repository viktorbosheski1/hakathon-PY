from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid

from database import ChromaDBManager
from pdf_processor import PDFProcessor

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


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: Optional[str] = "compliance",
    department: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    Upload and embed a PDF document into ChromaDB
    
    Parameters:
    - file: PDF file to upload
    - document_type: Type of document (default: compliance)
    - department: Department associated with the document
    - tags: Comma-separated tags for the document
    """
    # Validate file type
    if not pdf_processor.validate_pdf(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process PDF
        text_chunks, page_numbers = await pdf_processor.process_pdf(content, file.filename)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Store in ChromaDB
        stored_chunks = db_manager.add_document_chunks(
            chunks=text_chunks,
            doc_id=doc_id,
            filename=file.filename,
            document_type=document_type,
            department=department,
            tags=tag_list,
            page_numbers=page_numbers
        )
        
        return {
            "status": "success",
            "message": "PDF document successfully embedded",
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_stored": stored_chunks,
            "metadata": {
                "document_type": document_type,
                "department": department,
                "tags": tag_list
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/search")
async def search_documents(
    query: str,
    n_results: int = 5,
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
