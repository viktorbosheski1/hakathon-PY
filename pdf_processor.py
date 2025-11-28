from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os


class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, chunk_by_page: bool = True):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of text chunks (used only if chunk_by_page=False)
            chunk_overlap: Overlap between chunks (used only if chunk_by_page=False)
            chunk_by_page: If True, creates one chunk per page. If False, uses character-based chunking
        """
        self.chunk_by_page = chunk_by_page
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    async def process_pdf(self, file_content: bytes, filename: str) -> Tuple[List[str], List[int]]:
        """
        Process PDF file and extract text chunks
        
        Args:
            file_content: Binary content of the PDF file
            filename: Name of the file
            
        Returns:
            Tuple of (text_chunks, page_numbers)
        """
        tmp_file_path = None
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            # Load PDF using LangChain
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            
            if self.chunk_by_page:
                # Create one chunk per page
                text_chunks = [page.page_content for page in pages]
                page_numbers = [page.metadata.get("page", idx) for idx, page in enumerate(pages)]
            else:
                # Split documents into chunks by character count
                chunks = self.text_splitter.split_documents(pages)
                text_chunks = [chunk.page_content for chunk in chunks]
                page_numbers = [chunk.metadata.get("page", 0) for chunk in chunks]
            
            return text_chunks, page_numbers
        
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def validate_pdf(self, filename: str) -> bool:
        """
        Validate if file is a PDF
        
        Args:
            filename: Name of the file
            
        Returns:
            True if valid PDF filename
        """
        return filename.lower().endswith('.pdf')
