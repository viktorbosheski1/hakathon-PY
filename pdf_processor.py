from typing import List, Tuple
from docling.document_converter import DocumentConverter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os


class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self, processing_strategy="docling"):
        """
        Initialize PDF processor
        """
        self.processing_strategy = processing_strategy
    
    def process_docling(self, file_path: str) -> Tuple[List[str], List[int]]:

        converter = DocumentConverter(
            converters=["pdfminer"],  # Using pdfminer for PDF extraction
            ocr_languages=None  # No OCR for now
        )

        # Load PDF using Docling with pre-configured converter
        result = converter.convert(file_path)
        
        markdown_content = result.document.export_to_markdown(page_break_placeholder="<page>")
        
        text_chunks = markdown_content.split("<page>")
        page_numbers = list(range(1, len(text_chunks) + 1))

        return text_chunks, page_numbers


    async def process_pdf(self, file_content: bytes) -> Tuple[List[str], List[int]]:
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
            
            if self.processing_strategy == "docling":
                text_chunks, page_numbers = self.process_docling(tmp_file_path)
            # elif self.processing_strategy == "langchain":
            else:
                raise ValueError(f"Unsupported processing strategy: {self.processing_strategy}")
            return text_chunks, page_numbers
        
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return [], []
    
    def validate_pdf(self, filename: str) -> bool:
        """
        Validate if file is a PDF
        
        Args:
            filename: Name of the file
            
        Returns:
            True if valid PDF filename
        """
        return filename.lower().endswith('.pdf')
