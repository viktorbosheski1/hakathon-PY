import asyncio
from pdf_processor import PDFProcessor

async def main():
    pdf_processor = PDFProcessor()
    
    file_path = "IW_Cryptography.pdf"
    
    with open(file_path, "rb") as f:
        file_content = f.read()
        text_chunks, page_numbers = await pdf_processor.process_pdf(file_content, file_path)
    
    print(f"Total chunks: {len(text_chunks)}")
    print(f"Total pages: {len(page_numbers)}")
    
    # Print first chunk as example
    if text_chunks:
        print(f"\nFirst chunk (page {page_numbers[0]}):")
        print(text_chunks[0][:500])  # First 500 characters



if __name__ == "__main__":
    asyncio.run(main())