import uuid
from typing import Optional
from database import ChromaDBManager
from pdf_processor import PDFProcessor

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from datetime import datetime

async def ingest_pdf(
    file_path: str,
    document_type: str = "internal-document",
    department: Optional[str] = None,
    tags: Optional[str] = None
) -> dict:
    """
    Ingest a PDF document into ChromaDB
    
    Args:
        file_path: Path to the PDF file
        document_type: Type of document (default: internal-document)
        department: Department associated with the document
        tags: Comma-separated tags for the document
        
    Returns:
        Dictionary with ingestion results
    """

    # Initialize managers
    pdf_processor = PDFProcessor()

    # Validate file type
    if not pdf_processor.validate_pdf(file_path):
        raise ValueError("Only PDF files are supported")
    
    try:
        # Read file content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Process PDF
        text_chunks, page_numbers = await pdf_processor.process_pdf(content)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Store in ChromaDB
        db_manager = ChromaDBManager()
        stored_chunks = db_manager.add_document_chunks(
            chunks=text_chunks,
            doc_id=doc_id,
            filename=file_path.split("\\")[-1].split("/")[-1],  # Extract filename
            document_type=document_type,
            department=department,
            tags=tag_list,
            page_numbers=page_numbers
        )
        
        return {
            "status": "success",
            "message": "PDF document successfully embedded",
            "document_id": doc_id,
            "filename": file_path,
            "chunks_stored": stored_chunks,
            "metadata": {
                "document_type": document_type,
                "department": department,
                "tags": tag_list
            }
        }
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    

async def ingest_qa(
    file_path: str,
    source: Optional[str] = None
) -> dict:
    """
    Ingest Q&A pairs from Excel file into PostgreSQL
    
    Args:
        file_path: Path to the Excel file
        source: Source name (defaults to filename)
        
    Returns:
        Dictionary with ingestion results
    """
    
    load_dotenv()
    
    try:
        # 1. Read Excel file
        df = pd.read_excel(file_path)
        
        # Validate columns
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            raise ValueError("Excel file must contain 'Question' and 'Answer' columns")
        
        # 2. Store questions and answers in lists
        questions = df['Question'].astype(str).tolist()
        answers = df['Answer'].fillna('').astype(str).tolist()  # Fill NaN with empty string
        
        # 3. Initialize Azure OpenAI embedding model
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_api_key = os.getenv("AZURE_API_KEY")
        deployment_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
        api_version = os.getenv("EMBEDDING_API_VERSION", "2024-12-01-preview")
        
        embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            api_key=azure_api_key,
            api_version=api_version
        )
        
        # 4. Generate embeddings for questions and answers
        question_embeddings = []
        answer_embeddings = []
        
        for question, answer in zip(questions, answers):
            # Always generate question embedding
            q_embedding = embeddings_model.embed_query(question)
            question_embeddings.append(q_embedding)
            
            # Only generate answer embedding if answer is not empty
            if answer and answer.strip() and answer.lower() != 'nan':
                a_embedding = embeddings_model.embed_query(answer)
                answer_embeddings.append(a_embedding)
            else:
                answer_embeddings.append(None)  # Store None for empty answers
        
        # 5. We now have 4 lists: questions, question_embeddings, answers, answer_embeddings
        
        # 6. Initialize PostgreSQL connection
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER_NAME"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        cursor = conn.cursor()
        
        # 7. Insert each question, answer, and their embeddings into Postgres
        source_name = source or file_path.split("\\")[-1].split("/")[-1]
        created_at = datetime.now()
        
        # Prepare data for batch insert
        insert_data = []
        for q, q_emb, a, a_emb in zip(questions, question_embeddings, answers, answer_embeddings):
            # Skip rows where answer is empty/null to avoid duplicate key constraint
            if not a or not a.strip() or a.lower() == 'nan':
                continue
                
            insert_data.append((
                q,                  # question
                q_emb,              # question_vector
                a,                  # answer
                a_emb,              # answer_vector (can be None)
                source_name,        # source
                created_at,         # created_at
                None                # updated_at (NULL initially)
            ))
        
        # Only proceed if we have data to insert
        if not insert_data:
            return {
                "status": "warning",
                "message": "No valid Q&A pairs found (all answers were empty)",
                "source": source_name,
                "records_inserted": 0,
                "file": file_path
            }
        
        # Execute batch insert
        insert_query = """
            INSERT INTO public.documents 
            (question, question_vector, answer, answer_vector, source, created_at, updated_at)
            VALUES %s
        """
        
        execute_values(cursor, insert_query, insert_data)
        conn.commit()
        
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "message": "Q&A pairs successfully embedded and stored in PostgreSQL",
            "source": source_name,
            "records_inserted": len(insert_data),
            "file": file_path
        }
    
    except Exception as e:
        raise Exception(f"Error processing Q&A file: {str(e)}")