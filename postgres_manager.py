import os
import psycopg2
from typing import List, Dict, Any, Optional
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()


class PostgreSQLManager:
    """Manages PostgreSQL connections and Q&A search operations"""
    
    def __init__(self):
        """Initialize PostgreSQL connection and embeddings model"""
        # Initialize Azure OpenAI Embeddings
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_api_key = os.getenv("AZURE_API_KEY")
        deployment_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
        azure_api_version = os.getenv("EMBEDDING_API_VERSION", "2024-12-01-preview")
        
        self.embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
        
        # PostgreSQL connection parameters
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER_NAME"),
            "password": os.getenv("POSTGRES_PASSWORD")
        }
        
        print(f"PostgreSQL Manager initialized")
    
    def _get_connection(self):
        """Create and return a PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        return float((similarity + 1) / 2)
    
    def _calculate_recency_boost(self, row_index: int, total_rows: int, decay_rate: float = 0.2) -> float:
        """
        Calculate exponential recency boost based on row position
        Higher row index (later in table) = more recent = higher boost
        
        Args:
            row_index: Position of the row (0-indexed, higher = more recent)
            total_rows: Total number of rows
            decay_rate: Decay rate for exponential function (default: 0.2)
            
        Returns:
            Recency boost multiplier (between 0 and 1)
        """
        if total_rows <= 1:
            return 1.0
        
        # Normalize position to 0-1 range (0 = oldest, 1 = newest)
        normalized_position = row_index / (total_rows - 1)
        
        # Exponential boost: newer rows get higher boost
        # boost = e^(decay_rate * (normalized_position - 1))
        # This gives boost ≈ 0.135 for oldest (position=0) and 1.0 for newest (position=1) with decay_rate=0.2
        boost = np.exp(decay_rate * (normalized_position - 1))
        
        return float(boost)
    
    def search_qa(
        self,
        query: str,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar Q&A pairs in PostgreSQL using vector similarity
        
        Args:
            query: Search query
            n_results: Number of results to return (default: 1)
            
        Returns:
            List of dictionaries with query, answer, and similarity score
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Connect to PostgreSQL
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Query to get all Q&A pairs with their embeddings
            # Using PostgreSQL's vector similarity with pgvector extension
            # If pgvector is installed, we can use cosine distance operator (<=>)
            # Otherwise, we'll fetch all and calculate similarity in Python
            
            try:
                # Try using pgvector extension for efficient similarity search
                query_sql = """
                    SELECT 
                        question,
                        answer,
                        question_vector,
                        1 - (question_vector <=> %s::vector) as similarity
                    FROM public.documents
                    WHERE answer IS NOT NULL AND answer != ''
                    LIMIT %s
                """
                # ORDER BY question_vector <=> %s::vector

                cursor.execute(query_sql, (query_embedding, n_results))  # Fetch more for reranking
                
            except psycopg2.Error as e:
                # Fallback: fetch all records and calculate similarity in Python
                print(f"pgvector not available, using Python-based similarity: {e}")
                query_sql = """
                    SELECT 
                        question,
                        answer,
                        question_vector
                    FROM public.documents
                    WHERE answer IS NOT NULL AND answer != ''
                """
                cursor.execute(query_sql)
            
            rows = cursor.fetchall()
            
            # Close connection
            cursor.close()
            conn.close()
            
            if not rows:
                return []
            
            # Process results
            results = []
            total_rows = len(rows)
            
            # If pgvector was used, rows already have similarity scores
            if len(rows[0]) == 4:  # question, answer, question_vector, similarity
                for idx, row in enumerate(rows):
                    base_similarity = float(row[3])
                    
                    # Apply recency boost based on row position
                    recency_boost = self._calculate_recency_boost(idx, total_rows, decay_rate=0.2)
                    boosted_similarity = base_similarity * recency_boost
                    
                    results.append({
                        "query": row[0],
                        "answer": row[1],
                        "similarity": boosted_similarity,
                        "base_similarity": base_similarity,
                    })
            else:
                # Calculate similarity in Python
                for idx, row in enumerate(rows):
                    question = row[0]
                    answer = row[1]
                    question_vector = row[2]
                    
                    # Calculate cosine similarity
                    base_similarity = self._cosine_similarity(query_embedding, question_vector)
                    
                    # Apply recency boost based on row position
                    recency_boost = self._calculate_recency_boost(idx, total_rows, decay_rate=0.2)
                    boosted_similarity = base_similarity * recency_boost
                    
                    results.append({
                        "query": question,
                        "answer": answer,
                        "similarity": boosted_similarity,
                        "base_similarity": base_similarity,
                        "recency_boost": recency_boost
                    })
            
            # Sort by boosted similarity (descending) and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:n_results]
            
            return results
            
        except Exception as e:
            print(f"Error searching Q&A: {e}")
            raise Exception(f"Error searching Q&A in PostgreSQL: {str(e)}")
    
    def store_qa_from_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve Q&A pairs from question_documents_data table and store them in documents table with embeddings
        
        Args:
            document_id: The document ID to filter questions and answers
            
        Returns:
            Dictionary with insertion results
        """
        try:
            # Connect to PostgreSQL
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Retrieve all questions and answers for the given document_id
            select_sql = """
                SELECT question, answer_final
                FROM public.question_documents_data
                WHERE document_id = %s
                AND question IS NOT NULL 
                AND answer_final IS NOT NULL
                AND question != ''
                AND answer_final != ''
            """
            
            cursor.execute(select_sql, (document_id,))
            rows = cursor.fetchall()
            
            if not rows:
                cursor.close()
                conn.close()
                return {
                    "status": "success",
                    "message": f"No Q&A pairs found for document_id: {document_id}",
                    "document_id": document_id,
                    "pairs_stored": 0
                }
            
            # Process each Q&A pair
            inserted_count = 0
            from datetime import datetime
            
            for row in rows:
                question = row[0]
                answer = row[1]
                
                # Generate embeddings
                question_embedding = self.embeddings_model.embed_query(question)
                answer_embedding = self.embeddings_model.embed_query(answer)
                
                # Insert into documents table
                insert_sql = """
                    INSERT INTO public.documents 
                    (question, answer_final, question_vector, answer_vector, source, updated_at)
                    VALUES (%s, %s, %s::vector, %s::vector, %s, %s, %s)
                """
                
                current_time = datetime.now()
                source = f"document_{document_id}"
                
                cursor.execute(insert_sql, (
                    question,
                    answer,
                    question_embedding,
                    answer_embedding,
                    source,
                    current_time
                ))
                
                inserted_count += 1
            
            # Commit the transaction
            conn.commit()
            
            cursor.close()
            conn.close()
            
            return {
                "status": "success",
                "message": f"Successfully stored Q&A pairs from document {document_id}",
                "document_id": document_id,
                "pairs_stored": inserted_count
            }
            
        except Exception as e:
            print(f"Error storing Q&A from document: {e}")
            raise Exception(f"Error storing Q&A pairs: {str(e)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Check PostgreSQL connection health
        
        Returns:
            Dictionary with health status information
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            return {
                "status": "healthy",
                "postgres_connection": "connected",
                "database": self.db_config["database"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
