import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaDBManager:
    """Manages ChromaDB connections and operations"""
    
    def __init__(self):
        self.use_remote = os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true"
        
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
        
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client based on configuration"""
        if self.use_remote:
            chroma_host = os.getenv("CHROMA_HOST")
            chroma_port = int(os.getenv("CHROMA_PORT", 8000))
            
            client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            print(f"Connected to remote ChromaDB at {chroma_host}:{chroma_port}")
        else:
            client = chromadb.PersistentClient(path="./chroma_db")
            print("Using local ChromaDB storage at ./chroma_db")
        
        return client
    
    def _get_or_create_collection(self):
        """Get or create the compliance documents collection"""
        collection_name = "compliance_documents"
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Compliance documents and embeddings"}
            )
            return collection
        except Exception as e:
            print(f"Error initializing collection: {e}")
            return None
    
    def add_document_chunks(
        self,
        chunks: List[str],
        doc_id: str,
        filename: str,
        document_type: str,
        department: Optional[str] = None,
        tags: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None
    ) -> int:
        """
        Add document chunks to ChromaDB with metadata
        
        Args:
            chunks: List of text chunks
            doc_id: Unique document identifier
            filename: Original filename
            document_type: Type of document
            department: Department associated with document
            tags: List of tags
            page_numbers: Page number for each chunk
            
        Returns:
            Number of chunks stored
        """
        if not self.collection:
            raise Exception("ChromaDB collection not initialized")
        
        stored_chunks = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            
            # Generate embedding
            embedding = self.embeddings_model.embed_query(chunk)
            
            # Prepare metadata
            chunk_metadata = {
                "document_id": doc_id,
                "filename": filename,
                "document_type": document_type,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            
            if page_numbers and idx < len(page_numbers):
                chunk_metadata["page_number"] = page_numbers[idx]
            
            if department:
                chunk_metadata["department"] = department
            
            if tags:
                chunk_metadata["tags"] = ",".join(tags)
            
            # Add to ChromaDB
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            stored_chunks += 1
        
        return stored_chunks
    
    def search_documents(
        self,
        query: str,
        n_results: int = 1,
        document_type: Optional[str] = None,
        department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            document_type: Filter by document type
            department: Filter by department
            
        Returns:
            List of search results with metadata
        """
        if not self.collection:
            print("ChromaDB collection not initialized")
            raise Exception("ChromaDB collection not initialized")
        
        print(f"Searching ChromaDB for: {query} (n_results={n_results})")
        
        # Generate query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        print(f"Generated embedding with {len(query_embedding)} dimensions")
        
        # Prepare where filter
        where_filter = None
        if document_type and department:
            # Use $and operator for multiple conditions
            where_filter = {
                "$and": [
                    {"document_type": document_type},
                    {"department": department}
                ]
            }
        elif document_type:
            where_filter = {"document_type": document_type}
        elif department:
            where_filter = {"department": department}
        
        print(f"Using filter: {where_filter}")
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        print(f"ChromaDB query returned: {len(results['ids'][0]) if results and results['ids'] else 0} results")
        
        # Format results
        formatted_results = []
        if results and results['ids']:
            for idx in range(len(results['ids'][0])):
                # ChromaDB returns distances (lower = more similar)
                distance = results['distances'][0][idx] if 'distances' in results else None
                
                # Convert distance to similarity score (0 to 1)
                # ChromaDB uses squared L2 distance by default
                if distance is not None:
                    import math
                    # Take square root to get actual L2 distance
                    actual_distance = math.sqrt(distance) if distance >= 0 else 0
                    # Convert to similarity: 1 / (1 + distance)
                    # Lower distances yield higher similarity scores
                    similarity = 1 / (1 + actual_distance)
                else:
                    similarity = None
                
                formatted_results.append({
                    "id": results['ids'][0][idx],
                    "content": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx],
                    "similarity": similarity,
                    "distance": distance
                })
        
        print(f"Returning {len(formatted_results)} formatted results")
        return formatted_results
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the database
        
        Returns:
            List of documents with their metadata
        """
        if not self.collection:
            raise Exception("ChromaDB collection not initialized")
        
        # Get all items from collection
        results = self.collection.get()
        
        # Extract unique document IDs and their metadata
        documents = {}
        for metadata in results['metadatas']:
            doc_id = metadata.get('document_id')
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata.get('filename'),
                    "document_type": metadata.get('document_type'),
                    "department": metadata.get('department'),
                    "tags": metadata.get('tags'),
                    "total_chunks": metadata.get('total_chunks', 0)
                }
        
        return list(documents.values())
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete a document and all its chunks
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        if not self.collection:
            raise Exception("ChromaDB collection not initialized")
        
        # Get all chunks for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if not results['ids']:
            raise ValueError("Document not found")
        
        # Delete all chunks
        self.collection.delete(ids=results['ids'])
        
        return len(results['ids'])
    
    def delete_all_documents(self) -> int:
        """
        Delete all documents from the collection
        
        Returns:
            Number of chunks deleted
        """
        if not self.collection:
            raise Exception("ChromaDB collection not initialized")
        
        # Get all items from collection
        results = self.collection.get()
        
        if not results['ids']:
            return 0
        
        # Delete all chunks
        self.collection.delete(ids=results['ids'])
        
        return len(results['ids'])
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of ChromaDB connection
        
        Returns:
            Dictionary with health status information
        """
        try:
            collections = self.client.list_collections()
            return {
                "status": "healthy",
                "chroma_connection": "connected",
                "chroma_type": "remote" if self.use_remote else "local",
                "collections": [col.name for col in collections]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
