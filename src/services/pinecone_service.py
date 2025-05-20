import pinecone
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

class PineconeService:
    def __init__(self, api_key: str, environment: str, index_name: str = "research-chat"):
        """Initialize Pinecone service"""
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=768,  # Dimension for sentence-transformers/all-mpnet-base-v2
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def store_embeddings(self, article_id: str, embedding: List[float], metadata: Dict) -> None:
        """Store article embedding in Pinecone"""
        self.index.upsert(
            vectors=[{
                "id": article_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
    
    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Search for similar articles using vector similarity"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    
    def delete_article(self, article_id: str) -> None:
        """Delete an article from Pinecone"""
        self.index.delete(ids=[article_id]) 