import pytest
from src.services.pinecone_service import PineconeService
from unittest.mock import Mock, patch

def test_pinecone_service_initialization():
    with patch('pinecone.init') as mock_init:
        service = PineconeService(
            api_key="test_key",
            environment="test_env",
            index_name="test-index"
        )
        mock_init.assert_called_once_with(
            api_key="test_key",
            environment="test_env"
        )

def test_store_embeddings():
    with patch('pinecone.Index') as mock_index:
        service = PineconeService(
            api_key="test_key",
            environment="test_env"
        )
        service.store_embeddings(
            article_id="test123",
            embedding=[0.1, 0.2, 0.3],
            metadata={"title": "Test Article"}
        )
        service.index.upsert.assert_called_once()

def test_search_similar():
    with patch('pinecone.Index') as mock_index:
        service = PineconeService(
            api_key="test_key",
            environment="test_env"
        )
        mock_results = Mock()
        mock_results.matches = [{"id": "test123", "metadata": {"title": "Test Article"}}]
        service.index.query.return_value = mock_results
        
        results = service.search_similar([0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "test123"

def test_delete_article():
    with patch('pinecone.Index') as mock_index:
        service = PineconeService(
            api_key="test_key",
            environment="test_env"
        )
        service.delete_article("test123")
        service.index.delete.assert_called_once_with(ids=["test123"]) 