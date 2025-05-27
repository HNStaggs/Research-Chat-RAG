import pytest
from src.services.bing_service import BingGroundingService
from unittest.mock import Mock, patch

def test_bing_service_initialization():
    with patch('azure.search.documents.SearchClient') as mock_client:
        service = BingGroundingService()
        assert service.search_client is not None

def test_search_function():
    with patch('azure.search.documents.SearchClient') as mock_client:
        service = BingGroundingService()
        mock_results = [{"title": "Test Result"}]
        service.search_client.search.return_value = mock_results
        
        results = service.search("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Test Result"

def test_create_index():
    with patch('azure.search.documents.indexes.SearchIndexClient') as mock_index_client:
        service = BingGroundingService()
        mock_index_client.list_indexes.return_value = []
        
        service.create_index()
        service.index_client.create_index.assert_called_once()

def test_add_documents():
    with patch('azure.search.documents.SearchClient') as mock_client:
        service = BingGroundingService()
        test_documents = [{"id": "1", "content": "Test content"}]
        
        service.add_documents(test_documents)
        service.search_client.upload_documents.assert_called_once_with(test_documents) 