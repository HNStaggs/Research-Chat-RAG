import pytest
from src.config.settings import Settings
from unittest.mock import patch

@pytest.fixture
def test_settings():
    return Settings(
        ANTHROPIC_API_KEY="test_key",
        NCBI_API_KEY="test_key",
        PINECONE_API_KEY="test_key",
        ENTREZ_EMAIL="test@example.com",
        PINECONE_ENVIRONMENT="test_env",
        AZURE_STORAGE_CONNECTION_STRING="test_connection",
        AZURE_SEARCH_SERVICE_NAME="test_search",
        AZURE_SEARCH_API_KEY="test_key",
        AZURE_BING_SEARCH_KEY="test_key",
        AZURE_KEY_VAULT_NAME="test_vault"
    )

@pytest.fixture
def mock_pinecone_service():
    with patch('src.services.pinecone_service.PineconeService') as mock:
        yield mock

@pytest.fixture
def mock_bing_service():
    with patch('src.services.bing_service.BingGroundingService') as mock:
        yield mock

@pytest.fixture
def mock_azure_storage():
    with patch('azure.storage.blob.BlobServiceClient') as mock:
        yield mock

@pytest.fixture
def mock_azure_search():
    with patch('azure.search.documents.SearchClient') as mock:
        yield mock

@pytest.fixture
def sample_dataframe():
    import pandas as pd
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'B', 'A'],
        'date': pd.date_range(start='2024-01-01', periods=5)
    }) 