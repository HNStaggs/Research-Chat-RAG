from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import QueryType
from src.config.settings import settings

class BingGroundingService:
    def __init__(self):
        self.search_client = SearchClient(
            endpoint=f"https://{settings.AZURE_SEARCH_SERVICE_NAME}.search.windows.net",
            index_name="research-index",
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
        )
        
        self.index_client = SearchIndexClient(
            endpoint=f"https://{settings.AZURE_SEARCH_SERVICE_NAME}.search.windows.net",
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
        )

    async def create_index(self):
        """Create the search index if it doesn't exist"""
        index_name = "research-index"
        if index_name not in [index.name for index in self.index_client.list_indexes()]:
            index = SearchIndex(
                name=index_name,
                fields=[
                    {"name": "id", "type": "Edm.String", "key": True},
                    {"name": "content", "type": "Edm.String", "searchable": True},
                    {"name": "title", "type": "Edm.String", "searchable": True},
                    {"name": "url", "type": "Edm.String"},
                    {"name": "metadata", "type": "Edm.String"}
                ]
            )
            self.index_client.create_index(index)

    async def add_documents(self, documents):
        """Add documents to the search index"""
        self.search_client.upload_documents(documents)

    async def search(self, query: str, top_k: int = 5):
        """Search for relevant documents"""
        results = self.search_client.search(
            search_text=query,
            query_type=QueryType.SEMANTIC,
            query_language="en-us",
            semantic_configuration_name="default",
            top=top_k
        )
        return [doc for doc in results] 