from langchain_core.tools import tool
from src.services.pinecone_service import PineconeService
from src.config.settings import settings

@tool
class PineconeTool:
    name = "pinecone_search"
    description = "Performs vector similarity search using Pinecone."

    def __init__(self):
        self.service = PineconeService(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT,
            index_name=settings.PINECONE_INDEX_NAME
        )

    def run(self, embedding, top_k=3):
        return self.service.search_similar(embedding, top_k=top_k) 