from langchain_core.tools import tool
from src.services.pubmed_service import PubMedService
from src.config.settings import settings

@tool
class PubMedTool:
    name = "pubmed_search"
    description = "Searches PubMed for research articles."

    def __init__(self):
        self.service = PubMedService(
            email=settings.ENTREZ_EMAIL,
            api_key=settings.NCBI_API_KEY,
            pinecone_service=None,  # Set if needed
            embedding_model=settings.EMBEDDING_MODEL
        )

    async def run(self, query: str):
        return await self.service.fetch_pubmed_data(query) 