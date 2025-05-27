from langchain_core.tools import tool
from src.services.bing_service import BingGroundingService

@tool
class BingGroundingTool:
    name = "bing_grounding"
    description = "Searches and grounds information using Bing Search and Azure Cognitive Search."

    def __init__(self):
        self.service = BingGroundingService()

    async def run(self, query: str, top_k: int = 5):
        """Execute the grounding search"""
        results = await self.service.search(query, top_k)
        return {
            "results": results,
            "source": "bing_grounding"
        } 