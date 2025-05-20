from .base_agent import BaseAgent
from src.tools.pubmed_tool import PubMedTool
from src.tools.pinecone_tool import PineconeTool

class ResearchAgent(BaseAgent):
    def __init__(self, memory=None):
        tools = [PubMedTool(), PineconeTool()]
        super().__init__(tools, memory) 