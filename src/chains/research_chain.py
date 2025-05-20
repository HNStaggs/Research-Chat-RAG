from langgraph.graph import StateGraph
from src.agents.research_agent import ResearchAgent

# Define the LangGraph workflow
class ResearchChain:
    def __init__(self, memory=None):
        self.agent = ResearchAgent(memory=memory)
        self.graph = StateGraph()
        self.graph.add_node('research', self.agent.run)
        self.graph.set_entry_point('research')

    async def run(self, input):
        return await self.graph.ainvoke(input) 