from langchain_core.agents import AgentExecutor

class BaseAgent:
    def __init__(self, tools, memory=None):
        self.tools = tools
        self.memory = memory
        self.executor = AgentExecutor.from_tools(tools, memory=memory)

    async def run(self, input):
        return await self.executor.ainvoke(input) 