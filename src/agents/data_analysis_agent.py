from .base_agent import BaseAgent
from src.tools.data_analysis_tool import DataAnalysisTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class DataAnalysisAgent(BaseAgent):
    def __init__(self, memory=None):
        tools = [DataAnalysisTool()]
        super().__init__(tools, memory)
        
        # Create a prompt template for data analysis
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis expert. Your task is to:
            1. Understand the user's request for data analysis or visualization
            2. Determine the appropriate graph type and parameters
            3. Generate the requested analysis or visualization
            4. Provide insights about the results
            
            Available graph types: line, bar, scatter, histogram, box
            """),
            ("user", "{input}")
        ])
        
        # Create the chain
        self.chain = (
            {"input": RunnablePassthrough()}
            | self.prompt
            | StrOutputParser()
        )

    async def analyze_request(self, request: str, blob_name: str) -> dict:
        """Process a data analysis request"""
        # First, use the LLM to understand the request
        analysis_plan = await self.chain.ainvoke(request)
        
        # Then, use the data analysis tool to execute the plan
        result = await self.tools[0].run(
            blob_name=blob_name,
            analysis_type="graph" if "graph" in request.lower() else "analysis",
            graph_params=self._parse_graph_params(analysis_plan)
        )
        
        return {
            "analysis_plan": analysis_plan,
            "result": result
        }

    def _parse_graph_params(self, analysis_plan: str) -> dict:
        """Parse the analysis plan to extract graph parameters"""
        # This is a simplified version - you might want to make this more sophisticated
        params = {}
        if "line" in analysis_plan.lower():
            params["graph_type"] = "line"
        elif "bar" in analysis_plan.lower():
            params["graph_type"] = "bar"
        elif "scatter" in analysis_plan.lower():
            params["graph_type"] = "scatter"
        elif "histogram" in analysis_plan.lower():
            params["graph_type"] = "histogram"
        elif "box" in analysis_plan.lower():
            params["graph_type"] = "box"
            
        # You would need to extract x_col, y_col, and color_col from the analysis plan
        # This is a placeholder - implement proper parsing based on your needs
        params["x_col"] = "default_x_column"
        params["y_col"] = "default_y_column"
        
        return params 