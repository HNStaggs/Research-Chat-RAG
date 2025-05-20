from langchain_core.tools import tool
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import json
from azure.storage.blob import BlobServiceClient
from src.config.settings import settings

@tool
class DataAnalysisTool:
    name = "data_analysis"
    description = "Analyzes datasets and generates visualizations based on user requests."

    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self.blob_service_client.get_container_client(
            settings.AZURE_STORAGE_CONTAINER
        )

    def load_dataset(self, blob_name: str) -> pd.DataFrame:
        """Load dataset from Azure Blob Storage"""
        blob_client = self.container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()
        return pd.read_csv(data)

    def generate_graph(self, df: pd.DataFrame, graph_type: str, 
                      x_col: str, y_col: Optional[str] = None,
                      color_col: Optional[str] = None) -> Dict:
        """Generate various types of graphs using Plotly"""
        if graph_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
        elif graph_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col)
        elif graph_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        elif graph_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col)
        elif graph_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        return json.loads(fig.to_json())

    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistical analysis of the dataset"""
        return {
            "summary": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "correlation": df.corr().to_dict() if df.select_dtypes(include=['float64', 'int64']).shape[1] > 1 else {}
        }

    async def run(self, blob_name: str, analysis_type: str, 
                 graph_params: Optional[Dict] = None) -> Dict:
        """Main execution method for data analysis"""
        df = self.load_dataset(blob_name)
        
        if analysis_type == "graph" and graph_params:
            return self.generate_graph(df, **graph_params)
        elif analysis_type == "analysis":
            return self.analyze_data(df)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}") 