import pytest
import pandas as pd
from src.tools.data_analysis_tool import DataAnalysisTool
from unittest.mock import Mock, patch

def test_load_dataset():
    with patch('azure.storage.blob.BlobServiceClient') as mock_blob:
        tool = DataAnalysisTool()
        mock_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        tool.container_client.get_blob_client().download_blob().readall.return_value = mock_data.to_csv()
        
        result = tool.load_dataset("test.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

def test_generate_graph():
    tool = DataAnalysisTool()
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6],
        'color': ['red', 'blue', 'green']
    })
    
    result = tool.generate_graph(
        df=df,
        graph_type="line",
        x_col="x",
        y_col="y",
        color_col="color"
    )
    assert isinstance(result, dict)

def test_analyze_data():
    tool = DataAnalysisTool()
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'B', 'A']
    })
    
    result = tool.analyze_data(df)
    assert isinstance(result, dict)
    assert 'summary' in result
    assert 'missing_values' in result
    assert 'data_types' in result

def test_run_method():
    with patch('azure.storage.blob.BlobServiceClient') as mock_blob:
        tool = DataAnalysisTool()
        mock_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        tool.container_client.get_blob_client().download_blob().readall.return_value = mock_data.to_csv()
        
        result = tool.run(
            blob_name="test.csv",
            analysis_type="analysis"
        )
        assert isinstance(result, dict)
        assert 'summary' in result 