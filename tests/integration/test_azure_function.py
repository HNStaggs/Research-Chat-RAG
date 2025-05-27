import pytest
import azure.functions as func
from src.api.azure_function import main

@pytest.mark.integration
def test_azure_function_http_trigger():
    # Create a mock HTTP request
    req = func.HttpRequest(
        method='POST',
        body=b'{"type": "research", "query": "test query"}',
        url='/api/research'
    )
    
    # Call the function
    response = main(req)
    
    # Assert response
    assert response.status_code == 200
    assert response.get_body() is not None

@pytest.mark.integration
def test_azure_function_file_upload():
    # Create a mock file upload request
    req = func.HttpRequest(
        method='POST',
        body=b'test file content',
        url='/api/upload',
        files={'file': ('test.csv', b'test content')}
    )
    
    # Call the function
    response = main(req)
    
    # Assert response
    assert response.status_code == 200
    assert response.get_body() is not None

@pytest.mark.integration
def test_azure_function_data_analysis():
    # Create a mock data analysis request
    req = func.HttpRequest(
        method='POST',
        body=b'{"type": "data_analysis", "blob_name": "test.csv", "request": "analyze data"}',
        url='/api/analyze'
    )
    
    # Call the function
    response = main(req)
    
    # Assert response
    assert response.status_code == 200
    assert response.get_body() is not None

@pytest.mark.integration
def test_azure_function_error_handling():
    # Test with invalid request
    req = func.HttpRequest(
        method='POST',
        body=b'invalid json',
        url='/api/research'
    )
    
    # Call the function
    response = main(req)
    
    # Assert error response
    assert response.status_code == 400 