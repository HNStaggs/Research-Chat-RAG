import pytest
from src.chains.research_chain import ResearchChain
from src.services.pinecone_service import PineconeService
from src.services.bing_service import BingGroundingService

@pytest.mark.integration
def test_research_chain_with_services():
    # Initialize services with test credentials
    pinecone_service = PineconeService(
        api_key="test_key",
        environment="test_env"
    )
    bing_service = BingGroundingService()
    
    # Create research chain
    chain = ResearchChain()
    
    # Test query processing
    result = chain.run("test query")
    assert result is not None
    assert isinstance(result, dict)

@pytest.mark.integration
def test_research_chain_with_grounding():
    chain = ResearchChain()
    
    # Test with grounding results
    grounding_results = [{"title": "Test Result", "content": "Test Content"}]
    result = chain.run("test query", grounding_results)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "articles" in result
    assert "grounding_results" in result

@pytest.mark.integration
def test_research_chain_error_handling():
    chain = ResearchChain()
    
    # Test with invalid input
    result = chain.run("")
    assert result is not None
    assert "error" in result or "status" in result 