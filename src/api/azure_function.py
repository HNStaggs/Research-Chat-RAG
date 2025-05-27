import azure.functions as func
import json
from src.chains.research_chain import ResearchChain
from src.agents.data_analysis_agent import DataAnalysisAgent
from src.services.bing_service import BingGroundingService
from azure.storage.blob import BlobServiceClient
from src.config.settings import settings
import uuid

# Initialize agents
research_chain = ResearchChain()
data_analysis_agent = DataAnalysisAgent()
bing_grounding_service = BingGroundingService()

# Initialize Azure Storage
blob_service_client = BlobServiceClient.from_connection_string(
    settings.AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    settings.AZURE_STORAGE_CONTAINER
)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Handle file upload
        if req.method == "POST" and req.files:
            file = req.files.get('file')
            if file:
                # Generate unique blob name
                blob_name = f"{uuid.uuid4()}_{file.filename}"
                # Upload to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.upload_blob(file.read())
                return func.HttpResponse(
                    json.dumps({"blob_name": blob_name}),
                    status_code=200
                )

        # Handle data analysis request
        if req.method == "POST":
            req_body = req.get_json()
            request_type = req_body.get('type')
            
            if request_type == "data_analysis":
                blob_name = req_body.get('blob_name')
                analysis_request = req_body.get('request')
                result = await data_analysis_agent.analyze_request(
                    analysis_request,
                    blob_name
                )
                return func.HttpResponse(
                    json.dumps(result),
                    status_code=200
                )
            
            # Handle research request
            elif request_type == "research":
                user_input = req_body.get('query')
                # Add grounding results to the research chain
                grounding_results = await bing_grounding_service.search(user_input)
                result = await research_chain.run(user_input, grounding_results)
                return func.HttpResponse(
                    json.dumps({'result': result}),
                    status_code=200
                )
            
        return func.HttpResponse(
            "Invalid request",
            status_code=400
        )
            
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500) 