![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-orange.svg)
![Chainlit](https://img.shields.io/badge/Chainlit-1.0.0-purple.svg)
![Claude](https://img.shields.io/badge/Claude-3.0-yellow.svg)
![RAG](https://img.shields.io/badge/RAG-Enabled-brightgreen.svg)
![Gen AI](https://img.shields.io/badge/Generative_AI-Powered-blue.svg)
![Agents](https://img.shields.io/badge/Agents-LangChain%2FLangGraph-orange.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-blueviolet.svg)
![Azure](https://img.shields.io/badge/Azure-Search-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# HR Personnel Analytics Agent

An intelligent HR analyst that combines PubMed data analysis, vector search, and data visualization capabilities using LangChain/LangGraph agents.

## Features

1. **Research Analysis**
   - PubMed paper search and analysis
   - Automatic metadata extraction
   - Citation generation
   - Vector similarity search using Pinecone
   - Bing grounding for real-time web context

2. **Data Analysis & Visualization**
   - Dataset upload and storage in Azure Blob Storage
   - Interactive data visualization using Plotly
   - Statistical analysis and insights
   - Support for multiple graph types:
     - Line graphs
     - Bar charts
     - Scatter plots
     - Histograms
     - Box plots

3. **Agent Architecture**
   - Research Agent for paper analysis
   - Data Analysis Agent for visualization
   - LangGraph-based workflow management
   - Tool-based architecture for extensibility

4. **Azure Integration**
   - Azure Functions deployment
   - Azure Blob Storage for datasets
   - Azure Cognitive Search for web grounding
   - Azure Key Vault for secure key management
   - Managed identity and security
   - CI/CD pipeline with GitHub Actions

## Project Structure

```
src/
├── agents/                # Agent definitions
│   ├── base_agent.py
│   ├── research_agent.py
│   └── data_analysis_agent.py
├── chains/               # LangGraph workflows
│   └── research_chain.py
├── tools/               # LangChain tools
│   ├── pubmed_tool.py
│   ├── pinecone_tool.py
│   ├── bing_grounding_tool.py
│   └── data_analysis_tool.py
├── services/           # Service implementations
│   ├── pubmed_service.py
│   ├── pinecone_service.py
│   └── bing_service.py
├── api/                 # API endpoints
│   └── azure_function.py
├── config/             # Configuration
├── database/          # Database management
├── models/           # Data models
└── utils/           # Utility functions
```

## Prerequisites

- Python 3.11+
- Azure subscription
- PostgreSQL 12+ (for metadata storage)
- Git
- pip (Python package installer)
- LLM API (Anthropic Claude used for this demo)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-chat-rag.git
cd research-chat-rag
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
ANTHROPIC_API_KEY=your_claude_api_key_here
NCBI_API_KEY=your_ncbi_api_key_here
ENTREZ_EMAIL=your_email@example.com
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-search-api_key
AZURE_BING_SEARCH_KEY=your-bing-search-key
AZURE_KEY_VAULT_NAME=your-keyvault
```

5. Deploy to Azure:
```bash
# Login to Azure
az login

# Create resource group
az group create --name research-chat-rg --location eastus

# Create Azure Cognitive Search
az search service create --name your-search-service --resource-group research-chat-rg --sku standard --location eastus

# Create Azure Key Vault
az keyvault create --name your-keyvault --resource-group research-chat-rg --location eastus

# Store the search API key in Key Vault
az keyvault secret set --vault-name your-keyvault --name "search-api-key" --value "your-search-api-key"

# Store the Bing Search key in Key Vault
az keyvault secret set --vault-name your-keyvault --name "bing-search-key" --value "your-bing-search-key"

# Deploy using Bicep
az deployment group create \
  --resource-group research-chat-rg \
  --template-file azure/main.bicep \
  --parameters functionAppName=your-function-app-name \
  --parameters storageAccountName=yourstorageaccount \
  --parameters appServicePlanName=your-app-service-plan \
  --parameters searchServiceName=your-search-service \
  --parameters keyVaultName=your-keyvault
```

## API Usage

### Research Analysis
```http
POST /api/analyze
Content-Type: application/json
{
    "type": "research",
    "query": "What are the latest developments in preventing employee burnout?"
}
```

### Data Analysis
1. Upload dataset:
```http
POST /api/upload
Content-Type: multipart/form-data
file: your_dataset.csv
```

2. Request analysis:
```http
POST /api/analyze
Content-Type: application/json
{
    "type": "data_analysis",
    "blob_name": "uploaded_file_name",
    "request": "Create a graph showing salaries for department X and department Y"
}
```

## Development

### Adding New Agents
1. Create a new agent class in `src/agents/`
2. Implement required tools in `src/tools/`
3. Add agent to the appropriate chain in `src/chains/`

### Adding New Tools
1. Create a new tool class in `src/tools/`
2. Decorate with `@tool`
3. Implement the `run` method
4. Add to the appropriate agent

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:
- Automatic testing on push
- Deployment to Azure Functions
- Environment variable management
- Security scanning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangChain and LangGraph
- Uses Azure Functions for serverless deployment
- Powered by Claude for natural language understanding
- Data visualization with Plotly
- Web grounding with Azure Cognitive Search

## 🙏 Acknowledgments/Credits
* Built with Chainlit
* Uses Hugging Face Transformers Apache 2.0
* Sample HR personnel dataset from: https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data

### 📚 Additional Resources, Citations, and Attributes
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Documentation](https://huggingface.co/docs/hub/index)
* [Azure Cognitive Search Documentation](https://docs.microsoft.com/en-us/azure/search/)

# Made with ❤️ by Halee

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## Documentation References

### Azure Services
- [Azure Functions Documentation](https://docs.microsoft.com/en-us/azure/azure-functions/)
- [Azure Blob Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/blobs/)
- [Azure Cognitive Search Documentation](https://docs.microsoft.com/en-us/azure/search/)
- [Azure Key Vault Documentation](https://docs.microsoft.com/en-us/azure/key-vault/)

### SDKs and APIs
- [Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python)
- [Azure Functions Python Guide](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python)
