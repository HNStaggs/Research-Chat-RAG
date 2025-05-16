![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Research Chat RAG

A powerful research assistant that combines PubMed paper search with RAG (Retrieval-Augmented Generation) capabilities using Claude AI.

## Features

- Real-time PubMed paper search and analysis
- Vector-based similarity search for relevant papers
- PostgreSQL storage with pgvector for efficient vector operations
- Claude AI integration for intelligent responses
- Citation management and formatting
- GPU acceleration support

## Project Structure

```
research-chat-rag/
├── src/                    # Source code
│   ├── api/               # API endpoints and main application
│   │   └── app.py        # Main Chainlit application
│   ├── config/           # Configuration management
│   │   └── settings.py   # Settings and environment variables
│   ├── database/         # Database management
│   │   └── init_db.py    # Database initialization
│   ├── models/           # Database models
│   │   └── pubmed.py     # PubMed article models
│   ├── services/         # Business logic
│   │   └── pubmed_service.py  # PubMed interaction service
│   └── utils/            # Utility functions
│       └── memory.py     # Memory management utilities
├── tests/                # Test files
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── config/              # Configuration files
├── .env.example         # Example environment variables
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Create a PostgreSQL database and install the pgvector extension:
```sql
CREATE DATABASE research_chat;
\c research_chat
CREATE EXTENSION vector;
```

2. Create a `.env` file with your configuration:
```
ANTHROPIC_API_KEY=your_claude_api_key_here
NCBI_API_KEY=your_ncbi_api_key_here
ENTREZ_EMAIL=your_email@example.com
DATABASE_URL=postgresql://username:password@localhost:5432/research_chat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
chainlit run src/api/app.py
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Claude API key
- `NCBI_API_KEY`: Your NCBI/PubMed API key
- `ENTREZ_EMAIL`: Email for PubMed API access
- `DATABASE_URL`: PostgreSQL connection string
- `EMBEDDING_MODEL`: Model for generating embeddings (default: sentence-transformers/all-mpnet-base-v2)
- `TEMPERATURE`: Temperature for Claude responses (default: 0.7)
- `MAX_TOKENS`: Maximum tokens for responses (default: 2000)

## Features

1. **PubMed Integration**
   - Real-time paper search
   - Automatic metadata extraction
   - Citation generation

2. **Vector Search**
   - Efficient similarity search using pgvector
   - Automatic embedding generation
   - Hybrid search capabilities

3. **Claude AI Integration**
   - Context-aware responses
   - Citation inclusion
   - Natural language understanding

4. **Database Management**
   - PostgreSQL with vector operations
   - Efficient data storage and retrieval
   - Automatic schema management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Features
- 🚀 GPU-accelerated code generation using SantaCoder/CodeLlama
- 📚 RAG system for documentation reference
- 💻 Support for multiple programming languages
- 🔄 Real-time system monitoring
- 📊 Performance tracking
- 🛠️ Customizable generation parameters
- 💾 Offline operation (runs on local GPU) - no internet required after setup, improves safety of proprietery codebase information
  

## System Requirements
- Python 3.13+
- CUDA-capable GPU (tested with RTX A2000 8GB)
- 16GB+ RAM recommended
- Windows/Linux/MacOS  # Initially built on Windows

## 🚀 Quick Start

### Installation using bash

#### Step 1: Clone the repository in a new project folder
1. `mkdir translate-demo-app`
2. `cd translate-demo-app`
3. `git clone https://github.com/hnstaggs/CodeHelperRAG.git`

#### Step 2: Create a virtual environment
4. `python -m venv venv`
5. `cd venv`
6. `Scripts\activate`  # On Windows
  
#### Step 3: Install required packages
7. `cd ..`  # Back to root directory
8. `pip install -r requirements.txt`

#### Step 4: Download Models
9. `python setup.py`

#### Step 5: Setup Directory
10. `mkdir -p models docs cache`

#### Step 6: Upload/save in-house documentation in docs folder

#### Step 7: Run the app
11. `streamlit run app.py`

#### Step 5: App should automatically launch in your browser
* If not, Locate URL provided by streamlit console output and paste into web browser.
* Example: http://localhost:8501
* Test app functionality on your machine

#### Step 6: Deploying to Streamlit Cloud to Share with Others
* Push your code to your GitHub repository
* Log in to Streamlit Cloud
* Give Streamlit access to your github
* Click "New app"
* Select your repository, branch, and main file path (app.py)
* Click "Deploy"
* Your app will be live in the cloud at https://[code-helper]streamlit.app

## 📦 Dependencies
* Windows  # Built on windows, might need to update code for Mac or Linux run
* Public facing github repo for project
* Up-to-date requirements.txt for streamlit deploy
* Streamlit
* Transformers
* PyTorch
* langchain
* Git

## 💻 Usage
* Select programming language
* Enter your code request
* Click "Generate Code"

### Features in Detail
* Multiple programming language support
* Context-aware generation
* Code explanation generation
* Syntax highlighting
* Copy to clipboard functionality
* RAG System
* Documentation reference
* PDF and TXT support
* Automatic text chunking
* Vector similarity search
* GPU acceleration
* Memory optimization
* Cache management
* Performance monitoring
* System statistics

## 🛠️  Model Configuration: Adjust model parameters in utils.py
* temperature = 0.4
* top_p = 0.85
* top_k = 50

## ⚠️ Limitations & Issues
* Processing speed depends on your hardware capabilities and the latest CUDA driver support
`python manage_gpu.py`  # Clear GPU memory
  
## 🙏 Acknowledgments/Credits
* Built with Streamlit Apache 2.0
* Uses Hugging Face Transformers Apache 2.0
* Uses LangChain for RAG 

### 📚 Additional Resources, Citations, and Attributes
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Documentation](https://huggingface.co/docs/hub/index)

# Made with ❤️ by Halee

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
