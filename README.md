![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-orange.svg)
![Chainlit](https://img.shields.io/badge/Chainlit-1.0.0-purple.svg)
![Claude](https://img.shields.io/badge/Claude-3.0-yellow.svg)
![RAG](https://img.shields.io/badge/RAG-Enabled-brightgreen.svg)
![Gen AI](https://img.shields.io/badge/Generative_AI-Powered-blue.svg)
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

## Prerequisites

- Python 3.13+
- PostgreSQL 12+ with admin access
- Git
- pip (Python package installer)

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

4. Set up PostgreSQL database:
```sql
# Connect to PostgreSQL as admin
psql -U postgres

# Create database and extension
CREATE DATABASE research_chat;
\c research_chat
CREATE EXTENSION IF NOT EXISTS vector;
```

5. Get required API keys:
   - Claude API key from [Anthropic Console](https://console.anthropic.com/)
   - NCBI/PubMed API key from [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/)

6. Create environment file:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
# ANTHROPIC_API_KEY=your_claude_api_key_here
# NCBI_API_KEY=your_ncbi_api_key_here
# ENTREZ_EMAIL=your_email@example.com
# DATABASE_URL=postgresql://username:password@localhost:5432/research_chat
```

7. Initialize the database:
```bash
# The database tables will be created on first run
```

8. Run the application:
```bash
chainlit run src/api/app.py
```

9. Access the application:
   - Open your browser and navigate to `http://localhost:8000`
   - The first run will take longer as it downloads required models

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


## 🙏 Acknowledgments/Credits
* Built with Chainlit
* Uses Hugging Face Transformers Apache 2.0

### 📚 Additional Resources, Citations, and Attributes
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Documentation](https://huggingface.co/docs/hub/index)

# Made with ❤️ by Halee

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
