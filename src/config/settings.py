from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str
    NCBI_API_KEY: str
    PINECONE_API_KEY: str
    
    # Email Configuration
    ENTREZ_EMAIL: str
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://localhost/research_chat"
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "research-chat"
    
    # Azure Storage Configuration
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER: str = "datasets"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings instance
settings = Settings() 