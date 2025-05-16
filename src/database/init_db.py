from sqlalchemy import create_engine
from src.models.pubmed import Base

def init_db(database_url):
    """Initialize database connection and create tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine

def create_vector_extension(database_url):
    """Create pgvector extension if it doesn't exist"""
    engine = create_engine(database_url)
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit() 