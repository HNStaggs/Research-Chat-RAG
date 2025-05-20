from sqlalchemy import create_engine
from src.models.pubmed import Base

def init_db(database_url):
    """Initialize database connection and create tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine 