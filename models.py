from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
import numpy as np

Base = declarative_base()

class PubMedArticle(Base):
    __tablename__ = 'pubmed_articles'
    
    id = Column(Integer, primary_key=True)
    pmid = Column(String, unique=True, nullable=False)
    title = Column(Text)
    abstract = Column(Text)
    authors = Column(Text)
    publication_date = Column(DateTime)
    journal = Column(String)
    keywords = Column(ARRAY(String))
    raw_data = Column(Text)  # Store complete raw PubMed data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with vector embeddings
    embeddings = relationship("ArticleEmbedding", back_populates="article", uselist=False)

class ArticleEmbedding(Base):
    __tablename__ = 'article_embeddings'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('pubmed_articles.id'))
    embedding = Column(ARRAY(Float))  # Store vector embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with article
    article = relationship("PubMedArticle", back_populates="embeddings")

def init_db(database_url):
    """Initialize database connection and create tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def create_vector_extension(database_url):
    """Create pgvector extension if it doesn't exist"""
    engine = create_engine(database_url)
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit() 