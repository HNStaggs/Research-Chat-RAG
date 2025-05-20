from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from datetime import datetime

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
    is_cached = Column(Boolean, default=False)
    
    # HR-specific fields
    hr_categories = Column(ARRAY(String))  # Categories like 'employee_engagement', 'performance', etc.
    hr_relevance_scores = Column(JSONB)  # Store relevance scores for different HR categories
    practical_implications = Column(Text)  # Store key practical implications for HR
    methodology_type = Column(String)  # e.g., 'quantitative', 'qualitative', 'mixed'
    industry_context = Column(ARRAY(String))  # Relevant industries or sectors
    
    # Relationships
    cache_entry = relationship("CachedArticle", back_populates="article", uselist=False)
    search_history = relationship("SearchHistory", back_populates="articles")
    metrics_analysis = relationship("MetricsAnalysis", back_populates="article")

class SearchHistory(Base):
    __tablename__ = 'search_history'
    
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    result_count = Column(Integer)
    user_id = Column(String)  # For future user authentication
    article_ids = Column(ARRAY(Integer))  # Store IDs of returned articles
    search_category = Column(String)  # e.g., 'metrics', 'research', 'best_practices'
    
    # Relationships
    articles = relationship("PubMedArticle", secondary="search_history_articles")

class CachedArticle(Base):
    __tablename__ = 'cached_articles'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('pubmed_articles.id'), unique=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    cache_priority = Column(Integer, default=0)  # For cache eviction strategy
    hr_relevance_score = Column(Float, default=0.0)  # Overall HR relevance score
    
    # Relationships
    article = relationship("PubMedArticle", back_populates="cache_entry")

class MetricsAnalysis(Base):
    __tablename__ = 'metrics_analysis'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('pubmed_articles.id'))
    metric_type = Column(String)  # e.g., 'engagement', 'productivity', 'turnover'
    metric_name = Column(String)
    metric_description = Column(Text)
    measurement_method = Column(Text)
    benchmark_values = Column(JSONB)  # Store benchmark data
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article = relationship("PubMedArticle", back_populates="metrics_analysis")

# Association table for many-to-many relationship between SearchHistory and PubMedArticle
class SearchHistoryArticle(Base):
    __tablename__ = 'search_history_articles'
    
    search_history_id = Column(Integer, ForeignKey('search_history.id'), primary_key=True)
    article_id = Column(Integer, ForeignKey('pubmed_articles.id'), primary_key=True)
    rank = Column(Integer)  # Store the rank of the article in search results
    relevance_score = Column(Float)  # Store the relevance score if available
    hr_relevance_score = Column(Float)  # Store HR-specific relevance score 