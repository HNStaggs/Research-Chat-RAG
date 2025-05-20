from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import List, Optional
from datetime import datetime, timedelta
from src.config.settings import settings
from src.models.pubmed import PubMedArticle, SearchHistory, CachedArticle, SearchHistoryArticle

# Create engine and session factory
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_article(article_data: dict) -> Optional[PubMedArticle]:
    """Save a PubMed article to the database."""
    with get_db() as db:
        try:
            article = PubMedArticle(**article_data)
            db.add(article)
            db.commit()
            db.refresh(article)
            return article
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving article: {str(e)}")
            return None

def get_article_by_pmid(pmid: str) -> Optional[PubMedArticle]:
    """Retrieve an article by its PMID."""
    with get_db() as db:
        return db.query(PubMedArticle).filter(PubMedArticle.pmid == pmid).first()

def save_search_history(query: str, article_ids: List[int], user_id: Optional[str] = None) -> Optional[SearchHistory]:
    """Save a search history entry."""
    with get_db() as db:
        try:
            history = SearchHistory(
                query=query,
                article_ids=article_ids,
                result_count=len(article_ids),
                user_id=user_id
            )
            db.add(history)
            db.commit()
            db.refresh(history)
            return history
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving search history: {str(e)}")
            return None

def update_cache_entry(article_id: int, relevance_score: float = 1.0) -> None:
    """Update or create a cache entry for an article."""
    with get_db() as db:
        try:
            cache_entry = db.query(CachedArticle).filter(CachedArticle.article_id == article_id).first()
            if cache_entry:
                cache_entry.last_accessed = datetime.utcnow()
                cache_entry.access_count += 1
                cache_entry.cache_priority = int(cache_entry.access_count * relevance_score)
            else:
                cache_entry = CachedArticle(
                    article_id=article_id,
                    cache_priority=int(relevance_score)
                )
                db.add(cache_entry)
            
            # Update article's cache status
            article = db.query(PubMedArticle).filter(PubMedArticle.id == article_id).first()
            if article:
                article.is_cached = True
            
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error updating cache entry: {str(e)}")

def get_cached_articles(limit: int = 100) -> List[PubMedArticle]:
    """Get the most recently accessed cached articles."""
    with get_db() as db:
        return db.query(PubMedArticle)\
            .join(CachedArticle)\
            .order_by(CachedArticle.last_accessed.desc())\
            .limit(limit)\
            .all()

def get_high_priority_cached_articles(limit: int = 100) -> List[PubMedArticle]:
    """Get cached articles with high priority scores."""
    with get_db() as db:
        return db.query(PubMedArticle)\
            .join(CachedArticle)\
            .order_by(CachedArticle.cache_priority.desc())\
            .limit(limit)\
            .all()

def get_recent_searches(limit: int = 10, user_id: Optional[str] = None) -> List[SearchHistory]:
    """Get recent search history."""
    with get_db() as db:
        query = db.query(SearchHistory)
        if user_id:
            query = query.filter(SearchHistory.user_id == user_id)
        return query.order_by(SearchHistory.timestamp.desc()).limit(limit).all()

def cleanup_old_cache_entries(max_age_days: int = 30) -> int:
    """Remove cache entries older than max_age_days."""
    with get_db() as db:
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            result = db.query(CachedArticle)\
                .filter(CachedArticle.last_accessed < cutoff_date)\
                .delete()
            db.commit()
            return result
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error cleaning up cache entries: {str(e)}")
            return 0

# VS Code specific
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json

# Windows specific
$RECYCLE.BIN/

# macOS specific
.AppleDouble
.LSOverride
Icon
._*
.Spotlight-V100
.Trashes

# Azure
.azure/
*.publishsettings

# Pinecone
.pinecone/
vector_cache/

# Security
*.p12
*.pfx

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/
.pytype/ 