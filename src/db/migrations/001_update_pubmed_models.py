from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings

def run_migration():
    # Create engine
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Drop existing tables if they exist
        session.execute(text("""
            DROP TABLE IF EXISTS article_embeddings CASCADE;
            DROP TABLE IF EXISTS search_history_articles CASCADE;
            DROP TABLE IF EXISTS search_history CASCADE;
            DROP TABLE IF EXISTS cached_articles CASCADE;
            DROP TABLE IF EXISTS metrics_analysis CASCADE;
            DROP TABLE IF EXISTS pubmed_articles CASCADE;
        """))
        
        # Create new tables
        session.execute(text("""
            CREATE TABLE pubmed_articles (
                id SERIAL PRIMARY KEY,
                pmid VARCHAR NOT NULL UNIQUE,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                publication_date TIMESTAMP,
                journal VARCHAR,
                keywords VARCHAR[],
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_cached BOOLEAN DEFAULT FALSE,
                hr_categories VARCHAR[],
                hr_relevance_scores JSONB,
                practical_implications TEXT,
                methodology_type VARCHAR,
                industry_context VARCHAR[]
            );
            
            CREATE TABLE search_history (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result_count INTEGER,
                user_id VARCHAR,
                article_ids INTEGER[],
                search_category VARCHAR
            );
            
            CREATE TABLE cached_articles (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES pubmed_articles(id) UNIQUE,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                cache_priority INTEGER DEFAULT 0,
                hr_relevance_score FLOAT DEFAULT 0.0
            );
            
            CREATE TABLE metrics_analysis (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES pubmed_articles(id),
                metric_type VARCHAR,
                metric_name VARCHAR,
                metric_description TEXT,
                measurement_method TEXT,
                benchmark_values JSONB,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE search_history_articles (
                search_history_id INTEGER REFERENCES search_history(id),
                article_id INTEGER REFERENCES pubmed_articles(id),
                rank INTEGER,
                relevance_score FLOAT,
                hr_relevance_score FLOAT,
                PRIMARY KEY (search_history_id, article_id)
            );
            
            -- Create indexes
            CREATE INDEX idx_pubmed_articles_pmid ON pubmed_articles(pmid);
            CREATE INDEX idx_pubmed_articles_publication_date ON pubmed_articles(publication_date);
            CREATE INDEX idx_pubmed_articles_hr_categories ON pubmed_articles USING GIN (hr_categories);
            CREATE INDEX idx_pubmed_articles_industry_context ON pubmed_articles USING GIN (industry_context);
            CREATE INDEX idx_search_history_timestamp ON search_history(timestamp);
            CREATE INDEX idx_search_history_category ON search_history(search_category);
            CREATE INDEX idx_cached_articles_last_accessed ON cached_articles(last_accessed);
            CREATE INDEX idx_cached_articles_cache_priority ON cached_articles(cache_priority);
            CREATE INDEX idx_cached_articles_hr_relevance ON cached_articles(hr_relevance_score);
            CREATE INDEX idx_metrics_analysis_metric_type ON metrics_analysis(metric_type);
            CREATE INDEX idx_metrics_analysis_article_id ON metrics_analysis(article_id);
        """))
        
        session.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        session.rollback()
        print(f"Migration failed: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    run_migration() 