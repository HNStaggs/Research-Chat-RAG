from Bio import Entrez, Medline
from typing import List, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from src.models.pubmed import PubMedArticle
from sentence_transformers import SentenceTransformer
from src.services.pinecone_service import PineconeService

class PubMedService:
    def __init__(self, email: str, api_key: str, pinecone_service: PineconeService, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'):
        """Initialize PubMed service"""
        self.email = email
        Entrez.email = email
        Entrez.api_key = api_key
        self.model = SentenceTransformer(embedding_model)
        self.pinecone_service = pinecone_service
    
    def parse_pubmed_article(self, medline_record: str) -> Dict:
        """Parse PubMed article data from Medline format"""
        record = Medline.parse(medline_record)
        article_data = next(record)
        
        return {
            'pmid': article_data.get('PMID', ''),
            'title': article_data.get('TI', ''),
            'abstract': article_data.get('AB', ''),
            'authors': '; '.join(article_data.get('AU', [])),
            'publication_date': article_data.get('DP', ''),
            'journal': article_data.get('JT', ''),
            'keywords': article_data.get('MH', []),
            'raw_data': str(article_data)
        }
    
    def create_citation(self, article_data: Dict) -> str:
        """Create a citation string from article data"""
        return f"{article_data['authors']} ({article_data['publication_date']}). {article_data['title']}. {article_data['journal']}"
    
    async def store_pubmed_data(self, articles_data: List[Dict], session: Session) -> List[PubMedArticle]:
        """Store PubMed articles in the database and their embeddings in Pinecone"""
        stored_articles = []
        
        for article_data in articles_data:
            # Create article record
            article = PubMedArticle(
                pmid=article_data['pmid'],
                title=article_data['title'],
                abstract=article_data['abstract'],
                authors=article_data['authors'],
                publication_date=datetime.strptime(article_data['publication_date'], '%Y %b') if article_data['publication_date'] else None,
                journal=article_data['journal'],
                keywords=article_data['keywords'],
                raw_data=article_data['raw_data']
            )
            
            # Generate embedding for title + abstract
            text_for_embedding = f"{article_data['title']} {article_data['abstract']}"
            embedding = self.model.encode(text_for_embedding)
            
            # Store in Pinecone
            metadata = {
                'title': article_data['title'],
                'abstract': article_data['abstract'],
                'authors': article_data['authors'],
                'journal': article_data['journal'],
                'publication_date': article_data['publication_date'],
                'keywords': article_data['keywords']
            }
            self.pinecone_service.store_embeddings(
                article_id=article_data['pmid'],
                embedding=embedding.tolist(),
                metadata=metadata
            )
            
            session.add(article)
            stored_articles.append(article)
        
        session.commit()
        return stored_articles
    
    async def fetch_pubmed_data(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fetch research papers from PubMed based on the query"""
        try:
            # Search PubMed
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            results = Entrez.read(handle)
            handle.close()

            # Fetch details for each paper
            papers = []
            for id in results["IdList"]:
                handle = Entrez.efetch(db="pubmed", id=id, rettype="medline", retmode="text")
                papers.append(handle.read())
                handle.close()
            
            # Parse papers
            parsed_papers = [self.parse_pubmed_article(paper) for paper in papers]
            return parsed_papers
            
        except Exception as e:
            print(f"Error fetching PubMed data: {e}")
            return []
    
    async def search_similar_articles(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar articles using vector similarity"""
        # Generate embedding for the query
        query_embedding = self.model.encode(query)
        
        # Search in Pinecone
        similar_articles = self.pinecone_service.search_similar(
            query_embedding=query_embedding.tolist(),
            top_k=top_k
        )
        
        return [match.metadata for match in similar_articles] 