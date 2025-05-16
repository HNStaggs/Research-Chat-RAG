import os
import chainlit as cl
from anthropic import Anthropic
from Bio import Entrez, Medline
from typing import List, Dict, Optional
import json
from datetime import datetime
import torch
from database import DocumentDatabase
from utils import clear_memory, GPUManager
from dotenv import load_dotenv
from models import init_db, create_vector_extension, PubMedArticle, ArticleEmbedding
from sentence_transformers import SentenceTransformer
import re

# Load environment variables
load_dotenv()

# Configure Entrez for PubMed
Entrez.email = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize database
db = DocumentDatabase().create_or_load_db()

# Initialize PostgreSQL connection
database_url = os.getenv("DATABASE_URL", "postgresql://localhost/research_chat")
create_vector_extension(database_url)
Session = init_db(database_url)

# Initialize sentence transformer for embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def parse_pubmed_article(medline_record: str) -> Dict:
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

def create_citation(article_data: Dict) -> str:
    """Create a citation string from article data"""
    return f"{article_data['authors']} ({article_data['publication_date']}). {article_data['title']}. {article_data['journal']}"

async def store_pubmed_data(articles_data: List[Dict], session) -> List[PubMedArticle]:
    """Store PubMed articles and their embeddings in the database"""
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
        embedding = model.encode(text_for_embedding)
        
        # Create embedding record
        article_embedding = ArticleEmbedding(
            embedding=embedding.tolist()
        )
        
        # Link embedding to article
        article.embeddings = article_embedding
        
        session.add(article)
        stored_articles.append(article)
    
    session.commit()
    return stored_articles

async def fetch_pubmed_data(query: str, max_results: int = 5) -> List[Dict]:
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
        
        # Parse and store papers
        parsed_papers = [parse_pubmed_article(paper) for paper in papers]
        
        # Store in database
        session = Session()
        stored_articles = await store_pubmed_data(parsed_papers, session)
        session.close()
        
        return parsed_papers
    except Exception as e:
        print(f"Error fetching PubMed data: {e}")
        return []

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(
        content="👋 Welcome to the Research Assistant! I can help you with:",
        elements=[
            cl.Text(content="• Searching and analyzing research papers from PubMed\n• Answering questions about medical research\n• Providing summaries and insights with citations")
        ]
    ).send()

    cl.user_session.set("anthropic", anthropic)

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    query = message.content
    
    async with cl.Step(name="Searching PubMed..."):
        pubmed_results = await fetch_pubmed_data(query)
        if pubmed_results:
            await cl.Message(
                content=f"📚 Found {len(pubmed_results)} relevant papers from PubMed"
            ).send()

    similar_docs = db.similarity_search(query, k=3)
    
    # Create citations for each paper
    citations = [create_citation(paper) for paper in pubmed_results]
    
    context = f"""
    User Query: {query}
    
    Relevant PubMed Papers:
    {json.dumps(pubmed_results, indent=2)}
    
    Citations:
    {json.dumps(citations, indent=2)}
    
    Similar Documents from Database:
    {json.dumps([doc.page_content for doc in similar_docs], indent=2)}
    """
    
    async with cl.Step(name="Generating response..."):
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on the following context, please provide a comprehensive answer to the user's query. 
                    Include relevant citations using the provided citation format when referencing specific papers.
                    
                    Context: {context}"""
                }
            ]
        )
        
        await cl.Message(content=response.content[0].text).send()

@cl.on_stop
def on_stop():
    """Clean up resources when stopping the app"""
    clear_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    pass  # Chainlit handles the app execution 