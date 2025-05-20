import os
import chainlit as cl
from anthropic import Anthropic
import json
import torch
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings
from src.database.init_db import init_db, create_vector_extension
from src.services.pubmed_service import PubMedService
from src.utils.memory import clear_memory
from database import DocumentDatabase

# Initialize services
anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
pubmed_service = PubMedService(
    email=settings.ENTREZ_EMAIL,
    api_key=settings.NCBI_API_KEY,
    embedding_model=settings.EMBEDDING_MODEL
)

# Initialize database
create_vector_extension(settings.DATABASE_URL)
engine = init_db(settings.DATABASE_URL)
Session = sessionmaker(bind=engine)

# Initialize document database for similarity search
doc_db = DocumentDatabase().create_or_load_db()

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
    session = Session()
    
    try:
        async with cl.Step(name="Searching PubMed..."):
            pubmed_results = await pubmed_service.fetch_pubmed_data(query)
            if pubmed_results:
                # Store in database
                stored_articles = await pubmed_service.store_pubmed_data(pubmed_results, session)
                await cl.Message(
                    content=f"📚 Found and stored {len(stored_articles)} relevant papers from PubMed"
                ).send()

        similar_docs = doc_db.similarity_search(query, k=3)
        
        # Create citations for each paper
        citations = [pubmed_service.create_citation(paper) for paper in pubmed_results]
        
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
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
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
    
    finally:
        session.close()

@cl.on_stop
def on_stop():
    """Clean up resources when stopping the app"""
    clear_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    pass  # Chainlit handles the app execution 