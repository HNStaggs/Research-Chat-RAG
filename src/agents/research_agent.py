from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.config.settings import settings
from src.tools.pubmed_tool import PubMedTool
from src.tools.pinecone_tool import PineconeTool
from src.tools.bing_grounding_tool import BingGroundingTool
from src.db.db_utils import (
    save_article,
    get_article_by_pmid,
    save_search_history,
    update_cache_entry,
    get_cached_articles
)

class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        self.pubmed_tool = PubMedTool()
        self.pinecone_tool = PineconeTool()
        self.bing_grounding_tool = BingGroundingTool()
        
        # Initialize the research chain
        self.research_chain = self._create_research_chain()
    
    def _create_research_chain(self):
        # Create the research prompt template focused on HR and I-O psychology
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR and Industrial-Organizational Psychology research assistant that helps analyze workforce metrics and find relevant research.
            Your task is to:
            1. Understand the user's query about workforce metrics, HR practices, or organizational behavior
            2. Search for relevant papers in I-O psychology, HR management, and workforce studies
            3. Analyze the papers in the context of practical HR applications
            4. Generate citations and provide actionable insights for HR professionals
            
            Focus on:
            - Employee engagement and satisfaction
            - Performance metrics and productivity
            - Workforce diversity and inclusion
            - Talent management and development
            - Organizational culture and behavior
            - HR analytics and metrics
            - Leadership and management practices
            
            Always provide clear explanations and practical recommendations for HR professionals."""),
            ("human", "{query}")
        ])
        
        # Create the research chain
        chain = (
            {"query": RunnablePassthrough()}
            | research_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a research query and return results."""
        try:
            # Check cache first
            cached_articles = get_cached_articles(limit=10)
            if cached_articles:
                # Use Pinecone to find most relevant cached articles
                cached_results = await self.pinecone_tool.similarity_search(
                    query,
                    filter={"is_cached": True},
                    k=5
                )
                if cached_results:
                    return {
                        "status": "success",
                        "source": "cache",
                        "articles": [self._format_article(article) for article in cached_results],
                        "message": "Results retrieved from cache"
                    }
            
            # If no cached results, perform new search with HR/I-O focus
            search_results = await self.pubmed_tool.search(
                f"{query} AND (industrial psychology[MeSH] OR organizational behavior[MeSH] OR personnel management[MeSH])"
            )
            if not search_results:
                return {
                    "status": "error",
                    "message": "No results found"
                }
            
            # Add Bing grounding
            grounding_results = await self.bing_grounding_tool.run(query)
            
            # Save articles to database
            article_ids = []
            for article in search_results:
                db_article = save_article(article)
                if db_article:
                    article_ids.append(db_article.id)
                    # Update cache entry
                    update_cache_entry(db_article.id)
            
            # Save search history
            if article_ids:
                save_search_history(query, article_ids, user_id)
            
            # Get similar articles from Pinecone
            similar_articles = await self.pinecone_tool.similarity_search(
                query,
                k=5
            )
            
            return {
                "status": "success",
                "source": "new_search",
                "articles": [self._format_article(article) for article in search_results],
                "similar_articles": [self._format_article(article) for article in similar_articles],
                "grounding_results": grounding_results,
                "message": "New search completed successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }
    
    def _format_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Format article data for response with HR focus."""
        return {
            "pmid": article.get("pmid"),
            "title": article.get("title"),
            "abstract": article.get("abstract"),
            "authors": article.get("authors"),
            "publication_date": article.get("publication_date"),
            "journal": article.get("journal"),
            "keywords": article.get("keywords", []),
            "citation": self._generate_citation(article),
            "hr_relevance": self._assess_hr_relevance(article)
        }
    
    def _generate_citation(self, article: Dict[str, Any], format: str = "apa") -> str:
        """Generate citation in specified format."""
        if format.lower() == "apa":
            authors = article.get("authors", "").split(", ")
            if len(authors) > 1:
                author_str = f"{authors[0]} et al."
            else:
                author_str = authors[0]
            
            return f"{author_str} ({article.get('publication_date', '')[:4]}). {article.get('title')}. {article.get('journal')}."
        else:
            return f"{article.get('authors')} ({article.get('publication_date')}). {article.get('title')}. {article.get('journal')}."
    
    def _assess_hr_relevance(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the relevance of the article to HR practices."""
        keywords = article.get("keywords", [])
        title = article.get("title", "").lower()
        abstract = article.get("abstract", "").lower()
        
        # Define HR-related categories
        categories = {
            "employee_engagement": ["engagement", "satisfaction", "motivation", "morale"],
            "performance": ["performance", "productivity", "efficiency", "metrics"],
            "diversity": ["diversity", "inclusion", "equity", "discrimination"],
            "talent": ["talent", "recruitment", "retention", "development"],
            "culture": ["culture", "climate", "values", "norms"],
            "leadership": ["leadership", "management", "supervision"],
            "analytics": ["analytics", "metrics", "data", "measurement"]
        }
        
        relevance = {}
        for category, terms in categories.items():
            score = 0
            for term in terms:
                if term in title:
                    score += 2
                if term in abstract:
                    score += 1
                if any(term in k.lower() for k in keywords):
                    score += 1
            relevance[category] = score
        
        return relevance 