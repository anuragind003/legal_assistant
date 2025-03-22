from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from enum import Enum
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from ..chains.search_chain import SearchChain
from src.chains.retrieval_chain import RetrievalChain
from src.prompts.legal_prompts import SEARCH_DETERMINATION_PROMPT
from config import GOOGLE_API_KEY, MODEL_NAME
from google.api_core import exceptions as google_exceptions

class SearchDecision(str, Enum):
    NEEDS_SEARCH = "NEEDS_SEARCH"
    NO_SEARCH = "NO_SEARCH"

class LegalResearchOutput(TypedDict):
    answer: str
    references: List[str]
    confidence: float

class LegalResearcher:
    def __init__(self):
        """Initialize the legal researcher agent."""
        try:
            self.chat_model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",  # Updated model name
                temperature=0.7,
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",  # Updated model name
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
            
            self.search_chain = SearchChain()
            self.retrieval_chain = RetrievalChain()
            
            # Search determination chain
            self.search_determination_chain = (
                SEARCH_DETERMINATION_PROMPT 
                | self.llm 
                | StrOutputParser()
            )
        except google_exceptions.NotFound as e:
            print(f"Error initializing Gemini model: {e}")
            raise
    
    def determine_search_need(self, query: str) -> SearchDecision:
        """Determine if the query needs web search."""
        try:
            result = self.search_determination_chain.invoke({"query": query})
            if SearchDecision.NEEDS_SEARCH.value in result:
                return SearchDecision.NEEDS_SEARCH
            return SearchDecision.NO_SEARCH
        except google_exceptions.NotFound as e:
            print(f"Error during model inference: {e}")
            raise
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract reference citations from the text."""
        references = []
        lines = text.split('\n')
        
        # Look for common reference patterns
        for line in lines:
            line = line.strip()
            if line.startswith("Source:") or line.startswith("[") or "v." in line:
                references.append(line)
            elif line.startswith("See ") and len(line) > 10:
                references.append(line)
        
        return references
    
    def _evaluate_confidence(self, answer: str) -> float:
        """Evaluate the confidence of the answer based on language markers."""
        confidence_reducers = [
            "not clear", "uncertain", "might", "may", "possibly", 
            "cannot determine", "insufficient information"
        ]
        
        confidence_boosters = [
            "clearly", "established", "definitely", "specifically states",
            "explicitly", "according to", "demonstrates"
        ]
        
        base_confidence = 0.7  # Start with a reasonable base confidence
        
        # Adjust based on confidence markers
        for phrase in confidence_reducers:
            if phrase in answer.lower():
                base_confidence -= 0.1
                
        for phrase in confidence_boosters:
            if phrase in answer.lower():
                base_confidence += 0.05
        
        # Ensure confidence is between 0.1 and 0.95
        return max(0.1, min(0.95, base_confidence))
    
    def research(self, query: str, chat_history=None) -> LegalResearchOutput:
        """Conduct legal research based on the query."""
        try:
            # Validate and convert query
            if isinstance(query, dict):
                query = query.get("query", "")
            elif not isinstance(query, str):
                query = str(query)

            if not query.strip():
                raise ValueError("Empty query received")

            chat_history = chat_history or []
            
            search_decision = self.determine_search_need(query)
        
            search_performed = False
            if search_decision == SearchDecision.NEEDS_SEARCH:
                # Perform web search
                search_results = self.search_chain.search(query, use_refinement=True)
                search_performed = search_results.get("search_performed", False)
                
                # Format search results for context
                search_context = "\n\n".join(search_results["search_results"])
                
                # Get search-based answer
                search_answer = self.retrieval_chain.retrieve_and_answer(
                    query, 
                    chat_history=chat_history
                )
                
                # Format the answer with search-specific information
                answer = f"{search_answer}\n\nThis answer is based on web search results."
                references = self._extract_references(search_answer)
                confidence = self._evaluate_confidence(search_answer)
                
            else:
                # Use document retrieval only
                document_answer = self.retrieval_chain.retrieve_and_answer(
                    query, 
                    chat_history=chat_history
                )
                
                answer = document_answer
                references = self._extract_references(document_answer)
                confidence = self._evaluate_confidence(document_answer)
            
            return {
                "answer": answer,
                "references": references,
                "confidence": confidence,
                "search_performed": search_performed  # Add this field
            }
        
        except Exception as e:
            logger.error(f"Error in research: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "references": [],
                "confidence": 0.0,
                "search_performed": False
            }