from langchain_core.runnables import RunnablePassthrough
from tavily import TavilyClient
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from config import TAVILY_API_KEY, GOOGLE_API_KEY

class SearchChain:
    def __init__(self):
        """Initialize the search chain with Tavily API."""
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.search_wrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
        
        try:
            # Configure the Google Generative AI
            genai.configure(api_key=GOOGLE_API_KEY)
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=GOOGLE_API_KEY,
                temperature=0
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")
    
    def search(self, query: str, use_refinement: bool = False) -> dict:
        """Perform search using Tavily API.
        
        Args:
            query (str): The search query
            use_refinement (bool): Whether to use query refinement
            
        Returns:
            dict: Search results containing the list of results
        """
        try:
            # Use search() method instead of run()
            search_results = self.tavily_client.search(query)
            
            # Format the results
            if isinstance(search_results, dict):
                results = search_results.get('results', [])
                formatted_results = []
                for result in results:
                    formatted_results.append(f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}")
            else:
                formatted_results = [str(search_results)]
            
            return {
                "search_results": formatted_results,
                "search_performed": True
            }
            
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            return {
                "search_results": [f"Error performing search: {str(e)}"],
                "search_performed": False
            }