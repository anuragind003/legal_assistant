from typing import Dict, List, Any, Optional, TypedDict, Sequence, Literal
from enum import Enum
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import logging
from ..agents.legal_researcher import LegalResearcher
from ..chains.retrieval_chain import RetrievalChain
from ..config.config import GOOGLE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Action(str, Enum):
    SEARCH = "search"
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    FINALIZE = "finalize"

class WorkflowState(TypedDict):
    """State maintained between nodes."""
    messages: Sequence[BaseMessage]
    context: Dict[str, Any]
    current_step: str
    search_results: str
    research_output: str
    analysis_results: str
    final_answer: str
    references: List[str]
    confidence: float
    error_context: str

class LegalWorkflow:
    def __init__(self):
        """Initialize the workflow components."""
        self.legal_researcher = LegalResearcher()
        self.retrieval_chain = RetrievalChain()
        
        # Initialize Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.7
        )
        
        # Create and compile workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Define workflow nodes
        def search_node(state: WorkflowState) -> WorkflowState:
            try:
                query = state["messages"][-1].content if state["messages"] else ""
                search_results = self.legal_researcher.search_chain.search(query, use_refinement=True)
                state["search_results"] = str(search_results.get("search_results", []))
                state["current_step"] = Action.RETRIEVE
            except Exception as e:
                state["error_context"] = f"Error in search: {str(e)}"
            return state
        
        def research_node(state: WorkflowState) -> WorkflowState:
            try:
                query = state["messages"][-1].content if state["messages"] else ""
                research_output = self.legal_researcher.research(query)
                state["research_output"] = research_output["answer"]
                state["references"] = research_output["references"]
                state["current_step"] = Action.ANALYZE
            except Exception as e:
                state["error_context"] = f"Error in research: {str(e)}"
            return state
        
        def analysis_node(state: WorkflowState) -> WorkflowState:
            try:
                prompt = f"""Analyze the following legal information:
                Search Results: {state['search_results']}
                Research: {state['research_output']}
                
                Provide a clear analysis focusing on:
                1. Key legal principles
                2. Relevant precedents
                3. Practical implications
                """
                analysis = self.llm.invoke(prompt).content
                state["analysis_results"] = analysis
                state["current_step"] = Action.FINALIZE
            except Exception as e:
                state["error_context"] = f"Error in analysis: {str(e)}"
            return state
        
        def final_node(state: WorkflowState) -> WorkflowState:
            try:
                prompt = f"""Based on the research and analysis, provide a comprehensive answer:
                Research: {state['research_output']}
                Analysis: {state['analysis_results']}
                
                Format the response with:
                1. Clear explanation
                2. Legal basis
                3. Practical recommendations
                """
                final_answer = self.llm.invoke(prompt).content
                state["final_answer"] = final_answer
                state["confidence"] = 0.8 if not state["error_context"] else 0.4
                state["current_step"] = "complete"
            except Exception as e:
                state["error_context"] = f"Error in final answer: {str(e)}"
            return state
        
        # Add nodes
        workflow.add_node("search", search_node)
        workflow.add_node("research", research_node)
        workflow.add_node("analyze", analysis_node)
        workflow.add_node("finalize", final_node)
        
        # Add edges
        workflow.add_edge("search", "research")
        workflow.add_edge("research", "analyze")
        workflow.add_edge("analyze", "finalize")
        
        # Set entry and end points
        workflow.set_entry_point("search")
        workflow.set_finish_point("finalize")
        
        return workflow.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a legal query through the workflow.
        
        Args:
            query (str): The legal query to process
            
        Returns:
            Dict[str, Any]: Results containing answer, references, and confidence
        """
        try:
            # Initialize workflow state
            initial_state: WorkflowState = {
                "messages": [HumanMessage(content=query)],
                "context": {},
                "current_step": "search",
                "search_results": "",
                "research_output": "",
                "analysis_results": "",
                "final_answer": "",
                "references": [],
                "confidence": 0.0,
                "error_context": ""
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Format response
            return {
                "answer": final_state["final_answer"],
                "references": final_state["references"],
                "confidence": final_state["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "references": [],
                "confidence": 0.0
            }