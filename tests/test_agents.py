"""
Unit tests for the Legal RAG System.

To run these tests:
1. From command line: cd "c:\d drive\Anurag\Crew ai\legal-rag-system" && python -m unittest tests/test_agents.py
2. Using pytest: cd "c:\d drive\Anurag\Crew ai\legal-rag-system" && pytest tests/test_agents.py

Requirements:
- Python 3.x
- unittest (built-in)
- pytest (optional)
- mock (from unittest.mock)
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.legal_researcher import LegalResearcher, SearchDecision
from src.graphs.workflow import LegalWorkflow
from src.chains.search_chain import SearchChain
from src.chains.retrieval_chain import RetrievalChain

# Test components:
# 1. TestLegalResearcher
#    - Tests search need determination
#    - Tests full research process
# 2. TestLegalWorkflow
#    - Tests workflow orchestration
#    - Verifies output format

class TestLegalResearcher(unittest.TestCase):
    
    @patch('src.agents.legal_researcher.ChatGoogleGenerativeAI')
    def test_determine_search_need(self, mock_llm):
        # Setup mock response
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "NEEDS_SEARCH"
        mock_llm.return_value = mock_instance
        
        # Create researcher with mocked components
        researcher = LegalResearcher()
        
        # Test function
        result = researcher.determine_search_need("What are the latest changes to copyright law?")
        
        # Verify result
        self.assertEqual(result, SearchDecision.NEEDS_SEARCH)
        
    @patch('src.agents.legal_researcher.SearchChain')
    @patch('src.agents.legal_researcher.RetrievalChain')
    @patch('src.agents.legal_researcher.ChatGoogleGenerativeAI')
    def test_research(self, mock_llm, mock_retrieval_chain, mock_search_chain):
        # Setup mock responses
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = "NEEDS_SEARCH"
        mock_llm.return_value = mock_llm_instance
        
        mock_search_instance = MagicMock()
        mock_search_instance.search.return_value = {
            "search_results": ["Mock search result 1", "Mock search result 2"]
        }
        mock_search_chain.return_value = mock_search_instance
        
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.retrieve_and_answer.return_value = "Legal answer with reference to Smith v. Jones (2022)"
        mock_retrieval_chain.return_value = mock_retrieval_instance
        
        # Create researcher with mocked components
        researcher = LegalResearcher()
        
        # Test function
        result = researcher.research("What constitutes fair use?")
        
        # Verify result
        self.assertIn("answer", result)
        self.assertIn("references", result)
        self.assertIn("confidence", result)
        self.assertTrue(len(result["references"]) > 0)

class TestLegalWorkflow(unittest.TestCase):
    
    @patch('src.graphs.workflow.LegalResearcher')
    def test_workflow_execution(self, mock_researcher):
        # Setup mock responses
        mock_researcher_instance = MagicMock()
        mock_researcher_instance.research.return_value = {
            "answer": "Test legal answer",
            "references": ["Case 1", "Case 2"],
            "confidence": 0.8
        }
        mock_researcher_instance.determine_search_need.return_value = SearchDecision.NEEDS_SEARCH
        mock_researcher.return_value = mock_researcher_instance
        
        # Create workflow with mocked components
        workflow = LegalWorkflow()
        workflow.legal_researcher = mock_researcher_instance
        
        # Test function
        result = workflow.process_query("What are the requirements for a valid contract?")
        
        # Verify result
        self.assertIn("answer", result)
        self.assertIn("references", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["answer"], "Test legal answer")

if __name__ == '__main__':
    unittest.main()