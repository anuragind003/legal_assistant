from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.data.vector_store import VectorStore
from src.prompts.legal_prompts import LEGAL_RESEARCH_PROMPT, DOCUMENT_RELEVANCE_PROMPT
from config import GOOGLE_API_KEY, MODEL_NAME, MAX_DOCUMENTS_TO_RETRIEVE

class RetrievalChain:
    def __init__(self):
        """Initialize the retrieval chain with vector store and LLM."""
        self.vector_store = VectorStore()
        
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        
        # Document relevance evaluator
        self.relevance_evaluator = (
            DOCUMENT_RELEVANCE_PROMPT 
            | self.llm 
            | StrOutputParser()
        )
        
        # Setup retrieval chain
        self.retrieval_chain = (
            RunnableParallel(
                {"query": RunnablePassthrough(), "context": self._retrieve_documents}
            )
            | LEGAL_RESEARCH_PROMPT
            | self.llm
            | StrOutputParser()
        )
    
    def _retrieve_documents(self, query):
        """Retrieve relevant documents from vector store and format them."""
        try:
            # Handle dictionary input
            if isinstance(query, dict):
                query = query.get("query", "")
            elif not isinstance(query, str):
                query = str(query)

            # Ensure query is not empty
            if not query.strip():
                raise ValueError("Empty query received")

            docs = self.vector_store.similarity_search_with_score(query, k=MAX_DOCUMENTS_TO_RETRIEVE)
            
            # Sort by relevance score (lower distance is better)
            docs.sort(key=lambda x: x[1])
            
            # Format documents
            formatted_docs = []
            for i, (doc, score) in enumerate(docs, 1):
                metadata = doc.metadata
                source = metadata.get('source', 'Unknown')
                formatted_docs.append(
                    f"Document {i}:\n"
                    f"Source: {source}\n"
                    f"Relevance Score: {1/(1+score):.2f}\n"
                    f"Content: {doc.page_content}\n"
                )
            
            return "\n\n".join(formatted_docs)
        
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return "Error retrieving documents. Please try again with a different query."
    
    def evaluate_document_relevance(self, query, document_content):
        """Evaluate the relevance of a document to the query."""
        return self.relevance_evaluator.invoke({
            "query": query,
            "document_content": document_content
        })
    
    def retrieve_and_answer(self, query, chat_history=None):
        """Retrieve documents and answer the query."""
        try:
            # Handle dictionary input
            if isinstance(query, dict):
                query = query.get("query", "")
            elif not isinstance(query, str):
                query = str(query)
                
            if chat_history is None:
                chat_history = []
                
            return self.retrieval_chain.invoke({
                "query": query,
                "user_query": query,
                "chat_history": chat_history
            })
            
        except Exception as e:
            logger.error(f"Error in retrieval chain: {str(e)}")
            return "Error processing your query. Please try again."