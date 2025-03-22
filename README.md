# Legal RAG System

An agentic Retrieval-Augmented Generation (RAG) system for the legal domain, built with LangGraph, LangChain, Google's Gemini, Tavily Search API, and Chroma DB.

## Features

- **Intelligent Query Analysis**: Automatically determines whether a legal query needs web search or can be answered from existing documents
- **Legal Document Processing**: Processes and indexes various legal document formats (PDF, DOCX, TXT, etc.)
- **Web Search Integration**: Uses Tavily API to search for relevant legal information when needed
- **Vector Store Integration**: Uses Chroma DB to store and retrieve relevant document chunks
- **Agentic Workflow**: Implements a conditional workflow using LangGraph that can make decisions based on query type
- **Confidence Scoring**: Provides confidence scores for answers based on source quality and response certainty
- **Reference Citations**: Automatically extracts and provides legal citations and references
- **User-Friendly Interface**: Clean Streamlit UI for document management and legal research

## Architecture

```
legal-rag-system
├── src
│   ├── agents
│   │   ├── legal_researcher.py
│   │   └── query_agent.py
│   ├── chains
│   │   ├── document_chain.py
│   │   ├── search_chain.py
│   │   └── retrieval_chain.py
│   ├── data
│   │   ├── document_store.py
│   │   └── vector_store.py
│   ├── graphs
│   │   └── workflow.py
│   ├── prompts
│   │   ├── legal_prompts.py
│   │   └── search_prompts.py
│   ├── utils
│   │   ├── document_loader.py
│   │   └── text_splitter.py
│   └── main.py
├── tests
│   └── test_agents.py
├── config.py
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Google API key (for Gemini)
- Tavily API key (for web search)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/legal-rag-system.git
   cd legal-rag-system
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Running the application

Start the Streamlit application:

```bash
streamlit run src/main.py
```

Then open your browser and navigate to http://localhost:8501

## Usage

1. **Upload Documents**:

   - Upload legal documents through the Streamlit UI or specify a directory path
   - The system will process, split, and index these documents in the vector store

2. **Ask Legal Questions**:

   - Type your legal query in the chat input
   - The system will analyze your query and determine whether to use document retrieval, web search, or both
   - You'll receive a comprehensive answer with relevant citations and references

3. **View Vector Store Stats**:
   - Check the sidebar to see statistics about your document collection
   - Clear the vector store if needed

## Development

### Running Tests

Run the test suite:

```bash
python -m unittest discover tests
```

### Project Components

- **legal_researcher.py**: Main agent that orchestrates legal research using both documents and web search
- **workflow.py**: LangGraph implementation of the decision-making workflow
- **vector_store.py**: Interface with Chroma DB for document storage and retrieval
- **document_loader.py**: Utilities for loading and processing different document types
- **legal_prompts.py**: Specialized prompts for legal domain tasks
- **search_chain.py**: Chain for web search using Tavily API
- **retrieval_chain.py**: Chain for document retrieval and answer generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Gemini](https://ai.google.dev/)
- [Tavily Search API](https://tavily.com/)
- [Chroma DB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
