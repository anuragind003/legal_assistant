from setuptools import setup, find_packages

setup(
    name="legal-rag-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langgraph>=0.0.15",
        "langchain-google-genai>=0.0.5",
        "langchain-chroma>=0.0.1",
        "langchain-community>=0.0.10",
        "langchain-text-splitters>=0.0.1",
        "langchain-core>=0.1.0",
        "tavily-python>=0.2.8",
        "chromadb>=0.4.18",
        "streamlit>=1.27.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "unstructured>=0.10.30",
        "pdf2image>=1.16.3",
        "pytesseract>=0.3.10",
        "langchain-tavily>=0.0.1"
    ],
)
