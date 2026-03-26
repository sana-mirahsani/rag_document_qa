from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # Can also use Pinecone, Weaviate, Qdrant, Milvus, or FAISS
import os

class RAGSystem:
    def __init__(self, persist_directory="./rag_db"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.vector_store = None
        self.retriever = None
        
    def load_documents(self, data_path):
        """Load documents from various sources"""
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        text_loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents.extend(text_loader.load())
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into optimal chunks"""
        # Create text splitter with semantic awareness
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",    # Paragraph breaks
                "\n",       # Line breaks
                ". ",       # Sentence ends
                ", ",       # Clause breaks
                " ",        # Word breaks
                ""          # Character breaks
            ]
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
        print(f"Created {len(chunks)} chunks")
        return chunks