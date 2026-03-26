from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
#from langchain.chains import RetrievalQA
#from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

#from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

load_dotenv()

class RAGSystem:
    def __init__(self, persist_directory="./rag_db"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
    def load_documents(self, data_path):
        """Load documents from various sources"""
        documents = []
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        text_loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(text_loader.load())
        
        return documents
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create vector store from document chunks"""
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # Note: Chroma 0.4.x+ automatically persists documents
        
    def load_vector_store(self):
        """Load existing vector store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def setup_retriever(self, k=5, search_type="similarity"):
        """Setup retriever"""
        if not self.vector_store:
            raise ValueError("Vector store not created or loaded")
            
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def create_qa_chain(self, streaming=True):
        """Create the complete RAG chain"""
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1
        )
        
        # Create custom prompt
        prompt_template = """Use the following context to answer the question. 
        If you don't know the answer, just say you don't know.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question, verbose=False):
        """Query the RAG system"""
        if not self.qa_chain:
            self.create_qa_chain()
        
        response = self.qa_chain.invoke({"query": question})
        
        if verbose:
            print(f"Question: {question}")
            print(f"Sources: {len(response['source_documents'])}")
        
        return response

# Initialize RAG system - works with ChromaDB, Pinecone, Weaviate, Qdrant, or Milvus
rag = RAGSystem(persist_directory="./my_rag_db")

# Create sample documents for testing (replace with your own data)
import os
if not os.path.exists("./data"):
    os.makedirs("./data")
    
    # Create sample text file
    with open("./data/sample.txt", "w", encoding="utf-8") as f:
        f.write("""RAG (Retrieval Augmented Generation) Systems

RAG systems combine the power of retrieval and generation to provide accurate, contextual responses.

Key Benefits:
1. Reduces hallucinations in AI responses
2. Enables AI to access up-to-date information
3. Provides source attribution for answers
4. Allows domain-specific knowledge integration

How RAG Works:
- Documents are processed and stored in a vector database
- User queries are converted to embeddings
- Similar document chunks are retrieved
- Retrieved context is used to generate accurate responses

Popular vector databases for RAG include ChromaDB, Pinecone, Weaviate, and FAISS.""")

# Option 1: Build new RAG index from scratch (for beginners)
try:
    documents = rag.load_documents("./data")
    if documents:
        print(f"Loaded {len(documents)} documents")
        chunks = rag.chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
        print(f"Created {len(chunks)} chunks")
        rag.create_vector_store(chunks)  # Creates embeddings and stores in vector database
        print("Vector store created successfully!")
        
        # Setup retriever
        rag.setup_retriever(k=5, search_type="similarity")
        
        # How to query your RAG system - simple example
        question = "What are the main benefits of using RAG systems?"
        response = rag.query(question, verbose=True)  # Returns answer + sources
        
        print(f"\nAnswer: {response['result']}")
        print(f"\nSources used: {len(response['source_documents'])}")
        
        # Show source content
        for i, doc in enumerate(response['source_documents']):
            print(f"\nSource {i+1}: {doc.page_content[:200]}...")
    else:
        print("No documents found in ./data directory")
        
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have documents in ./data directory and GOOGLE_API_KEY is set")