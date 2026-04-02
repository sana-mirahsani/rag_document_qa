# 🚀 RAG Document QA

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-blueviolet)
![Database](https://img.shields.io/badge/Database-Chroma%20%2B%20SQLite-blueviolet)
![LLM](https://img.shields.io/badge/LLM-google%2Fflan--t5--large-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📌 Description

This project implements a **Retrieval-Augmented Generation (RAG) system** that allows users to **query and interact with their own documents** using natural language.

The system combines:

* **Semantic search** over document chunks  
* **Large Language Models (LLMs)** for context-aware answers  

It enables **accurate, relevant, and explainable responses** from your documents.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/sana-mirahsani/rag_document_qa.git
cd rag_document_qa
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place your documents in the `/data` folder (e.g., `sample.txt`).
2. Run the main script:

```bash
python main.py
```

3. Enter your query when prompted. The system will retrieve relevant document chunks and generate an answer.

**Note:** Due to hardware limitations, large LLMs like GPT-4 are not used; the baseline model is `google/flan-t5-large`. You can use Google Colab for heavier models.

---

## 🧠 Model

* **Embedding model:** `HuggingFaceEmbeddings` (from `sentence-transformers`)
* **LLM for generation:** `google/flan-t5-large`
* **Pipeline:** LangChain `HuggingFacePipeline` + retrieval chain

**Recommendation:** Using a more powerful LLM (e.g., GPT-4 or `langchain-google-genai`) will significantly improve answer quality.

---

## 🗄️ Database

This project uses **Chroma** as the vector database to store **document embeddings**.

* Embeddings are created using `HuggingFaceEmbeddings`.
* Document chunks and metadata are stored locally in a **SQLite database** (`chroma.sqlite3`) inside the folder specified by `persist_directory` (default: `./my_rag_db`).
* Chroma enables **fast similarity search** for retrieval in the RAG system.

**Optional:** You can switch to other vector databases like **Pinecone, Qdrant, Weaviate, or Milvus** if desired.

**Example structure:**

```bash
my_rag_db/
└── chroma.sqlite3   # Local database storing embeddings & metadata
```

---

## 📂 Project Structure

```bash
project/
│
├── data/                         # Input documents (e.g., sample.txt)
│
├── my_rag_db/                    # Local vector database
│   └── chroma.sqlite3
│
├── main.py                       # RAG system main script
│
├── LICENSE                       
├── README.md                      
└── requirements.txt              # Python dependencies
```

---

## 📊 Results

* Baseline model `google/flan-t5-large` demonstrates the pipeline and LangChain integration.
* Responses are accurate for small-scale documents.
* For production-grade results, a more powerful LLM is recommended.

---

## 🛠️ Technologies Used

* Python 3.10
* LangChain
* Hugging Face Transformers
* sentence-transformers
* python-dotenv
* ChromaDB

---

## 👩‍💻 Author

**Sana Mirahsani**
🎓 Master’s Student in Machine Learning, University of Lille

🔗 LinkedIn: [https://www.linkedin.com/in/sana-mirahsani](https://www.linkedin.com/in/sana-mirahsani)
💻 GitHub: [https://github.com/sana-mirahsani](https://github.com/sana-mirahsani)