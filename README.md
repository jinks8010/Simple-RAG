# 🧠 RAG-based PDF QA App

This is a simple **Retrieval-Augmented Generation (RAG)** application that allows you to upload a PDF, retrieve the most relevant content using semantic similarity, and generate answers using a lightweight LLM. It's built using Sentence Transformers, Qdrant vector store, and a Streamlit UI.

---

## 📘 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture that combines information retrieval and natural language generation. Instead of generating answers purely from a model's training data, RAG retrieves relevant documents from a knowledge base and feeds them into the language model to ground the answer in actual facts.

---

## 🧩 What is an Embedding?

An **embedding** is a numerical representation of data (like text) in a high-dimensional vector space. Similar meanings result in similar vectors. This is crucial for finding semantically relevant documents using distance-based search.

---

## 🗃️ What is a Vector Store / Vector Database?

A **vector database** stores these high-dimensional embeddings and allows for efficient similarity searches using methods like cosine similarity or Euclidean distance. It's the backbone of retrieval in RAG systems.

---

## 🛠️ What We Used

| Component           | Tool/Library                                    |
|---------------------|-------------------------------------------------|
| **Embedding Model** | `all-MiniLM-L6-v2` from `sentence-transformers` |
| **Vector Store**    | `Qdrant` (in-memory instance)                   |
| **PDF Parsing**     | `pdfplumber`                                    |
| **LLM**             | `HuggingFace Pipeline` (distil model)           |
| **UI**              | `Streamlit`                                     |
| **Language**        | Python                                          |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/jinks8010/Simple-RAG
cd your-repo-name
```
### 2. Create and activate a virtual environment
```bash
python -m venv rag_env
source rag_env/bin/activate   # On Windows use: rag_env\Scripts\activate
```
### 3. Install required dependencies
```bash
pip install -r requirements.txt
```
### 4. Start the Streamlit app
```bash
streamlit run app.py
```


