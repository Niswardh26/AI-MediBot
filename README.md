# 🩺 MediBot – AI Medical Assistant (RAG Chatbot)

An intelligent medical question-answering chatbot built using **LLMs + Retrieval Augmented Generation (RAG)**.

The system retrieves relevant medical knowledge from a custom document database using semantic search and generates accurate answers using a fast Large Language Model.

---

## 🚀 Features

✅ Ask medical questions in natural language  
✅ Context-aware answers from your knowledge base  
✅ Semantic search using vector embeddings  
✅ Ultra-fast LLM inference  
✅ Chat-style web UI  
✅ Fully local vector database (no cloud DB required)

---

## 🧠 Tech Stack

- Python
- LangChain
- FAISS (Vector DB)
- HuggingFace Embeddings
- Groq LLM API
- Streamlit UI

---

## ⚙️ Architecture (RAG Pipeline)

User Question  
→ Embeddings  
→ Vector Search (FAISS)  
→ Retrieve Relevant Docs  
→ LLM (Groq)  
→ Final Answer  

This approach is called **Retrieval Augmented Generation (RAG)**.

---

## 📦 Libraries Used

- LangChain – LLM orchestration
- FAISS – fast similarity search
- HuggingFace sentence-transformers – embeddings
- Groq – ultra-fast inference
- Streamlit – frontend UI

---

## 🖥️ Demo

Run locally:

```bash
streamlit run app.py
