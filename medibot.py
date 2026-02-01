import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load .env
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


# =========================
# Vector Store
# =========================
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


# =========================
# Prompt
# =========================
def set_custom_prompt():
    template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly.
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# =========================
# Groq LLM (FREE + FAST)
# =========================
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",   # fast + free
        temperature=0.7,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )


# =========================
# Main App
# =========================
def main():
    st.title("🩺 MediBot - Medical Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
