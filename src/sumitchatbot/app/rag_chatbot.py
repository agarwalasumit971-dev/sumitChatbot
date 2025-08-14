import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- CONFIG ---
CHROMA_DB_DIR = r"C:\\Users\\sumit\\Downloads\\chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # You can change to any local LLM supported by HuggingFace

# --- LOAD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vectorstore

# --- LOAD LLM ---
@st.cache_resource(show_spinner=True)
def load_llm():
    llm_pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=llm_pipe)

# --- MAIN APP ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) Chatbot")

vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything about your documents...")

if user_input:
    with st.spinner("Generating response..."):
        result = rag_chain({"query": user_input})
        response = result["result"]
        st.session_state.chat_history.append((user_input, response))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

st.info("This chatbot uses RAG: it retrieves relevant context from your Chroma vector DB and augments a local LLM's response.")
