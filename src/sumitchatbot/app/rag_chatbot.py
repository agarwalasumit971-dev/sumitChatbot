import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- CONFIG ---
CHROMA_DB_DIR = r"C:\\Users\\sumit\\Downloads\\chroma_db"  # Should match pdf_to_chroma.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # You can change to any local LLM supported by HuggingFace

# --- LOAD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        st.stop()

# --- LOAD LLM ---
@st.cache_resource(show_spinner=True)
def load_llm():
    try:
        llm_pipe = pipeline(
            "text-generation",
            model=LLM_MODEL,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            device_map="auto"
        )
        return HuggingFacePipeline(pipeline=llm_pipe)
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        st.stop()

# --- MAIN APP ---
st.set_page_config(page_title="SumitChatBot", layout="wide")
st.title("SumitChatBot: Document Q&A")
st.caption("Ask questions about your documents. (Admin: Add documents via setup script)")

vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("Generating response..."):
        try:
            result = rag_chain({"query": user_input})
            response = result["result"]
        except Exception as e:
            response = f"Error: {e}"
        st.session_state.chat_history.append((user_input, response))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

st.info("This chatbot uses RAG: it retrieves relevant context from your Chroma vector DB and augments a local LLM's response.")