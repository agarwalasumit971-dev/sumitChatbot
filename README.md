# SumitChatBot

SumitChatBot is a Retrieval-Augmented Generation (RAG) chatbot that allows you to query your own PDF documents using a local LLM and a vector database (Chroma). It uses LangChain, HuggingFace, and Streamlit for a modern, interactive experience.

## Features
- Extracts text chunks from PDF files
- Stores chunks in a Chroma vector database using HuggingFace embeddings
- RAG chatbot interface built with Streamlit
- Retrieves relevant context from your documents and augments LLM responses
- Runs fully locally (no cloud required)

## Setup Instructions

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd sumitChatBot
```

### 2. Install dependencies
```sh
poetry install
poetry add streamlit langchain langchain-community chromadb sentence-transformers transformers accelerate
```

### 3. Prepare your PDF data
Extract chunks from your PDF using the provided scripts:

```sh
poetry run python src/sumitchatbot/services/chunks_display.py
```
This will save text chunks to a directory (e.g., `C:\Users\sumit\Downloads\chunksKuberenetes`).

### 4. Store chunks in Chroma vector DB
```sh
poetry run python src/sumitchatbot/services/vectorstore.py
```
This will create a Chroma DB at your chosen location (e.g., `C:\Users\sumit\Downloads\chroma_db`).

### 5. Run the RAG Chatbot
```sh
poetry run streamlit run src/sumitchatbot/app/rag_chatbot.py
```
Open the provided local URL in your browser to use the chatbot.

## Configuration
- **PDF path, chunk output directory, and Chroma DB path** can be set in the respective scripts.
- **LLM model**: By default, uses `google/flan-t5-base` (small, runs on CPU). You can change to any HuggingFace-supported local model in `rag_chatbot.py`.

## Troubleshooting
- If you see errors about missing packages, run `poetry add <package>` for any missing dependency.
- For large LLMs, ensure you have enough RAM/GPU and install `accelerate`.
- Models are downloaded once and cached locally by HuggingFace.

## Folder Structure
- `src/sumitchatbot/services/` — PDF chunking and vector DB scripts
- `src/sumitchatbot/app/` — Streamlit RAG chatbot app
- `src/sumitchatbot/persist/` — (optional) alternate location for vectorstore scripts

## Credits
- Built with [LangChain](https://github.com/langchain-ai/langchain), [HuggingFace Transformers](https://github.com/huggingface/transformers), [Streamlit](https://streamlit.io/), and [ChromaDB](https://www.trychroma.com/)