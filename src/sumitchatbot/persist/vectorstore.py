import os

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class VectorStoreLoader:
    def __init__(self, chunks_dir, persist_dir="chroma_db"):
        self.chunks_dir = chunks_dir
        self.persist_dir = persist_dir
        self.chunks = []

    def load_chunks(self):
        """
        Reads all text files from the chunks directory and loads their content into a list.
        """
        self.chunks = []
        for filename in os.listdir(self.chunks_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.chunks_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.chunks.append(f.read())
        return self.chunks


    def store_in_chroma(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Creates embeddings for the loaded chunks using HuggingFace and stores them in a Chroma vector database.
        Args:
            model_name (str): The HuggingFace model to use for embeddings.
        """
        if not self.chunks:
            self.load_chunks()
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma.from_texts(self.chunks, embedding=embeddings, persist_directory=self.persist_dir)
        vectorstore.persist()
        print(f"Stored {len(self.chunks)} chunks in Chroma vector DB at {self.persist_dir}")
        return vectorstore

if __name__ == "__main__":
    # Make sure to install: poetry add sentence-transformers langchain chromadb
    # You can change the model_name to any HuggingFace sentence-transformers model you prefer
    chunks_dir = r"C:\\Users\\sumit\\Downloads\\chunksKuberenetes"
    persist_dir = r"C:\\Users\\sumit\\Downloads\\chroma_db"
    print("Starting VectorStoreLoader...")
    print(f"Chunks directory: {chunks_dir}")
    print(f"Chroma persist directory: {persist_dir}")
    try:
        loader = VectorStoreLoader(chunks_dir, persist_dir)
        chunks = loader.load_chunks()
        print(f"Loaded {len(chunks)} chunks from directory.")
        vectorstore = loader.store_in_chroma(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Vector store creation and persistence complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
