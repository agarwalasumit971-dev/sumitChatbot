from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class PDFToChroma:
    def __init__(self, pdf_path, chroma_dir, chunk_size=1000, chunk_overlap=200, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.chroma_dir = chroma_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

    def extract_chunks(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        print(f"Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len)
        chunks = text_splitter.split_documents(documents)
        print(f"Extracted {len(chunks)} chunks from PDF.")
        return [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]

    def store_chunks_in_chroma(self):
        chunks = self.extract_chunks()
        print("Creating embeddings and storing in ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=self.chroma_dir)
        vectorstore.persist()
        print(f"Stored {len(chunks)} chunks in Chroma vector DB at {self.chroma_dir}")
        return vectorstore

if __name__ == "__main__":
    # Example usage
    pdf_path = r"C:\\Users\\sumit\\Downloads\\kubernetespdf.pdf"  # Update as needed
    chroma_dir = r"C:\\Users\\sumit\\Downloads\\chroma_db"         # Update as needed
    processor = PDFToChroma(pdf_path, chroma_dir)
    processor.store_chunks_in_chroma()
