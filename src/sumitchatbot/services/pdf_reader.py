"""
PDF reader service for extracting text from PDF files.
This service uses PyMuPDF to read PDF files and extract text from them.
It supports reading both local files and files from URLs.

"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFReader:
    """
    PDF reader service for extracting text from PDF files.
    This service uses PyMuPDF to read PDF files and extract text from them.
    It supports reading both local files and files from URLs.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = None

    def read(self):
        """
        Reads a PDF file and extracts its text content as chunks.
        Returns:
            list: The extracted text chunks from the PDF file.
        """
        # verify file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        print(f"Reading PDF file: {self.file_path}")
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        self.chunks = text_splitter.split_documents(documents)

        print(f"Extracted {len(self.chunks)} chunks from the PDF file.")
        return self.chunks

if __name__ == "__main__":
    # path to the PDF file
    pdf_file_path = "C:/Program%20Files/Oracle/VirtualBox/doc/UserManual.pdf"

    try:
        reader = PDFReader(pdf_file_path)
        extracted_chunks = reader.read()
        print("PDF text extraction completed successfully.")
        print(f"Extracted {len(extracted_chunks)} chunks.")
    except Exception as e:
        print(f"An error occurred while reading the PDF file: {e}")