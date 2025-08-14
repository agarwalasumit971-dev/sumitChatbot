

from .pdf_reader import PDFReader

class DocumentChunkService:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chunks = None

    def chunk_document(self, chunk_size=1000, chunk_overlap=200):
        """
        Loads the PDF file using PDFReader and returns the extracted chunks.
        Args:
            chunk_size (int): Size of each chunk (default 1000)
            chunk_overlap (int): Overlap between chunks (default 200)
        Returns:
            list: The extracted text chunks from the PDF file.
        """
        reader = PDFReader(self.file_path)
        self.chunks = reader.read()
        return self.chunks
