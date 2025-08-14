from .documentChunk import DocumentChunkService

class ChunksDisplay:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
        self.chunks = []

    def extract_and_store_chunks(self):
        """
        Uses DocumentChunkService to extract chunks from the PDF and stores each chunk as a separate text file in the output directory.
        """
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        chunk_service = DocumentChunkService(self.file_path)
        self.chunks = chunk_service.chunk_document()

        for idx, chunk in enumerate(self.chunks):
            chunk_path = os.path.join(self.output_dir, f"chunk_{idx+1}.txt")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                # If chunk is a Document object, get its content
                if hasattr(chunk, 'page_content'):
                    f.write(chunk.page_content)
                else:
                    f.write(str(chunk))

        return [os.path.join(self.output_dir, f"chunk_{idx+1}.txt") for idx in range(len(self.chunks))]

if __name__ == "__main__":
    # Example usage
    pdf_file_path = r"C:\\Users\\sumit\\Downloads\\kubernetespdf.pdf"  # Use raw string or double backslashes
    output_dir = r"C:\\Users\\sumit\\Downloads\\chunksKuberenetes"      # Use raw string or double backslashes

    display = ChunksDisplay(pdf_file_path, output_dir)
    chunk_files = display.extract_and_store_chunks()
    print(f"Chunks saved to: {chunk_files}")