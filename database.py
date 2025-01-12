from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
import chardet

class CustomTextLoader:
    """Custom text loader that handles different encodings"""
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            # First try UTF-8
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            try:
                # Detect the file encoding
                with open(self.file_path, 'rb') as file:
                    raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
                
                # Try reading with detected encoding
                with open(self.file_path, 'r', encoding=encoding) as file:
                    text = file.read()
            except Exception as e:
                raise RuntimeError(f"Could not load {self.file_path}: {str(e)}")
        
        from langchain_core.documents import Document
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

class DocumentDatabase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/sentence-transformer",
            model_kwargs={'device': 'cpu'}
        )
        self.index_file = "faiss_index"

    def load_pdf(self, file_path):
        try:
            loader = PDFPlumberLoader(file_path)
            return loader.load()
        except Exception as e:
            logging.warning(f"Failed to load PDF {file_path}: {str(e)}")
            return []

    def load_documents(self):
        documents = []
        errors = []
        
        # Check if docs directory exists
        if not os.path.exists('./docs'):
            os.makedirs('./docs')
            raise ValueError("No 'docs' directory found. Created empty 'docs' directory. Please add documentation files.")
        
        # Count documents
        doc_count = 0
        for root, _, files in os.walk('./docs'):
            for file in files:
                if file.lower().endswith(('.pdf', '.txt')):
                    doc_count += 1
        
        if doc_count == 0:
            raise ValueError("No PDF or TXT files found in 'docs' directory. Please add documentation files.")
        
        # Load documents
        for root, _, files in os.walk('./docs'):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith('.pdf'):
                        docs = self.load_pdf(file_path)
                        documents.extend(docs)
                    elif file.lower().endswith('.txt'):
                        loader = CustomTextLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                except Exception as e:
                    errors.append(f"Error loading {file}: {str(e)}")
        
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Errors loading some documents:\n{error_msg}")
        
        if not documents:
            raise ValueError("No content could be extracted from the documents. Please check file formats and contents.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = text_splitter.split_documents(documents)
        return texts

    def create_or_load_db(self):
        try:
            if not os.path.exists(f"{self.index_file}.faiss"):
                texts = self.load_documents()
                if not texts:
                    raise ValueError("No texts were extracted from documents.")
                db = FAISS.from_documents(texts, self.embeddings)
                # Save the index
                db.save_local(self.index_file)
            else:
                # Load the index
                db = FAISS.load_local(self.index_file, self.embeddings)
            return db
        except Exception as e:
            raise ValueError(f"Database creation failed: {str(e)}")

    def refresh_database(self):
        """Method to force refresh the database with new documents"""
        if os.path.exists(f"{self.index_file}.faiss"):
            os.remove(f"{self.index_file}.faiss")
        if os.path.exists(f"{self.index_file}.pkl"):
            os.remove(f"{self.index_file}.pkl")
        return self.create_or_load_db()