from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import logging
import pickle

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
        
        for root, _, files in os.walk('./docs'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith('.pdf'):
                    documents.extend(self.load_pdf(file_path))
                elif file.lower().endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = text_splitter.split_documents(documents)
        return texts

    def create_or_load_db(self):
        if not os.path.exists(f"{self.index_file}.faiss"):
            texts = self.load_documents()
            db = FAISS.from_documents(texts, self.embeddings)
            # Save the index
            db.save_local(self.index_file)
        else:
            # Load the index
            db = FAISS.load_local(self.index_file, self.embeddings)
        return db

    def refresh_database(self):
        """Method to force refresh the database with new documents"""
        if os.path.exists(f"{self.index_file}.faiss"):
            os.remove(f"{self.index_file}.faiss")
        if os.path.exists(f"{self.index_file}.pkl"):
            os.remove(f"{self.index_file}.pkl")
        return self.create_or_load_db()
