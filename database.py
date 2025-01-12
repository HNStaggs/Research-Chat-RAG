from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
import chardet
import pickle
import time
import torch
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    filename='database_operations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomTextLoader:
    """Custom text loader that handles different encodings"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and process a text file with encoding detection"""
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
                encoding = detected['encoding'] or 'utf-8'
                
                # Try reading with detected encoding
                with open(self.file_path, 'r', encoding=encoding) as file:
                    text = file.read()
            except Exception as e:
                logging.error(f"Could not load {self.file_path}: {str(e)}")
                raise RuntimeError(f"Could not load {self.file_path}: {str(e)}")
        
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

class DocumentDatabase:
    """Manages document loading, processing, and vector storage with GPU support"""
    
    def __init__(self):
        self.cache_dir = "./cache"
        self.models_dir = "./models"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set up GPU/CPU device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # Initialize embeddings with GPU support
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/sentence-transformer",
            model_kwargs={
                'device': self.device,
                'cache_folder': os.path.join(self.models_dir, "cache")
            },
            cache_folder=os.path.join(self.models_dir, "cache"),
            encode_kwargs={
                'device': self.device,
                'batch_size': 32 if self.device == "cuda" else 8
            }
        )
        
        self.index_file = "faiss_index"
        self.doc_cache_file = os.path.join(self.cache_dir, "doc_cache.pkl")
        
        # Configure chunk settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        logging.info("DocumentDatabase initialized")

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process a PDF file"""
        try:
            loader = PDFPlumberLoader(file_path)
            return loader.load()
        except Exception as e:
            logging.error(f"Failed to load PDF {file_path}: {str(e)}")
            return []

    def get_cached_documents(self) -> Optional[List[Document]]:
        """Retrieve cached documents if available"""
        if os.path.exists(self.doc_cache_file):
            try:
                with open(self.doc_cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache: {str(e)}")
                if os.path.exists(self.doc_cache_file):
                    os.remove(self.doc_cache_file)
                return None
        return None

    def cache_documents(self, documents: List[Document]) -> None:
        """Cache processed documents"""
        try:
            with open(self.doc_cache_file, 'wb') as f:
                pickle.dump(documents, f)
            logging.info(f"Cached {len(documents)} documents")
        except Exception as e:
            logging.error(f"Failed to cache documents: {str(e)}")

    def load_documents(self) -> List[Document]:
        """Load and process all documents from the docs directory"""
        start_time = time.time()
        documents = []
        errors = []
        
        # Check cache first
        cached_docs = self.get_cached_documents()
        if cached_docs:
            logging.info("Using cached documents")
            return cached_docs
        
        # Check if docs directory exists
        if not os.path.exists('./docs'):
            os.makedirs('./docs')
            raise ValueError("No 'docs' directory found. Created empty 'docs' directory. Please add documentation files.")
        
        # Count documents
        doc_count = sum(1 for root, _, files in os.walk('./docs') 
                       for file in files if file.lower().endswith(('.pdf', '.txt')))
        
        if doc_count == 0:
            raise ValueError("No PDF or TXT files found in 'docs' directory. Please add documentation files.")
        
        # Load documents with progress tracking
        processed_count = 0
        for root, _, files in os.walk('./docs'):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith('.pdf'):
                        docs = self.load_pdf(file_path)
                    elif file.lower().endswith('.txt'):
                        loader = CustomTextLoader(file_path)
                        docs = loader.load()
                    else:
                        continue
                    
                    if docs:
                        documents.extend(docs)
                        processed_count += 1
                        logging.info(f"Processed {processed_count}/{doc_count}: {file}")
                    else:
                        errors.append(f"No content extracted from {file}")
                    
                except Exception as e:
                    error_msg = f"Error loading {file}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(error_msg)
        
        if not documents:
            raise ValueError("No content could be extracted from the documents. Please check file formats and contents.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Cache the processed documents
        self.cache_documents(texts)
        
        processing_time = time.time() - start_time
        logging.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        if errors:
            logging.warning("Processed with errors:\n" + "\n".join(errors))
        
        return texts

    def create_or_load_db(self):
        """Create or load the vector database"""
        try:
            if not os.path.exists(f"{self.index_file}.faiss"):
                start_time = time.time()
                texts = self.load_documents()
                if not texts:
                    raise ValueError("No texts were extracted from documents.")
                
                logging.info("Creating new vector database...")
                db = FAISS.from_documents(texts, self.embeddings)
                db.save_local(self.index_file)
                
                creation_time = time.time() - start_time
                logging.info(f"Vector database created in {creation_time:.2f} seconds")
            else:
                logging.info("Loading existing vector database...")
                db = FAISS.load_local(self.index_file, self.embeddings)
            
            return db
            
        except Exception as e:
            error_msg = f"Database creation failed: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def refresh_database(self):
        """Force refresh the database with new documents"""
        try:
            logging.info("Starting database refresh...")
            start_time = time.time()
            
            # Remove existing database files
            if os.path.exists(f"{self.index_file}.faiss"):
                os.remove(f"{self.index_file}.faiss")
            if os.path.exists(f"{self.index_file}.pkl"):
                os.remove(f"{self.index_file}.pkl")
            if os.path.exists(self.doc_cache_file):
                os.remove(self.doc_cache_file)
            
            # Create new database
            db = self.create_or_load_db()
            
            refresh_time = time.time() - start_time
            logging.info(f"Database refresh completed in {refresh_time:.2f} seconds")
            
            return db
            
        except Exception as e:
            error_msg = f"Database refresh failed: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            stats = {
                "total_documents": 0,
                "file_types": {"pdf": 0, "txt": 0},
                "total_chunks": 0,
                "database_size_mb": 0,
                "device": self.device
            }
            
            # Count documents
            for root, _, files in os.walk('./docs'):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        stats["file_types"]["pdf"] += 1
                    elif file.lower().endswith('.txt'):
                        stats["file_types"]["txt"] += 1
            
            stats["total_documents"] = sum(stats["file_types"].values())
            
            # Get database size
            if os.path.exists(f"{self.index_file}.faiss"):
                stats["database_size_mb"] = os.path.getsize(f"{self.index_file}.faiss") / (1024 * 1024)
            
            # Get memory usage if using GPU
            if self.device == "cuda":
                stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logging.info("Database cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")