import os
import logging
import time
from typing import Tuple
import psutil
import warnings
from contextlib import nullcontext, contextmanager

# Set offline environment variables
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Import torch after setting warnings
import torch
import GPUtil
import streamlit as st
from database import DocumentDatabase
from utils import CodeGenerator, PerformanceMonitor, clear_memory, GPUManager

# Configure logging
logging.basicConfig(
    filename='app_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize performance monitor
monitor = PerformanceMonitor()

@contextmanager
def suppress_warnings():
    """Context manager to suppress warnings"""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    yield
    logging.getLogger("transformers").setLevel(logging.WARNING)

def configure_torch():
    """Configure PyTorch settings"""
    if torch.cuda.is_available():
        # Configure CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set default tensor type
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "gpu_available": True
        }
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        return {
            "device": "cpu",
            "device_name": "CPU",
            "cuda_version": None,
            "gpu_available": False
        }

def get_system_stats():
    """Get current system statistics including GPU"""
    stats = {}
    
    # GPU Stats
    if torch.cuda.is_available():
        try:
            gpu = GPUtil.getGPUs()[0]
            stats.update({
                "GPU Device": torch.cuda.get_device_name(0),
                "GPU Memory": f"{gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB",
                "GPU Util": f"{gpu.load*100:.1f}%",
                "CUDA Ver": torch.version.cuda
            })
        except Exception as e:
            logging.error(f"Error getting GPU stats: {e}")
            stats.update({"GPU Error": str(e)})
    
    # CPU Stats
    stats.update({
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "RAM Usage": f"{psutil.virtual_memory().percent}%"
    })
    
    return stats

@st.cache_resource(ttl=3600)
def init_components() -> Tuple[DocumentDatabase, CodeGenerator]:
    """Initialize and cache the main components"""
    try:
        monitor.start("init_components")
        
        # Configure torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
        
        # Initialize components
        db = DocumentDatabase().create_or_load_db()
        generator = CodeGenerator()
        
        monitor.end("init_components")
        return db, generator
        
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_similar_docs(_db, query: str, k: int = 3):
    """Cache similarity search results"""
    monitor.start("similarity_search")
    docs = _db.similarity_search(query, k=k)
    monitor.end("similarity_search")
    return docs

@st.cache_data(ttl=300)
def generate_code_cached(_generator, prompt: str, max_length: int, language: str):
    """Cache code generation results"""
    monitor.start("code_generation")
    
    # Create a structured prompt
    structured_prompt = f"""
Task: Create {language} code for the following requirement:
{prompt}

Requirements:
1. Use clean, readable {language} code
2. Include comprehensive comments
3. Follow {language} best practices
4. Implement proper error handling
5. Use meaningful variable/function names

Additional Context:
- Code should be well-structured and maintainable
- Include necessary imports/dependencies
- Add input validation where appropriate
- Consider edge cases

Please provide the implementation below:
"""
    
    result = _generator.generate_code(structured_prompt, max_length, language)
    monitor.end("code_generation")
    return result

def main():
    # Configure PyTorch
    torch_config = configure_torch()
    
    # Page configuration
    st.set_page_config(
        page_title="Code Assistant",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    monitor.start("app_startup")
    
    # Display device information
    if torch_config["gpu_available"]:
        st.sidebar.success(f"🚀 GPU Accelerated: {torch_config['device_name']}")
        st.sidebar.info(f"CUDA Version: {torch_config['cuda_version']}")
    else:
        st.sidebar.warning("⚠️ Running on CPU (No GPU Detected)")
    
    st.title("Code Assistant (GPU Accelerated)")
    
    # Initialize components with loading indicator
    with st.spinner("Loading AI models..."):
        db, generator = init_components()
    
    if db is None or generator is None:
        st.stop()
    
    # Sidebar for settings and monitoring
    with st.sidebar:
        # System monitoring
        st.header("System Monitor")
        if st.checkbox("Show System Stats", value=False):
            stats_placeholder = st.empty()
            stats = get_system_stats()
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            stats_placeholder.text(stats_text)
        
        # Settings
        st.header("Settings")
        language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "Java", "C++", "SQL"]
        )
        
        max_length = st.slider(
            "Maximum Code Length",
            min_value=100,
            max_value=1000,
            value=500,
            step=50
        )
        
        # Database management
        st.header("Database Management")
        if st.button("Refresh Documentation Database"):
            try:
                with st.spinner("Refreshing database..."):
                    monitor.start("refresh_database")
                    db = DocumentDatabase().refresh_database()
                    monitor.end("refresh_database")
                st.success("Database refreshed successfully!")
                if torch_config["gpu_available"]:
                    GPUManager.clear_memory()
            except Exception as e:
                logging.error(f"Error refreshing database: {str(e)}")
                st.error(f"Error refreshing database: {str(e)}")
    
    # Main interface
    st.header("Code Generation")
    
    # Two-column layout for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_query = st.text_area(
            "What code would you like to generate?",
            height=100,
            placeholder="Describe the code you need..."
        )
    
    with col2:
        st.markdown("### Options")
        generate_button = st.button("Generate Code", type="primary")
        include_explanation = st.checkbox("Include explanation", value=True)
        show_context = st.checkbox("Show reference docs", value=False)
    
    if generate_button and user_query:
        try:
            # Clear GPU memory before generation
            if torch_config["gpu_available"]:
                GPUManager.clear_memory()
            
            # Search documentation
            with st.spinner("Searching documentation..."):
                docs = get_similar_docs(db, user_query)
                context = "\n".join([doc.page_content for doc in docs])
            
            # Create prompt with context
            prompt = f"""
Based on the following documentation and requirements:

Documentation Reference:
{context}

User Request:
{user_query}

Technical Requirements:
1. Language: {language}
2. Include error handling
3. Add input validation
4. Use proper documentation
5. Follow coding standards

Please generate a complete, well-documented solution.
"""
            
            # Generate code
            with st.spinner("Generating code..."):
                generated_code = generate_code_cached(generator, prompt, max_length, language)
                
                st.subheader("Generated Code")
                st.code(generated_code, language=language.lower())
                
                # Add copy button
                if st.button("📋 Copy Code"):
                    st.write(st.clipboard.copy(generated_code))
                    st.success("Code copied to clipboard!")
            
            # Generate explanation if requested
            if include_explanation:
                with st.spinner("Generating explanation..."):
                    explanation_prompt = f"""
Explain this {language} code in detail:

{generated_code}

Focus on:
1. Overall purpose and functionality
2. Key components and their interactions
3. Important implementation details
4. Error handling and edge cases
5. Any assumptions or limitations
"""
                    explanation = generate_code_cached(
                        generator,
                        explanation_prompt,
                        300,
                        language
                    )
                    st.subheader("Code Explanation")
                    st.write(explanation)
            
            # Show reference documentation if requested
            if show_context:
                with st.expander("Reference Documentation Used"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Reference {i}:**")
                        st.text(doc.page_content)
            
            # Clear GPU memory after generation
            if torch_config["gpu_available"]:
                GPUManager.clear_memory()
            
        except Exception as e:
            logging.error(f"Error during code generation: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Code Assistant - GPU Accelerated Mode</p>
            <p style='color: gray; font-size: 0.8em;'>
                Using local models for offline code generation
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    monitor.end("app_startup")

if __name__ == "__main__":
    try:
        with suppress_warnings():
            main()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please check the logs for details.")