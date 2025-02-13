![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Code Helper Chatbot
A GPU-accelerated code generation assistant that uses local LLMs and RAG (Retrieval-Augmented Generation) to generate code based on documentation references.

## 🌟 Features
- 🚀 GPU-accelerated code generation using SantaCoder/CodeLlama
- 📚 RAG system for documentation reference
- 💻 Support for multiple programming languages
- 🔄 Real-time system monitoring
- 📊 Performance tracking
- 🛠️ Customizable generation parameters
- 💾 Offline operation (runs on local GPU) - no internet required after setup, improves safety of proprietery codebase information
  

## System Requirements
- Python 3.9+
- CUDA-capable GPU (tested with RTX A2000 8GB)
- 16GB+ RAM recommended
- Windows/Linux/MacOS  # Initially built on Windows

## 🚀 Quick Start

### Installation using bash

#### Step 1: Clone the repository in a new project folder
1. `mkdir translate-demo-app`
2. `cd translate-demo-app`
3. `git clone https://github.com/hnstaggs/CodeHelperRAG.git`

#### Step 2: Create a virtual environment
4. `python -m venv venv`
5. `cd venv`
6. `Scripts\activate`  # On Windows
  
#### Step 3: Install required packages
7. `cd ..`  # Back to root directory
8. `pip install -r requirements.txt`

#### Step 4: Download Models
9. `python setup.py`

#### Step 5: Setup Directory
10. `mkdir -p models docs cache`

#### Step 6: Upload/save in-house documentation in docs folder

#### Step 7: Run the app
11. `streamlit run app.py`

#### Step 5: App should automatically launch in your browser
* If not, Locate URL provided by streamlit console output and paste into web browser.
* Example: http://localhost:8501
* Test app functionality on your machine

#### Step 6: Deploying to Streamlit Cloud to Share with Others
* Push your code to your GitHub repository
* Log in to Streamlit Cloud
* Give Streamlit access to your github
* Click "New app"
* Select your repository, branch, and main file path (app.py)
* Click "Deploy"
* Your app will be live in the cloud at https://[code-helper]streamlit.app

## 📦 Dependencies
* Windows  # Built on windows, might need to update code for Mac or Linux run
* Public facing github repo for project
* Up-to-date requirements.txt for streamlit deploy
* Streamlit
* Transformers
* PyTorch
* langchain
* Git

## 💻 Usage
* Select programming language
* Enter your code request
* Click "Generate Code"

### Features in Detail
* Multiple programming language support
* Context-aware generation
* Code explanation generation
* Syntax highlighting
* Copy to clipboard functionality
* RAG System
* Documentation reference
* PDF and TXT support
* Automatic text chunking
* Vector similarity search
* GPU acceleration
* Memory optimization
* Cache management
* Performance monitoring
* System statistics

## 🛠️  Model Configuration
* Adjust model parameters in utils.py
"""temperature = 0.4
top_p = 0.85
top_k = 50"""

## ⚠️ Limitations & Issues
* Processing speed depends on your hardware capabilities and the latest CUDA driver support
`python manage_gpu.py`  # Clear GPU memory
  
## 🙏 Acknowledgments/Credits
* Built with Streamlit Apache 2.0
* Uses Hugging Face Transformers Apache 2.0
* Uses LangChain for RAG 

### 📚 Additional Resources, Citations, and Attributes
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Documentation](https://huggingface.co/docs/hub/index)

# Made with ❤️ by Halee

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
