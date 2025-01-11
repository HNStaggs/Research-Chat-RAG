import os
# Set offline environment variables
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import streamlit as st
from database import DocumentDatabase
from utils import CodeGenerator

# Disable streamlit telemetry
st.set_option('browser.gatherUsageStats', False)

@st.cache_resource
def init_components():
    db = DocumentDatabase().create_or_load_db()
    generator = CodeGenerator()
    return db, generator

def main():
    st.title("Code Assistant (Offline Mode)")
    
    # Initialize components
    db, generator = init_components()
    
    # Sidebar for settings
    with st.sidebar:
        st.info("🔒 Running in offline mode")
        
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
            with st.spinner("Refreshing database..."):
                db = DocumentDatabase().refresh_database()
            st.success("Database refreshed successfully!")

    # Main interface
    st.header("Code Generation")
    user_query = st.text_area(
        "What code would you like to generate?",
        height=100,
        placeholder="Describe the code you need..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        generate_button = st.button("Generate Code", type="primary")
    with col2:
        include_explanation = st.checkbox("Include explanation", value=True)
    
    if generate_button and user_query:
        docs = db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
Language: {language}
Task: {user_query}

Reference Documentation:
{context}

Please generate code that follows best practices and includes comments.
"""
        
        with st.spinner("Generating code..."):
            generated_code = generator.generate_code(prompt, max_length)
            
            st.subheader("Generated Code")
            st.code(generated_code, language=language.lower())
            
            if st.button("Copy Code"):
                st.write(st.clipboard.copy(generated_code))
            
            if include_explanation:
                st.subheader("Code Explanation")
                with st.spinner("Generating explanation..."):
                    explanation = generator.generate_explanation(generated_code)
                    st.write(explanation)
        
        with st.expander("Reference Documentation Used"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Reference {i}:**")
                st.text(doc.page_content)

if __name__ == "__main__":
    main()

