import os
import arxiv
import streamlit as st
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import re
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="PDF & ArXiv Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PDFArXivResearchAssistant:
    def __init__(self, google_api_key: str, model: str = "gemini-1.5-flash"):
        """Initialize the research assistant with Google API key and model."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_api_key,
                temperature=0.1
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )
        except Exception as e:
            st.error(f"Error initializing Google AI: {str(e)}")
            self.llm = None
    
    def load_and_summarize_pdf(self, pdf_file) -> str:
        """Load PDF and generate a 5-sentence summary."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Split text into chunks if document is large
            texts = self.text_splitter.split_documents(documents)
            
            # Combine all text
            full_text = "\n".join([doc.page_content for doc in texts])
            
            # Create summarization prompt
            summary_prompt = f"""
            Please provide a concise summary of the following document in exactly 5 sentences. 
            Focus on the main research objectives, methodology, key findings, and conclusions.
            
            Document text:
            {full_text[:8000]}  # Limit text to avoid token limits
            
            Summary (exactly 5 sentences):
            """
            
            # Generate summary
            if self.llm:
                response = self.llm.invoke(summary_prompt)
                return response.content.strip()
            else:
                return "Error: LLM not initialized properly"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def search_arxiv_papers(self, research_topic: str, max_results: int = 5) -> List[Dict]:
        """Search ArXiv for papers related to the research topic."""
        try:
            # Create ArXiv client
            client = arxiv.Client()
            
            # Search for papers
            search = arxiv.Search(
                query=research_topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            progress_bar = st.progress(0)
            
            for i, paper in enumerate(client.results(search)):
                # Update progress
                progress_bar.progress((i + 1) / max_results)
                
                # Extract year from published date
                year = paper.published.year
                
                # Get authors names
                authors = [author.name for author in paper.authors]
                
                # Generate main contribution using LLM
                contribution_prompt = f"""
                Based on the following paper abstract, provide a concise 1-2 sentence summary of the main contribution:
                
                Title: {paper.title}
                Abstract: {paper.summary}
                
                Main contribution:
                """
                
                try:
                    if self.llm:
                        contribution_response = self.llm.invoke(contribution_prompt)
                        main_contribution = contribution_response.content.strip()
                    else:
                        main_contribution = "Unable to generate contribution summary (LLM error)"
                except Exception as e:
                    main_contribution = f"Error generating contribution: {str(e)}"
                
                papers.append({
                    'title': paper.title,
                    'authors': authors,
                    'year': year,
                    'main_contribution': main_contribution,
                    'arxiv_id': paper.entry_id.split('/')[-1],
                    'url': paper.entry_id,
                    'abstract': paper.summary,
                    'published_date': paper.published.strftime("%Y-%m-%d")
                })
            
            progress_bar.empty()
            return papers
            
        except Exception as e:
            st.error(f"Error searching ArXiv: {str(e)}")
            return []

def main():
    # Title and description
    st.title("📚 PDF & ArXiv Research Assistant")
    st.markdown("""
    Upload a PDF file to get a 5-sentence summary, then search ArXiv for related papers 
    with detailed information including authors, contributions, and abstracts.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    api_key = os.environ.get("GOOGLE_API_KEY")
    google_api_key = st.sidebar.text_input(
        "Google API Key",
        value=api_key,
        type="password",
        help="Enter your Google AI API key"
    )
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Choose the Gemini model to use"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload & Summary")
        
        # PDF file upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to summarize"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show file details
            st.info(f"File size: {uploaded_file.size / 1024:.2f} KB")
    
    with col2:
        st.header("🔍 ArXiv Search Settings")
        
        # Research topic input
        research_topic = st.text_input(
            "Research Topic",
            placeholder="e.g., machine learning transformers",
            help="Enter keywords for ArXiv search"
        )
        
        # Number of papers to extract
        num_papers = st.slider(
            "Number of Papers to Extract",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many papers to retrieve from ArXiv"
        )
    
    # Process button
    if st.button("🚀 Analyze PDF & Search ArXiv", type="primary"):
        if not google_api_key:
            st.error("Please provide a Google API key")
            return
        
        if uploaded_file is None:
            st.error("Please upload a PDF file")
            return
        
        if not research_topic.strip():
            st.error("Please enter a research topic")
            return
        
        # Initialize assistant
        with st.spinner("Initializing research assistant..."):
            assistant = PDFArXivResearchAssistant(google_api_key, model)
        
        # Create two columns for results
        st.markdown("---")
        
        # PDF Summary Section
        st.header("📋 PDF Summary")
        with st.spinner("Analyzing PDF content..."):
            pdf_summary = assistant.load_and_summarize_pdf(uploaded_file)
        
        st.markdown("### Summary")
        st.write(pdf_summary)
        
        # ArXiv Search Section
        st.header(f"📖 Top {num_papers} Related ArXiv Papers")
        st.markdown(f"**Search Topic:** {research_topic}")
        
        with st.spinner(f"Searching ArXiv for {num_papers} papers..."):
            papers = assistant.search_arxiv_papers(research_topic, num_papers)
        
        if papers:
            # Display papers in expandable sections
            for i, paper in enumerate(papers, 1):
                with st.expander(f"📄 Paper {i}: {paper['title'][:100]}...", expanded=i <= 3):
                    col_info, col_abstract = st.columns([1, 1])
                    
                    with col_info:
                        st.markdown("**📝 Paper Information**")
                        st.markdown(f"**Title:** {paper['title']}")
                        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                        st.markdown(f"**Year:** {paper['year']}")
                        st.markdown(f"**Published:** {paper['published_date']}")
                        st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                        st.markdown(f"**URL:** [View Paper]({paper['url']})")
                        
                        st.markdown("**🎯 Main Contribution:**")
                        st.info(paper['main_contribution'])
                    
                    with col_abstract:
                        st.markdown("**📖 Abstract**")
                        st.write(paper['abstract'])
            
            # Download results option
            st.markdown("---")
            
            # Create downloadable report
            report_content = f"# Research Analysis Report\n\n"
            report_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"**Research Topic:** {research_topic}\n"
            report_content += f"**PDF File:** {uploaded_file.name}\n\n"
            
            report_content += f"## PDF Summary\n{pdf_summary}\n\n"
            
            report_content += f"## Top {num_papers} Related ArXiv Papers\n\n"
            
            for i, paper in enumerate(papers, 1):
                report_content += f"### {i}. {paper['title']}\n"
                report_content += f"**Authors:** {', '.join(paper['authors'])}\n"
                report_content += f"**Year:** {paper['year']}\n"
                report_content += f"**ArXiv ID:** {paper['arxiv_id']}\n"
                report_content += f"**URL:** {paper['url']}\n"
                report_content += f"**Main Contribution:** {paper['main_contribution']}\n"
                report_content += f"**Abstract:** {paper['abstract']}\n\n"
            
            st.download_button(
                label="📥 Download Research Report",
                data=report_content,
                file_name=f"research_report_{research_topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
        else:
            st.warning("No papers found for the given research topic.")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This app uses:
- **LangChain** for PDF processing
- **Google Gemini** for summarization
- **ArXiv API** for paper search
- **Streamlit** for the web interface

**Required packages:**
```bash
pip install streamlit langchain 
pip install langchain-google-genai 
pip install pypdf arxiv
```
""")

if __name__ == "__main__":
    main()
