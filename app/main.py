import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from resume import Resume
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="JobMatch Composer", page_icon="💼")
    
    # Custom Styles for UI
    st.markdown("""
        <style>
            .title-text {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #4CAF50;
            }
            .small-text {
                font-size: 14px;
                color: gray;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title-text'>💼 JobMatch Composer</div>", unsafe_allow_html=True)
    st.markdown("<p class='small-text'>Generate professional emails tailored to job postings and your resume.</p>", unsafe_allow_html=True)

    # Layout: Split UI into two columns
    col1, col2 = st.columns(2)
    with col1:
        st.header("📋 Job Posting URL")
        url_input = st.text_input(
            "Enter the URL of the job posting:",
            placeholder="Paste job URL here",
            value="https://www.amazon.jobs/en/jobs/2848228/software-development-engineer"
        )
        submit_button = st.button("Generate Email ✉️")

    with col2:
        st.header("📜 About This Tool")
        st.info("This tool extracts job details from the URL and helps to compose a mail based on your personal resume.")
        st.success("Simple. Fast. Professional.")
    
    # Results Section
    st.divider()
    st.subheader("Generated Cold Email")

    if submit_button:
        try:
            with st.spinner("Extracting job details and generating your email..."):
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                portfolio.load_resume()
                jobs = llm.extract_jobs(data)
                
                for job in jobs:
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills)
                    email = llm.write_mail(job, links)
                    
                    # Display email content with syntax highlighting
                    st.code(email, language='markdown')
                    st.success("JobMatch Email Generated Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    resume = Resume()
    create_streamlit_app(chain, resume, clean_text)