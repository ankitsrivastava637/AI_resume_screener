import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"

def process_documents(job_description, resumes):
    files = {
        'job_description': ('job_description.pdf', job_description, 'application/pdf')
    }
    for i, resume in enumerate(resumes):
        files[f'resumes'] = (f'resume_{i}.pdf', resume, 'application/pdf')

    try:
        response = requests.post(f"{API_URL}/process_documents", files=files)
        response.raise_for_status()
        return response.json()['vector_store_id']
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

def match_resumes(query, vector_store_id):
    payload = {
        "query": query,
        "vector_store_id": vector_store_id
    }
    try:
        response = requests.post(f"{API_URL}/match_resumes", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error matching resumes: {str(e)}")
        return None

def main():
    st.title("AI Resume Screener")

    st.header("Upload Documents")
    job_description = st.file_uploader("Upload Job Description (PDF)", type="pdf")
    resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents"):
        if job_description is not None and resumes:
            with st.spinner("Processing documents..."):
                vector_store_id = process_documents(job_description, resumes)
                if vector_store_id:
                    st.success("Documents processed successfully!")
                    st.session_state['vector_store_id'] = vector_store_id
        else:
            st.warning("Please upload both a job description and at least one resume.")

    st.header("Match Resumes")
    query = st.text_input("Enter your query")

    if st.button("Match Resumes"):
        if 'vector_store_id' in st.session_state and query:
            with st.spinner("Matching resumes..."):
                results = match_resumes(query, st.session_state['vector_store_id'])
                if results:
                    st.subheader("Analysis")
                    st.write(results['analysis'])
                    
                    st.subheader("Top Matches")
                    for i, match in enumerate(results['matches'], 1):
                        st.markdown(f"**Match {i}**")
                        st.markdown(f"Score: {match['score']:.2f}")
                        st.markdown(f"Highlights: {match['highlights']}")
                        st.markdown(f"Content: {match['content']}")
                        st.markdown("---")
        else:
            st.warning("Please process documents and enter a query first.")

if __name__ == "__main__":
    main()