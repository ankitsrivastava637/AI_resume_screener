import os
from typing import List
from fastapi import UploadFile
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from .utils import get_text_and_metadata, extract_highlights
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import pickle
import logging

logger = logging.getLogger(__name__)

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

async def process_documents(job_description: UploadFile, resumes: List[UploadFile]):
    jd_text, jd_metadata, _ = await get_text_and_metadata(job_description, "pdf")
    
    resume_text = ""
    resume_metadata = []
    for resume in resumes:
        text, metadata, _ = await get_text_and_metadata(resume, "pdf")
        resume_text += text + "\n\n"
        resume_metadata.append(metadata)
    
    all_text = jd_text + " " + resume_text
    all_metadata = [jd_metadata] + resume_metadata
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    text_chunks = text_splitter.split_text(all_text)

    embedded_chunks = embeddings.embed_documents(text_chunks)

    d = len(embedded_chunks[0])
    M = 32
    index = faiss.IndexHNSWFlat(d, M)
    
    index.add(np.array(embedded_chunks).astype('float32'))

    vector_store = LangchainFAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=LangchainFAISS.DocStore(dict(enumerate(text_chunks))),
        index_to_docstore_id=dict(enumerate(range(len(text_chunks))))
    )
    
    os.makedirs("faiss_index", exist_ok=True)
    vector_store_id = f"faiss_index_{len(os.listdir('faiss_index')) + 1}"
    faiss.write_index(index, f"faiss_index/{vector_store_id}")
    
    with open(f"faiss_index/{vector_store_id}_docstore.pkl", "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open(f"faiss_index/{vector_store_id}_metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)
    
    return vector_store_id, vector_store

async def match_resumes(query: str, vector_store_id: str):
    # Load the saved index
    index = faiss.read_index(f"faiss_index/{vector_store_id}")

    # Load the docstore and metadata
    with open(f"faiss_index/{vector_store_id}_docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(f"faiss_index/{vector_store_id}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Recreate the vector store
    vector_store = LangchainFAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=dict(enumerate(range(len(docstore.dict))))
    )

    # Create retrievers
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in vector_store.docstore.values()])
    bm25_retriever.k = 5

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    retrieved_docs = ensemble_retriever.get_relevant_documents(query)
    
    processed_matches = []
    for i, doc in enumerate(retrieved_docs[:5]):
        highlights = extract_highlights(doc.page_content, query)
        processed_matches.append({
            "score": 1 - (i * 0.1),
            "highlights": highlights,
            "content": doc.page_content[:500]
        })
    
    context = "\n\n".join([match['content'] for match in processed_matches])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    
    advanced_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert recruiter tasked with analyzing resumes and finding the best match for a given job description.

        Candidate Resumes:
        {context}

        Based on the candidate resumes provided, analyze and evaluate the candidates' suitability for the position. Consider the following:

        1. Skills match: Identify key skills and how well each candidate's skills align with typical job requirements.
        2. Experience relevance: Evaluate the candidates' past experiences and their relevance to typical job requirements.
        3. Education and qualifications: Assess the candidates' educational background.
        4. Potential for growth: Consider any indicators of the candidates' potential to grow into roles.
        5. Cultural fit: Look for any information that might indicate how well the candidates would fit into typical company cultures.

        For each candidate, provide:
        - A brief summary of their strengths and weaknesses.
        - A suitability score out of 10.
        - Suggestions for areas where the candidate might need further development.

        Finally, rank the candidates in order of suitability, explaining your reasoning.

        Question: {question}

        Your analysis:
        """
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=advanced_prompt)
    
    response = chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)
    
    return {
        "matches": processed_matches,
        "analysis": response['output_text']
    }