from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from dotenv import load_dotenv
from .rag_operations import process_documents, match_resumes
from .models import QueryRequest, MatchResponse
import logging

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Resume Screener API!"}

@app.post("/process_documents")
async def process_documents_endpoint(
    job_description: UploadFile = File(...),
    resumes: List[UploadFile] = File(...)
):
    try:
        if not job_description or not resumes:
            raise HTTPException(status_code=400, detail="Missing files")
        
        logger.info(f"Processing job description: {job_description.filename}")
        logger.info(f"Processing {len(resumes)} resumes")
        
        vector_store_id, _ = await process_documents(job_description, resumes)
        return {"message": "Documents processed successfully", "vector_store_id": vector_store_id}
    except Exception as e:
        logger.exception("Error processing documents")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/match_resumes", response_model=MatchResponse)
async def match_resumes_endpoint(request: QueryRequest):
    try:
        matches = await match_resumes(request.query, request.vector_store_id)
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)