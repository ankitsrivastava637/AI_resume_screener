from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    vector_store_id: str

class DocumentMetadata(BaseModel):
    file_name: str
    file_type: str
    experience: str = None
    location: str = None
    job_category: str = None

class Match(BaseModel):
    score: float
    highlights: str
    content: str

class MatchResponse(BaseModel):
    matches: List[Match]
    analysis: str

class ProcessedDocument(BaseModel):
    text: str
    metadata: DocumentMetadata