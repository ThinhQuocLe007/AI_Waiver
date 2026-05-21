from pydantic import BaseModel
from langchain_core.documents import Document

class SearchResult(BaseModel):
    document: Document
    score: float
    bm25_score: float
    bm25_normalized: float
    vector_score: float
    source: str
    doc_type: str

    class Config:
        arbitrary_types_allowed = True # Allow Document objects

class DatabaseSummary(BaseModel):
    total_documents: int
    menu_documents: int
    restaurant_documents: int
    vector_db_path: str
    bm25_index_path: str
    score_threshold: float
