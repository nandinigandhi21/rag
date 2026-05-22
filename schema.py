from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    # Core Location
    pages: List[int] = Field(default_factory=list)
    total_pages: int
    
    # Hierarchy & Context
    headings: List[str] = Field(default_factory=list)
    breadcrumb: str  # e.g., "Intro > Methods > Results"
    
    # Content Properties
    is_table: bool = False
    is_formula: bool = False
    char_count: int
    
    # Source Info
    strategy: str
    source_name: str
    doc_title: Optional[str] = None
    doc_author: Optional[str] = None
    
    processed_at: datetime = Field(default_factory=datetime.now)

class Chunk(BaseModel):
    chunk_id: int
    text: str
    metadata: DocumentMetadata

class IngestionResult(BaseModel):
    job_id: str
    pdf_name: str
    output_path: str
    total_chunks: int
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    generated_at: datetime = Field(default_factory=datetime.now)
