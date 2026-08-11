from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    """
    Structured metadata associated with a document chunk.
    
    This model captures spatial, hierarchical, and contextual information 
    about where a chunk originated within the source document.
    """
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
    """
    A discrete segment of document content with associated metadata.
    """
    chunk_id: int
    text: str
    metadata: DocumentMetadata

class IngestionResult(BaseModel):
    """
    Summary of a completed document ingestion job.
    """
    job_id: str
    pdf_name: str
    output_path: str
    total_chunks: int
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchResult(BaseModel):
    """
    A single hit from the vector search or re-ranker.
    """
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    """
    The final generated answer from the RAG pipeline.
    """
    answer: str
    sources: List[SearchResult]
    generated_at: datetime = Field(default_factory=datetime.now)
