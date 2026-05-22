import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from config import config
from schema import SearchResult

logger = logging.getLogger("VectorEngine")

class VectorEngine:
    """
    Enhanced Production-grade Vector Engine.
    """
    def __init__(self, model_path: Optional[Path] = None, db_path: Optional[Path] = None):
        self.model_path = str(model_path or config.EMBEDDING_MODEL_PATH)
        self.db_path = str(db_path or config.CHROMA_DB_DIR)
        
        logger.info(f"Initializing VectorEngine with model: {self.model_path}")
        
        try:
            self.model = SentenceTransformer(self.model_path)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name="docling_documents_enriched",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"VectorEngine Init Error: {e}")
            raise

    def add_processed_folder(self, folder_path: str, batch_size: int = 64) -> bool:
        folder = Path(folder_path)
        chunks_file = folder / "chunks.json"
        
        if not chunks_file.exists():
            logger.warning(f"No chunks.json in {folder_path}")
            return False

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not chunks_data:
            return False

        # Duplicate Check
        test_id = f"{folder.name}_1"
        if self.collection.get(ids=[test_id])['ids']:
            logger.info(f"Folder '{folder.name}' already indexed.")
            return True

        logger.info(f"Indexing {len(chunks_data)} enriched chunks from {folder.name}...")

        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i : i + batch_size]
            
            documents, metadatas, ids = [], [], []
            for chunk in batch:
                documents.append(chunk["text"])
                ids.append(f"{folder.name}_{chunk['chunk_id']}")
                
                # Extract meta from the enriched Chunk object
                raw_meta = chunk["metadata"]
                
                # Chroma requires flat dictionaries (no lists)
                flat_meta = {
                    "source": folder.name,
                    "pdf_name": raw_meta["source_name"],
                    "doc_title": raw_meta.get("doc_title") or "N/A",
                    "chunk_id": chunk["chunk_id"],
                    "pages": ", ".join(map(str, raw_meta["pages"])),
                    "total_pages": raw_meta["total_pages"],
                    "breadcrumb": raw_meta["breadcrumb"],
                    "is_table": raw_meta["is_table"],
                    "is_formula": raw_meta["is_formula"],
                    "char_count": raw_meta["char_count"]
                }
                metadatas.append(flat_meta)

            embeddings = self.model.encode(documents, show_progress_bar=False).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        logger.info(f"Successfully indexed enriched folder '{folder.name}'.")
        return True

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        logger.info(f"Semantic Search: '{query}'")
        
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        hits = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                hits.append(SearchResult(
                    id=results['ids'][0][i],
                    score=round(1 - results['distances'][0][i], 4),
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                ))
            
        return hits
