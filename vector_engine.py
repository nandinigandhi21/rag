import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import config
from schema import SearchResult

logger = logging.getLogger("VectorEngine")

class VectorEngine:
    """
    Enhanced Production-grade Vector Engine with Re-ranking and JIT Loading.
    """
    def __init__(self, model_path: Optional[Path] = None, db_path: Optional[Path] = None, reranker_path: Optional[Path] = None):
        self.model_path = str(model_path or config.EMBEDDING_MODEL_PATH)
        self.db_path = str(db_path or config.CHROMA_DB_DIR)
        self.reranker_path = str(reranker_path or config.RERANKER_MODEL_PATH)
        
        self.model = None
        self.reranker = None
        self.client = None
        self.collection = None
        
        logger.info(f"VectorEngine initialized (Deferred loading for {self.model_path})")

    def load(self):
        """Loads embedding and reranker models into memory."""
        if self.model is not None:
            return

        logger.info(f"JIT Loading Vector Models...")
        try:
            self.model = SentenceTransformer(self.model_path)
            self.reranker = CrossEncoder(self.reranker_path)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name="docling_documents_enriched",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector models loaded successfully.")
        except Exception as e:
            logger.error(f"VectorEngine Load Error: {e}")
            raise

    def unload(self):
        """Purges vector models from memory."""
        if self.model is None:
            return
            
        logger.info("Unloading Vector models...")
        try:
            del self.model
            del self.reranker
            self.model = None
            self.reranker = None
            gc.collect()
            logger.info("Vector models purged.")
        except Exception as e:
            logger.warning(f"Error during Vector unload: {e}")

    def add_processed_folder(self, folder_path: str, batch_size: int = 64) -> bool:
        self.load() # Ensure models are in memory
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
        self.load() # Ensure models are in memory
        logger.info(f"Two-Stage Retrieval (Vector + Re-ranker): '{query}'")
        
        # --- STAGE 1: VECTOR RETRIEVAL ---
        # We retrieve 4x the requested top_k to give the re-ranker enough candidates
        candidate_count = top_k * 4
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []

        # --- STAGE 2: CROSS-ENCODER RE-RANKING ---
        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })

        # Prepare pairs for re-ranking: [Query, Document Text]
        pairs = [[query, c["text"]] for c in candidates]
        
        # Get relevance scores from the Cross-Encoder
        # The scores are logits; higher is more relevant
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach scores to candidates
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by the new re-ranker score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Take the top_k best results
        final_hits = []
        for c in ranked_candidates[:top_k]:
            final_hits.append(SearchResult(
                id=c["id"],
                score=round(c["rerank_score"], 4),
                text=c["text"],
                metadata=c["metadata"]
            ))
            
        logger.info(f"Re-ranking complete. Best score: {final_hits[0].score if final_hits else 'N/A'}")
        return final_hits
