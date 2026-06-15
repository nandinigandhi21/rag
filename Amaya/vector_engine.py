import os
import json
import logging
import gc
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from config import config
from schema import SearchResult

logger = logging.getLogger("VectorEngine")

class ModelManager:
    """
    A Singleton manager for heavy neural models.
    
    Prevents Out-Of-Memory (OOM) errors by ensuring that large models like 
    the Cross-Encoder are loaded exactly once and shared across all engine 
    instances.
    """
    _reranker = None

    @classmethod
    def get_reranker(cls, model_path: str) -> Optional[CrossEncoder]:
        if cls._reranker is None:
            if os.path.exists(model_path):
                logger.info(f"Loading Cross-Encoder re-ranker from {model_path}...")
                cls._reranker = CrossEncoder(model_path)
            else:
                logger.warning(f"Reranker model not found at {model_path}")
        return cls._reranker

    @classmethod
    def unload_reranker(cls):
        if cls._reranker:
            del cls._reranker
            cls._reranker = None
            gc.collect()
            logger.info("Cross-Encoder re-ranker purged from memory.")

class BaseRetrievalEngine(ABC):
    """
    Abstract Base Class defining the formal contract for all Retrieval Engines.
    
    Ensures that the RAG Orchestrator can interact with different search 
    methodologies (Vector DB, Direct JSON, Graph, etc.) using a uniform interface.
    """
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, hybrid: bool = False) -> List[SearchResult]:
        pass

    def _reciprocal_rank_fusion(self, vector_hits: List[Dict], bm25_hits: List[Dict], k: int = 60) -> List[Dict]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion (RRF).
        """
        scores = {}
        
        def process_list(hits):
            for rank, hit in enumerate(hits):
                doc_id = str(hit.get("id") or hit.get("chunk_id"))
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0.0, "hit": hit}
                scores[doc_id]["score"] += 1.0 / (rank + k)

        process_list(vector_hits)
        process_list(bm25_hits)

        fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [f["hit"] for f in fused]

class VectorEngine(BaseRetrievalEngine):
    """
    Enterprise Vector Search Engine using ChromaDB + Hybrid BM25.
    """
    def __init__(self, db_path: Optional[Path] = None, reranker_path: Optional[Path] = None, embed_model: Optional[str] = None, collection_name: str = "docling_documents_enriched"):
        self.use_ollama = config.USE_OLLAMA
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.embed_model = embed_model or config.OLLAMA_EMBED_MODEL
        
        self.db_path = str(db_path or config.CHROMA_DB_DIR)
        self.reranker_path = str(reranker_path or config.RERANKER_MODEL_PATH)
        self.collection_name = collection_name
        
        self.client = None
        self.collection = None
        self.bm25 = None
        self.bm25_chunks = []
        
        logger.info(f"VectorEngine initialized (Collection: {self.collection_name})")

    def _get_ollama_embedding(self, text: str) -> List[float]:
        payload = {"model": self.embed_model, "prompt": text}
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["embedding"]

    def _get_ollama_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._get_ollama_embedding(t) for t in texts]

    def list_collections(self) -> List[str]:
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.db_path)
        return [c.name for c in self.client.list_collections()]

    def _init_bm25(self):
        """Initializes or refreshes the BM25 index for keyword search."""
        if self.collection:
            data = self.collection.get(include=["documents", "metadatas"])
            if data["documents"]:
                tokenized_corpus = [doc.lower().split() for doc in data["documents"]]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.bm25_chunks = []
                for i in range(len(data["documents"])):
                    self.bm25_chunks.append({
                        "id": data["ids"][i],
                        "text": data["documents"][i],
                        "metadata": data["metadatas"][i]
                    })
                logger.info(f"BM25 index synchronized for {self.collection_name}")

    def load(self):
        if self.client is not None and self.collection is not None:
            return

        try:
            if self.client is None:
                self.client = chromadb.PersistentClient(path=self.db_path)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._init_bm25()
        except Exception as e:
            logger.error(f"VectorEngine Load Error: {e}")
            raise

    def count(self) -> int:
        self.load()
        return self.collection.count() if self.collection else 0

    def add_processed_folder(self, folder_path: str, batch_size: int = 32) -> bool:
        self.load()
        folder = Path(folder_path)
        chunks_file = folder / "chunks.json"
        
        if not chunks_file.exists(): return False

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not chunks_data: return False

        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i : i + batch_size]
            documents, metadatas, ids = [], [], []
            
            for chunk in batch:
                documents.append(chunk["text"])
                ids.append(f"{folder.name}_{chunk['chunk_id']}")
                raw_meta = chunk["metadata"]
                flat_meta = {
                    "source": folder.name, "pdf_name": raw_meta["source_name"],
                    "doc_title": raw_meta.get("doc_title") or "N/A",
                    "chunk_id": chunk["chunk_id"], "pages": ", ".join(map(str, raw_meta["pages"])),
                    "breadcrumb": raw_meta["breadcrumb"], "is_table": raw_meta["is_table"]
                }
                metadatas.append(flat_meta)

            embeddings = self._get_ollama_embeddings_batch(documents)
            self.collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        
        # Refresh BM25 after indexing new data
        self._init_bm25()
        return True

    def search(self, query: str, top_k: int = 5, hybrid: bool = False) -> List[SearchResult]:
        self.load()
        
        # 1. Vector Search
        query_embedding = self._get_ollama_embedding(query)
        # Professional candidate pool (4x top_k)
        candidate_pool_size = top_k * 4
        v_results = self.collection.query(query_embeddings=[query_embedding], n_results=candidate_pool_size)
        
        vector_candidates = []
        if v_results['ids'] and v_results['ids'][0]:
            for i in range(len(v_results['ids'][0])):
                vector_candidates.append({
                    "id": v_results['ids'][0][i],
                    "text": v_results['documents'][0][i],
                    "metadata": v_results['metadatas'][0][i],
                    "score": v_results['distances'][0][i]
                })

        # 2. Keyword Search
        bm25_candidates = []
        if hybrid and self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[-candidate_pool_size:][::-1]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    bm25_candidates.append(self.bm25_chunks[idx])

        # 3. Hybrid Fusion
        if hybrid:
            candidates = self._reciprocal_rank_fusion(vector_candidates, bm25_candidates)
        else:
            candidates = vector_candidates

        # 4. Rerank
        reranker = ModelManager.get_reranker(self.reranker_path)
        if reranker:
            pairs = [[query, c["text"]] for c in candidates[:candidate_pool_size]]
            rerank_scores = reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            ranked = sorted(candidates[:candidate_pool_size], key=lambda x: x.get("rerank_score", 0), reverse=True)
        else:
            ranked = candidates

        final_hits = []
        for c in ranked[:top_k]:
            # Normalize score: ChromaDB dist (0 to 2) -> Sim (1 to -1)
            # We clip to 0-1 for cleaner UI display.
            relevance = c.get("rerank_score", max(0.0, 1.0 - c.get("score", 0.5)))
            final_hits.append(SearchResult(
                id=str(c["id"]), text=c["text"], metadata=c["metadata"],
                score=round(float(relevance), 4)
            ))
        return final_hits

class DirectRetrievalEngine(BaseRetrievalEngine):
    """
    Optimized Direct Retrieval Engine with Hybrid Search and Smart Caching.
    """
    def __init__(self, folder_paths: List[str], reranker_path: Optional[Path] = None, embed_model: Optional[str] = None):
        self.folder_paths = [Path(p) for p in folder_paths]
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.embed_model = embed_model or config.OLLAMA_EMBED_MODEL
        self.reranker_path = str(reranker_path or config.RERANKER_MODEL_PATH)
        
        self.chunks_cache = []
        self.embeddings_cache = None
        self.bm25 = None
        self.collection_name = "direct_search"

    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        payload = {"model": self.embed_model, "prompt": text}
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=300)
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def load(self):
        """Loads chunks and manages the intelligent embedding cache."""
        if self.chunks_cache: return

        all_embeddings = []
        for folder in self.folder_paths:
            chunks_file = folder / "chunks.json"
            cache_file = folder / f"embeddings_{self.embed_model.replace(':', '_')}.npy"
            
            if not chunks_file.exists(): continue

            with open(chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for chunk in data:
                    raw_meta = chunk["metadata"]
                    chunk["flat_metadata"] = {
                        "source": folder.name, "pdf_name": raw_meta["source_name"],
                        "doc_title": raw_meta.get("doc_title") or "N/A",
                        "chunk_id": chunk["chunk_id"], "pages": ", ".join(map(str, raw_meta["pages"])),
                        "breadcrumb": raw_meta["breadcrumb"], "is_table": raw_meta["is_table"]
                    }
                self.chunks_cache.extend(data)

            # --- SMART CACHING LOGIC ---
            if cache_file.exists():
                folder_vecs = np.load(cache_file)
            else:
                folder_vecs = []
                for chunk in data:
                    folder_vecs.append(self._get_ollama_embedding(chunk["text"]))
                folder_vecs = np.array(folder_vecs)
                np.save(cache_file, folder_vecs)
            
            all_embeddings.append(folder_vecs)

        if all_embeddings:
            self.embeddings_cache = np.vstack(all_embeddings)
            
        # BM25 Initialization
        if self.chunks_cache:
            tokenized_corpus = [c["text"].lower().split() for c in self.chunks_cache]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def unload(self):
        pass

    def count(self) -> int:
        self.load()
        return len(self.chunks_cache)

    def search(self, query: str, top_k: int = 5, hybrid: bool = False) -> List[SearchResult]:
        self.load()
        if not self.chunks_cache or self.embeddings_cache is None: return []

        # 1. Vectorized Path
        query_vec = self._get_ollama_embedding(query)
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(self.embeddings_cache, axis=1)
        v_similarities = np.dot(self.embeddings_cache, query_vec) / (norm_q * norm_c)
        
        v_indices = np.argsort(v_similarities)[-top_k*4:][::-1]
        vector_candidates = []
        for idx in v_indices:
            vector_candidates.append({
                "id": str(self.chunks_cache[idx]["chunk_id"]),
                "text": self.chunks_cache[idx]["text"],
                "metadata": self.chunks_cache[idx]["flat_metadata"],
                "score": float(v_similarities[idx])
            })

        # 2. Keyword Path
        bm25_candidates = []
        if hybrid and self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[-top_k*4:][::-1]
            for idx in bm25_indices:
                if bm25_scores[idx] > 0:
                    bm25_candidates.append({
                        "id": str(self.chunks_cache[idx]["chunk_id"]),
                        "text": self.chunks_cache[idx]["text"],
                        "metadata": self.chunks_cache[idx]["flat_metadata"]
                    })

        # 3. Fusion
        if hybrid:
            candidates = self._reciprocal_rank_fusion(vector_candidates, bm25_candidates)
        else:
            candidates = vector_candidates

        # 4. Rerank
        reranker = ModelManager.get_reranker(self.reranker_path)
        if reranker:
            pairs = [[query, c["text"]] for c in candidates[:top_k*4]]
            rerank_scores = reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            ranked = sorted(candidates[:top_k*4], key=lambda x: x.get("rerank_score", 0), reverse=True)
        else:
            ranked = candidates

        return [SearchResult(
            id=c["id"], text=c["text"], metadata=c["metadata"],
            score=round(float(c.get("rerank_score", c.get("score", 0.5))), 4)
        ) for c in ranked[:top_k]]



