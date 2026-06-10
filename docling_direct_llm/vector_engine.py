import os
import json
import logging
import gc
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import CrossEncoder

from config import config
from schema import SearchResult

logger = logging.getLogger("VectorEngine")

class VectorEngine:
    """
    A high-performance Vector Search Engine integrating ChromaDB and Ollama.
    
    This engine handles the indexing and retrieval of document chunks. It uses 
    Ollama for generating semantic embeddings and implements a multi-stage 
    retrieval process: initial vector search followed by local cross-encoder 
    re-ranking for maximum precision.

    Attributes:
        use_ollama (bool): Whether to use the Ollama API for embeddings.
        embed_model (str): The name of the embedding model to use in Ollama.
        db_path (str): Path to the persistent ChromaDB storage.
        reranker_path (str): Path to the local Cross-Encoder model.
        collection_name (str): The name of the ChromaDB collection.
    """
    def __init__(self, model_path: Optional[Path] = None, db_path: Optional[Path] = None, reranker_path: Optional[Path] = None, embed_model: Optional[str] = None, collection_name: str = "docling_documents_enriched"):
        """
        Initializes the VectorEngine with database and model configurations.

        Args:
            model_path (Optional[Path]): Deprecated or unused path for local models.
            db_path (Optional[Path]): Custom path for ChromaDB.
            reranker_path (Optional[Path]): Custom path for the Reranker model.
            embed_model (Optional[str]): Override for the Ollama embedding model.
            collection_name (str): The specific collection to operate on.
        """
        self.use_ollama = config.USE_OLLAMA
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.embed_model = embed_model or config.OLLAMA_EMBED_MODEL
        
        self.db_path = str(db_path or config.CHROMA_DB_DIR)
        self.reranker_path = str(reranker_path or config.RERANKER_MODEL_PATH)
        self.collection_name = collection_name
        
        self.reranker = None
        self.client = None
        self.collection = None
        
        logger.info(f"VectorEngine initialized (Mode: {'Ollama' if self.use_ollama else 'Local'}, Collection: {self.collection_name})")

    def _get_ollama_embedding(self, text: str) -> List[float]:
        """
        Fetches a semantic embedding for a given text string using the Ollama API.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The resulting embedding vector.

        Raises:
            requests.exceptions.RequestException: If the Ollama API call fails.
        """
        payload = {"model": self.embed_model, "prompt": text}
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["embedding"]

    def _get_ollama_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Processes a batch of texts to generate embeddings. 
        Note: Currently sequential as standard Ollama /api/embeddings is single-prompt.

        Args:
            texts (List[str]): A list of text strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._get_ollama_embedding(text))
        return embeddings

    def list_collections(self) -> List[str]:
        """
        Retrieves all available collection names from the ChromaDB instance.

        Returns:
            List[str]: A list of collection names.
        """
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.db_path)
        return [c.name for c in self.client.list_collections()]

    def load(self):
        """
        Performs JIT (Just-In-Time) loading of the vector database and re-ranker model.
        This minimizes startup time and memory footprint until search/index is required.
        """
        if self.client is not None and self.collection is not None:
            return

        logger.info(f"JIT Loading Vector Infrastructure for {self.collection_name}...")
        
        try:
            # We keep the Reranker local for high accuracy as Ollama doesn't natively do Cross-Encoding yet.
            # Cross-Encoders are significantly more accurate than bi-encoders for relevance scoring.
            if self.reranker is None:
                if os.path.exists(self.reranker_path):
                    self.reranker = CrossEncoder(self.reranker_path)
                else:
                    logger.warning(f"Reranker path not found, proceeding without re-ranking: {self.reranker_path}")

            if self.client is None:
                self.client = chromadb.PersistentClient(path=self.db_path)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Vector infrastructure loaded for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"VectorEngine Load Error: {e}")
            raise

    def unload(self):
        """
        Purges the local re-ranker model from memory and triggers garbage collection.
        """
        if self.reranker is None: return
        del self.reranker
        self.reranker = None
        gc.collect()
        logger.info("Vector reranker purged.")

    def add_processed_folder(self, folder_path: str, batch_size: int = 32) -> bool:
        """
        Indexes all chunks from a processed document folder into the vector database.

        Args:
            folder_path (str): The directory containing 'chunks.json'.
            batch_size (int): Number of chunks to process in a single database transaction.

        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        self.load()
        folder = Path(folder_path)
        chunks_file = folder / "chunks.json"
        
        if not chunks_file.exists():
            return False

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not chunks_data: return False

        logger.info(f"Indexing {len(chunks_data)} chunks via Ollama Embeddings...")

        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i : i + batch_size]
            documents, metadatas, ids = [], [], []
            
            for chunk in batch:
                documents.append(chunk["text"])
                ids.append(f"{folder.name}_{chunk['chunk_id']}")
                raw_meta = chunk["metadata"]
                # Flatten metadata for ChromaDB compatibility (no nested lists/dicts).
                flat_meta = {
                    "source": folder.name, "pdf_name": raw_meta["source_name"],
                    "doc_title": raw_meta.get("doc_title") or "N/A",
                    "chunk_id": chunk["chunk_id"], "pages": ", ".join(map(str, raw_meta["pages"])),
                    "breadcrumb": raw_meta["breadcrumb"], "is_table": raw_meta["is_table"]
                }
                metadatas.append(flat_meta)

            embeddings = self._get_ollama_embeddings_batch(documents)
            self.collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            
        return True

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Performs a two-stage hybrid search: 
        1. Vector search to retrieve candidates.
        2. Cross-Encoder re-ranking to refine top results.

        Args:
            query (str): The search query or question.
            top_k (int): The number of final results to return.

        Returns:
            List[SearchResult]: A ranked list of search results with relevance scores.
        """
        self.load()
        logger.info(f"Ollama-Retrieval: '{query}'")
        
        query_embedding = self._get_ollama_embedding(query)
        
        # Retrieve more candidates than top_k if a reranker is available to provide a better pool for refinement.
        candidate_count = top_k * 4 if self.reranker else top_k
        results = self.collection.query(query_embeddings=[query_embedding], n_results=candidate_count)
        
        if not results['ids'] or not results['ids'][0]: return []

        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": results['distances'][0][i] if 'distances' in results else 0
            })

        if self.reranker:
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        else:
            ranked = candidates

        final_hits = []
        for c in ranked[:top_k]:
            # Normalize score: higher is better.
            # If reranked, score is the logit/probability from the Cross-Encoder.
            # If not reranked, we convert cosine distance (0-2) to relevance (1.0 - distance).
            relevance = c.get("rerank_score", 1.0 - c["score"])
            
            final_hits.append(SearchResult(
                id=c["id"], text=c["text"], metadata=c["metadata"],
                score=round(float(relevance), 4)
            ))
        return final_hits

