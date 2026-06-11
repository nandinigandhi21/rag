import os
import requests
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

from config import config
from schema import RAGResponse, SearchResult

logger = logging.getLogger("GenerationEngine")

class GenerationEngine:
    """
    An LLM-based Generation Engine powered by Ollama.
    
    This engine is responsible for generating human-like responses based on 
    retrieved document context. It uses specialized system prompts to ensure 
    that answers are grounded in the provided sources and properly cited.

    Attributes:
        use_ollama (bool): Configuration toggle for Ollama mode.
        base_url (str): The endpoint URL for the Ollama server.
        model_name (str): The specific LLM model being used (e.g., 'qwen2.5:3b').
    """
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes the GenerationEngine with settings from the global config.

        Args:
            model_path (Optional[Path]): Unused in current Ollama-first implementation.
        """
        self.use_ollama = config.USE_OLLAMA
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model_name = config.OLLAMA_LLM_MODEL
        logger.info(f"GenerationEngine initialized (Mode: {'Ollama' if self.use_ollama else 'Local-Disabled'})")

    def load(self):
        """
        Verifies connectivity to the Ollama server and validates the environment.

        Raises:
            ValueError: If Ollama mode is disabled.
            ConnectionError: If the server is unreachable.
        """
        if not self.use_ollama:
            raise ValueError("Ollama mode is disabled in config.")
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama Server at {self.base_url}")
            else:
                logger.warning(f"Ollama Server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Could not reach Ollama server at {self.base_url}")

    def list_models(self) -> List[str]:
        """
        Retrieves a list of all model tags currently available on the Ollama server.

        Returns:
            List[str]: A list of model names.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []

    def unload(self):
        """
        Placeholder for cleanup logic. No-op for current API-based implementation.
        """
        pass

    def generate_stream(self, query: str, context: str, max_new_tokens: int = 1024) -> Iterator[str]:
        """
        Generates a streaming response using a professional 'Cited Answer' prompt.

        Args:
            query (str): The user's question.
            context (str): The retrieved document chunks to use as grounding data.
            max_new_tokens (int): Maximum length of the generated response.

        Yields:
            str: Token chunks from the LLM response.
        """
        # --- CITED ANSWER PROMPT ---
        # This prompt is critical for preventing hallucinations and ensuring auditability.
        system_prompt = (
            "You are a professional research assistant. Your goal is to answer questions using the provided context. "
            "For every fact or answer you provide, you MUST cite the source using the format [Source X]. "
            "Always specify the section (breadcrumb) and page number if available in the source description. "
            "If the answer is not in the context, explicitly state that you cannot find the information. "
            "Be precise, professional, and highlight the validation details."
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
            ],
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.1, # Low temperature for high factual consistency.
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                    if chunk.get("done"):
                        break
        except Exception as e:
            logger.error(f"Ollama Generation Error: {e}")
            yield f"\n[ERROR: Failed to communicate with Ollama server: {e}]"

class RAGOrchestrator:
    """
    The central coordinator for the Retrieval-Augmented Generation (RAG) workflow.
    
    This class manages the interaction between multiple vector engines (knowledge bases) 
    and the generation engine. It handles cross-collection search, global result 
    ranking, and context construction.

    Attributes:
        ves (List[VectorEngine]): A list of initialized vector engine instances.
        ge (GenerationEngine): The generation engine instance.
    """
    def __init__(self, vector_engines: List[Any], generation_engine: GenerationEngine):
        """
        Initializes the orchestrator with its required engine components.

        Args:
            vector_engines (List[VectorEngine]): Knowledge sources for retrieval.
            generation_engine (GenerationEngine): The generator for answering.
        """
        self.ves = vector_engines 
        self.ge = generation_engine

    def query_stream(self, query: str, top_k: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Executes a full RAG cycle and yields results in a structured format.

        Step 1: Retrieve relevant chunks from all vector collections.
        Step 2: Perform global ranking based on similarity/rerank scores.
        Step 3: Construct a metadata-rich context block for the LLM.
        Step 4: Stream the grounded answer.

        Args:
            query (str): The user's question.
            top_k (int): Number of top sources to consider for the context.

        Yields:
            Dict[str, Any]: Structured packets indicating sources, status, or answer chunks.
        """
        # 1. Retrieve enriched chunks from all selected engines.
        all_sources = []
        for ve in self.ves:
            try:
                all_sources.extend(ve.search(query, top_k=top_k))
            except Exception as e:
                logger.error(f"Search failed for collection {ve.collection_name}: {e}")
        
        # 2. Global Ranking (Candidate Selection)
        # We sort all candidates from all collections by their scores to find the globally best matches.
        ranked_sources = sorted(all_sources, key=lambda x: x.score, reverse=True)[:top_k]
        
        yield {"type": "sources", "content": ranked_sources}
        
        # 3. Construct Rich Context
        # We build a structured context block where each source is clearly demarcated with its metadata.
        context_blocks = []
        for i, s in enumerate(ranked_sources):
            meta = s.metadata
            breadcrumb = meta.get("breadcrumb", "General")
            page = meta.get("pages", "Unknown")
            source_file = meta.get("pdf_name", "Unknown Document")
            col_source = meta.get("source", "Unknown Collection")
            
            header = f"--- [Source {i+1}]: {source_file} (Vault: {col_source}) | Section: {breadcrumb} | Page: {page} ---"
            context_blocks.append(f"{header}\n{s.text}")
        
        full_context = "\n\n".join(context_blocks)
        
        # 4. Stream Cited Answer
        yield {"type": "start_answer", "content": None}
        for chunk in self.ge.generate_stream(query, full_context):
            yield {"type": "answer_chunk", "content": chunk}
        yield {"type": "end_answer", "content": None}

