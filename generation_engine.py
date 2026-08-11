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

    def generate_stream(self, query: str, context: str, persona: str = "Standard", custom_instructions: str = "", temperature: float = 0.1, top_p: float = 0.9, max_new_tokens: int = 1024) -> Iterator[str]:
        """
        Generates a streaming response using dynamic parameters, personas, or custom instructions.

        Args:
            query (str): The user's question.
            context (str): The retrieved document chunks.
            persona (str): The AI persona (e.g., 'Strict Auditor', 'Custom').
            custom_instructions (str): Raw system instructions if persona is 'Custom'.
            temperature (float): Creativity control (0.0 to 1.0).
            top_p (float): Nucleus sampling control.
            max_new_tokens (int): Maximum length of the response.

        Yields:
            str: Token chunks from the LLM.
        """
        # --- PERSONA MAPPING ---
        personas = {
            "Standard": "You are a professional research assistant. Your goal is to answer questions using the provided context.",
            "Strict Auditor": "You are a senior auditor. Be extremely critical, precise, and literal. Flag any inconsistencies or missing data in the context.",
            "Creative Analyst": "You are a creative strategic analyst. Look for hidden patterns, connect distant ideas, and suggest innovative implications from the data.",
            "Executive Summarizer": "You are a chief of staff. Be brief, use bullet points, and focus only on the bottom-line business impact.",
            "Technical Writer": "You are a technical documentation expert. Use precise terminology, define acronyms, and focus on procedural accuracy."
        }
        
        if persona == "Custom" and custom_instructions:
            system_base = custom_instructions
        else:
            system_base = personas.get(persona, personas["Standard"])
        
        system_prompt = (
            f"{system_base} "
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
                "temperature": temperature,
                "top_p": top_p
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
                        # Extract metrics from Ollama's final payload
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 1) / 1e9 # Convert nanoseconds to seconds
                        total_duration = chunk.get("total_duration", 1) / 1e9
                        
                        tps = eval_count / eval_duration if eval_duration > 0 else 0
                        
                        # We yield a special structured string that the orchestrator can catch
                        yield f"__METRICS__|{tps}|{total_duration}"
                        break
        except Exception as e:
            logger.error(f"Ollama Generation Error: {e}")
            yield f"\n[ERROR: Failed to communicate with Ollama server: {e}]"

from vector_engine import BaseRetrievalEngine

class RAGOrchestrator:
    """
    The central coordinator for the Retrieval-Augmented Generation (RAG) workflow.
    
    This class manages the interaction between multiple retrieval engines 
    and the generation engine. It handles cross-collection search, global result 
    ranking, and context construction.

    Attributes:
        ves (List[BaseRetrievalEngine]): A list of initialized retrieval engine instances.
        ge (GenerationEngine): The generation engine instance.
    """
    def __init__(self, vector_engines: List[BaseRetrievalEngine], generation_engine: GenerationEngine):
        """
        Initializes the orchestrator with its required engine components.

        Args:
            vector_engines (List[BaseRetrievalEngine]): Knowledge sources for retrieval.
            generation_engine (GenerationEngine): The generator for answering.
        """
        self.ves = vector_engines 
        self.ge = generation_engine

    def query_stream(self, query: str, top_k: int = 5, hybrid: bool = False, persona: str = "Standard", custom_instructions: str = "", temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 1024) -> Iterator[Dict[str, Any]]:
        """
        Executes a full RAG cycle and yields results in a structured format.

        Step 1: Retrieve relevant chunks from all vector collections.
        Step 2: Perform global ranking based on similarity/rerank scores.
        Step 3: Construct a metadata-rich context block for the LLM.
        Step 4: Stream the grounded answer with specific persona and controls.

        Args:
            query (str): The user's question.
            top_k (int): Number of top sources to consider for the context.
            hybrid (bool): Whether to use Hybrid Search (Vector + BM25).
            persona (str): The AI persona for the response.
            custom_instructions (str): Manual system override if persona is 'Custom'.
            temperature (float): Generation temperature.
            top_p (float): Nucleus sampling parameter.
            max_tokens (int): Maximum response length.

        Yields:
            Dict[str, Any]: Structured packets indicating sources, status, or answer chunks.
        """
        # 1. Retrieve enriched chunks from all selected engines.
        all_sources = []
        for ve in self.ves:
            try:
                all_sources.extend(ve.search(query, top_k=top_k, hybrid=hybrid))
            except Exception as e:
                logger.error(f"Search failed for collection {ve.collection_name}: {e}")
        
        # 2. Global Ranking (Candidate Selection)
        ranked_sources = sorted(all_sources, key=lambda x: x.score, reverse=True)[:top_k]
        
        yield {"type": "sources", "content": ranked_sources}
        
        # 3. Construct Rich Context
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
        
        # 4. Stream Cited Answer with Controls
        yield {"type": "start_answer", "content": None}
        for chunk in self.ge.generate_stream(query, full_context, persona=persona, custom_instructions=custom_instructions, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens):
            if chunk.startswith("__METRICS__|"):
                parts = chunk.split("|")
                yield {"type": "metrics", "tps": float(parts[1]), "latency": float(parts[2])}
            else:
                yield {"type": "answer_chunk", "content": chunk}
        yield {"type": "end_answer", "content": None}

