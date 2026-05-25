import os
import torch
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from config import config
from schema import RAGResponse, SearchResult

logger = logging.getLogger("GenerationEngine")

class GenerationEngine:
    """
    Production-grade Generation Engine with Cited Answers and JIT Loading.
    """
    def __init__(self, model_path: Optional[Path] = None):
        config.setup_environment()
        self.model_dir = self._resolve_path(model_path or config.LLM_MODEL_PATH)
        self.model = None
        self.tokenizer = None
        logger.info(f"GenerationEngine initialized (Deferred loading for {self.model_dir})")

    def load(self):
        """Loads the LLM into memory only when needed."""
        if self.model is not None:
            return

        logger.info(f"JIT Loading LLM from: {self.model_dir}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map=device,
                torch_dtype=dtype,
                local_files_only=True
            )
            logger.info(f"LLM loaded on {device}")
        except Exception as e:
            logger.error(f"LLM Load Error: {e}")
            raise

    def unload(self):
        """Purges the LLM from memory to free up RAM/VRAM."""
        if self.model is None:
            return
            
        logger.info("Unloading LLM to free resources...")
        try:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("LLM purged from memory.")
        except Exception as e:
            logger.warning(f"Error during LLM unload: {e}")

    def _resolve_path(self, path: Path) -> str:
        if path.name == "snapshots":
            snaps = list(path.iterdir())
            if snaps: return str(snaps[0])
        return str(path)

    def generate_stream(self, query: str, context: str, max_new_tokens: int = 1024) -> Iterator[str]:
        """
        Generates a streaming response with citations, loading the model if necessary.
        """
        self.load() # Ensure model is in memory
        
        # --- CITED ANSWER PROMPT ---
        system_prompt = (
            "You are a professional research assistant. Your goal is to answer questions using the provided context. "
            "For every fact or answer you provide, you MUST cite the source using the format [Source X]. "
            "Always specify the section (breadcrumb) and page number if available in the source description. "
            "If the answer is not in the context, explicitly state that you cannot find the information. "
            "Be precise, professional, and highlight the validation details (e.g., 'According to the Methodology section on Page 4...')."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

class RAGOrchestrator:
    """
    Coordinates Retrieval and Generation with Metadata-rich Context.
    """
    def __init__(self, vector_engine, generation_engine):
        self.ve = vector_engine
        self.ge = generation_engine

    def query_stream(self, query: str, top_k: int = 5) -> Iterator[Dict[str, Any]]:
        # 1. Retrieve enriched chunks
        sources = self.ve.search(query, top_k=top_k)
        yield {"type": "sources", "content": sources}
        
        # 2. Construct Rich Context (Injecting metadata for the LLM)
        context_blocks = []
        for i, s in enumerate(sources):
            # Extract metadata from Chroma's flat dictionary
            meta = s.metadata
            breadcrumb = meta.get("breadcrumb", "General")
            page = meta.get("pages", "Unknown")
            source_file = meta.get("pdf_name", "Unknown Document")
            
            # Format the source header so the LLM knows where it is
            header = f"--- [Source {i+1}]: {source_file} | Section: {breadcrumb} | Page: {page} ---"
            context_blocks.append(f"{header}\n{s.text}")
        
        full_context = "\n\n".join(context_blocks)
        
        # 3. Stream Cited Answer
        yield {"type": "start_answer", "content": None}
        for chunk in self.ge.generate_stream(query, full_context):
            yield {"type": "answer_chunk", "content": chunk}
        yield {"type": "end_answer", "content": None}
