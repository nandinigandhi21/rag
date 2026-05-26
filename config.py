import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    # --- PATHS ---
    BASE_DIR: Path = Path(r"C:\docling_dist-313")
    MODELS_CACHE: Path = BASE_DIR / "models_cache_311"
    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"
    OUTPUT_ROOT: Path = BASE_DIR / "newresults"
    
    # --- DOCLING MODELS ---
    DOCLING_ARTIFACTS: Path = MODELS_CACHE
    
    # --- EMBEDDING MODEL ---
    EMBEDDING_MODEL_PATH: Path = MODELS_CACHE / "bge-base-en-v1.5"
    
    # --- RERANKER MODEL ---
    RERANKER_MODEL_PATH: Path = MODELS_CACHE / "ms-macro-MiniLM-L6-v2"
    
    # --- GENERATION MODEL ---
    # Defaulting to Qwen 3B
    LLM_MODEL_PATH: Path = MODELS_CACHE / "Qwen2.5-3B-Instruct"
    
    # --- CHUNKER SETTINGS ---
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    DEFAULT_MERGE_PEERS: bool = True
    DEFAULT_TABLE_MODE: str = "accurate"
    
    # --- APP SETTINGS ---
    OFFLINE_MODE: bool = True
    DEBUG: bool = False
    LOG_FILE: Path = BASE_DIR / "app.log"
    
    model_config = SettingsConfigDict(env_prefix="DOCLING_PRO_")

    def setup_logging(self):
        """Configures centralized logging to both console and a file."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.LOG_FILE, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging system initialized (File: app.log)")

    def setup_environment(self):
        """Forces the environment variables for offline mode."""
        if self.OFFLINE_MODE:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DOCLING_ARTIFACTS_PATH"] = str(self.DOCLING_ARTIFACTS)
            os.environ["HF_HOME"] = str(self.MODELS_CACHE)
            os.environ["NO_PROXY"] = "*"

# Global Instance
config = AppConfig()
