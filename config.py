import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    # --- PATHS ---
    BASE_DIR: Path = Path(r"C:\docling_dist-313")
    MODELS_CACHE: Path = BASE_DIR / "models_cache_311"
    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"
    OUTPUT_ROOT: Path = Path.home() / "parsing_output"
    
    # --- DOCLING MODELS ---
    DOCLING_ARTIFACTS: Path = MODELS_CACHE
    
    # --- EMBEDDING MODEL ---
    EMBEDDING_MODEL_PATH: Path = MODELS_CACHE / "bge-base-en-v1.5"
    
    # --- GENERATION MODEL ---
    # Defaulting to Qwen 3B
    LLM_MODEL_PATH: Path = Path(r"C:\users\nandi\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots")
    
    # --- CHUNKER SETTINGS ---
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    DEFAULT_MERGE_PEERS: bool = True
    DEFAULT_TABLE_MODE: str = "accurate"
    
    # --- APP SETTINGS ---
    OFFLINE_MODE: bool = True
    DEBUG: bool = False
    
    model_config = SettingsConfigDict(env_prefix="DOCLING_PRO_")

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
