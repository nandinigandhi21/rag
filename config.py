import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """
    Global Application Configuration management using Pydantic Settings.
    
    This class handles environment variables, file paths, and model configurations 
    for the Docling RAG system. It supports prefix-based environment overrides 
    (e.g., DOCLING_PRO_OFFLINE_MODE=True).

    Attributes:
        BASE_DIR (Path): The root directory for application data and models.
        MODELS_CACHE (Path): Local cache directory for HuggingFace and Docling models.
        CHROMA_DB_DIR (Path): Persistent storage path for the ChromaDB vector store.
        OUTPUT_ROOT (Path): Default directory for processed document outputs.
        OFFLINE_MODE (bool): If True, forces libraries to operate without internet access.
        USE_OLLAMA (bool): Toggle between using the Ollama server and local inference.
    """
    # --- PATHS ---
    # Centralized path management to ensure consistency across engines.
    BASE_DIR: Path = Path(r"C:/Users/nandi/OneDrive/Desktop/amaya")
    MODELS_CACHE: Path = BASE_DIR / "models_cache"
    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"
    OUTPUT_ROOT: Path = BASE_DIR / "results"
    
    @property
    def SESSIONS_ROOT(self) -> Path:
        path = self.OUTPUT_ROOT / "research_sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # --- DOCLING MODELS ---
    DOCLING_ARTIFACTS: Path = MODELS_CACHE
    
    # --- EMBEDDING MODEL ---
    # Path for local bi-encoder models if not using Ollama.
    EMBEDDING_MODEL_PATH: Path = MODELS_CACHE / "bge-base-en-v1.5"
    
    # --- RERANKER MODEL ---
    # Path for the local Cross-Encoder model used for high-precision ranking.
    RERANKER_MODEL_PATH: Path = MODELS_CACHE / "ms-marco-MiniLM-L6-v2"
    
    # --- GENERATION MODEL ---
    # Defaulting to Qwen 3B for local LLM inference.
    LLM_MODEL_PATH: Path = MODELS_CACHE / "Qwen2.5-3B-Instruct"
    
    # --- CHUNKER SETTINGS ---
    # Default parameters for document segmentation.
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    DEFAULT_MERGE_PEERS: bool = True
    DEFAULT_TABLE_MODE: str = "accurate"
    
    # --- APP SETTINGS ---
    OFFLINE_MODE: bool = True
    USE_OLLAMA: bool = True  # Toggle this to True to use the Ollama API server.
    OLLAMA_BASE_URL: str = "http://localhost:11434" # Default Ollama API endpoint.
    OLLAMA_LLM_MODEL: str = "qwen2.5:3b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest" 
    
    DEBUG: bool = False
    LOG_FILE: Path = BASE_DIR / "app.log"
    
    # Pydantic configuration for environment variable handling.
    model_config = SettingsConfigDict(env_prefix="DOCLING_PRO_")

    def setup_logging(self):
        """
        Configures centralized logging and suppresses verbose library warnings.
        """
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Suppress chatty internal libraries
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("docling").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)

        logger = logging.getLogger()
        if logger.handlers:
            return 

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.LOG_FILE, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging system initialized (Silence Mode: Active)")

    def setup_environment(self):
        """
        Configures system environment variables to enforce offline behavior 
        and suppress library noise.
        """
        # Suppress library telemetry and logs
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["datasets_verbosity"] = "error"
        os.environ["TQDM_DISABLE"] = "1" # Hide progress bars from console
        
        if self.OFFLINE_MODE:
            # Force HuggingFace and Transformers into offline mode.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            # Point Docling to local artifact storage.
            os.environ["DOCLING_ARTIFACTS_PATH"] = str(self.DOCLING_ARTIFACTS)
            os.environ["HF_HOME"] = str(self.MODELS_CACHE)
            # Disable proxy settings to ensure local-only communication.
            os.environ["NO_PROXY"] = "*"

# Global Instance
config = AppConfig()
