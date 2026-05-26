import streamlit as st
import os
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Core Engine Imports
from config import config
config.setup_logging()

from schema import IngestionResult, SearchResult
from ingestion_engine import IngestionEngine
from vector_engine import VectorEngine
from generation_engine import GenerationEngine, RAGOrchestrator

# --- JOB MANAGEMENT (ASYNC ENGINE) ---
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_job(self, name: str) -> str:
        job_id = f"JOB_{datetime.now().strftime('%H%M%S')}_{name[:10]}"
        with self.lock:
            self.jobs[job_id] = {
                "name": name,
                "status": "Initializing",
                "progress": 0.0,
                "logs": [],
                "result": None,
                "start_time": datetime.now()
            }
        return job_id

    def update_job(self, job_id: str, status: str = None, progress: float = None, log: str = None, result: Any = None):
        with self.lock:
            if job_id in self.jobs:
                if status: self.jobs[job_id]["status"] = status
                if progress is not None: self.jobs[job_id]["progress"] = progress
                if log: self.jobs[job_id]["logs"].append(log)
                if result is not None: 
                    self.jobs[job_id]["result"] = result
                    self.jobs[job_id]["status"] = "Completed"

    def get_job(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)

@st.cache_resource
def get_job_manager():
    return JobManager()

# --- ENGINE PROVIDER ---
@st.cache_resource
def initialize_system():
    try:
        ve = VectorEngine()
        ge = GenerationEngine()
        return ve, ge, RAGOrchestrator(ve, ge)
    except Exception as e:
        st.error(f"System Initialization Failure: {e}")
        return None, None, None

# --- BACKGROUND WORKER ---
def background_worker(job_id: str, files: list, config_params: dict):
    jm = get_job_manager()
    target_path = config_params['target_path']
    
    # 1. Initialize Ingestion Engine
    ie = IngestionEngine(
        use_ocr=config_params['use_ocr'], 
        use_formula=config_params['use_formula'], 
        chunking_strategy=config_params['seg_strategy'], 
        table_mode=config_params['extract_mode'],
        max_tokens=config_params['chunk_val']
    )
    
    # 2. Sequential Process
    results_batch = []
    for idx, f in enumerate(files):
        fname = Path(f).name
        jm.update_job(job_id, status=f"Processing {fname}", progress=(idx / len(files)))
        
        try:
            # Extraction
            res = ie.process(f, output_root=target_path, skip_start=config_params['skip_head'], skip_end=config_params['skip_tail'])
            results_batch.append(res.output_path)
            jm.update_job(job_id, log=f"SUCCESS: Extracted {res.total_chunks} segments from {fname}")
            
            # Indexing (needs engines)
            if config_params['run_indexing']:
                jm.update_job(job_id, status=f"Indexing {fname}...")
                ve = VectorEngine() # Local instance for thread safety
                ve.add_processed_folder(res.output_path)
                jm.update_job(job_id, log=f"SUCCESS: Indexed metadata for {fname}")
                ve.unload()
                
        except Exception as e:
            jm.update_job(job_id, log=f"FAILURE: {fname} - {str(e)}")
            
    jm.update_job(job_id, status="Completed", progress=1.0, result=results_batch)

# --- ARCHITECTURAL STYLING ---
st.set_page_config(
    page_title="Zenith Vault | Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e1e1e; }
    
    /* Sidebar Sophistication */
    section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }
    .st-emotion-cache-16idsys p { font-weight: 600; color: #495057; }
    
    /* Card-based Citations */
    .citation-card {
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .citation-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .citation-text { font-size: 0.9rem; line-height: 1.5; color: #343a40; }
    .source-label { background: #e7f1ff; color: #0056b3; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    
    /* Button Refinement */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.01em;
        padding: 0.5rem 1.5rem;
    }
    
    /* Metrics & Log Styling */
    .log-entry { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 4px 8px; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- STATE PERSISTENCE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Zenith Vault")
    st.caption("Intelligence at the Edge")
    
    st.divider()
    nav_choice = st.radio("Management", ["Process Center", "Research Lab", "System Status"], label_visibility="collapsed")
    
    st.divider()
    st.subheader("Workflow Scope")
    run_extraction = st.checkbox("Data Extraction", value=True)
    run_indexing = st.checkbox("Vector Indexing", value=True)
    
    st.divider()
    st.subheader("Preferences")
    theme_accurate = st.toggle("Focus on Accuracy", value=True)
    
    if st.button("Reset Environment", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_logs = []
        st.cache_resource.clear()
        st.rerun()

# --- MAIN INTERFACE ---

# 1. PROCESS CENTER (Ingestion & Pipeline)
if nav_choice == "Process Center":
    st.header("Document Processing Center")
    st.markdown("Automate the transformation of unstructured documents into searchable knowledge assets.")
    
    jm = get_job_manager()
    
    # Robust Auto-Refresh Monitor
    if "last_job_id" in st.session_state:
        job = jm.get_job(st.session_state.last_job_id)
        if job:
            status_container = st.empty()
            with status_container.container(border=True):
                if job["status"] == "Completed":
                    st.success(f"✅ **Success:** {job['name']} processed and indexed successfully.")
                    st.balloons()
                    if st.button("Dismiss & Start New Job"):
                        del st.session_state.last_job_id
                        st.rerun()
                else:
                    st.info(f"⏳ **Extraction in Progress:** {job['name']}")
                    st.write(f"Current Phase: `{job['status']}`")
                    st.progress(job["progress"])
                    st.caption("This view updates automatically as background tasks complete.")
                    time.sleep(2)
                    st.rerun()

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            source_path = st.text_input("Source Path (PDF or Folder)", placeholder="e.g., C:\\Analysis\\Reports")
            target_path = st.text_input("Output Destination", value=str(config.OUTPUT_ROOT))
        with c2:
            s_range = st.expander("Page Constraints", expanded=True)
            with s_range:
                skip_head = st.number_input("Header Skip", 0, 500, 0)
                skip_tail = st.number_input("Footer Skip", 0, 500, 0)

    st.subheader("Configuration")
    t1, t2, t3 = st.tabs(["OCR Settings", "Segmentation", "Extraction"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        ocr_enabled = cc1.toggle("Optical Character Recognition", value=True)
        formula_enabled = cc2.toggle("Mathematical Notation", value=True)
    with t2:
        seg_strategy = st.selectbox("Methodology", ["Hybrid (Context Aware)", "Hierarchical (Structural)"])
        cc1, cc2 = st.columns(2)
        chunk_val = cc1.number_input("Max Token Capacity", 256, 1024, 512, step=64)
        overlap_val = cc2.number_input("Semantic Overlap", 0, 128, 64, step=8)
    with t3:
        extract_mode = st.selectbox("Extraction Fidelity", ["accurate", "fast"], index=0 if theme_accurate else 1)

    if st.button("Execute Async Workflow", type="primary", use_container_width=True):
        if not source_path or not os.path.exists(source_path):
            st.error("Operation Aborted: Valid source path required.")
        else:
            files = [str(f) for f in Path(source_path).glob("*.pdf")] if os.path.isdir(source_path) else ([source_path] if source_path.endswith(".pdf") else [])
            if not files:
                st.warning("Notification: No valid documents identified.")
            else:
                # submission
                job_id = jm.create_job(Path(files[0]).name if len(files)==1 else f"Batch ({len(files)} files)")
                config_params = {
                    "target_path": target_path, "use_ocr": ocr_enabled, "use_formula": formula_enabled,
                    "seg_strategy": seg_strategy.split()[0].lower(), "extract_mode": extract_mode,
                    "chunk_val": chunk_val, "skip_head": skip_head, "skip_tail": skip_tail, "run_indexing": run_indexing
                }
                thread = threading.Thread(target=background_worker, args=(job_id, files, config_params))
                thread.start()
                st.session_state.last_job_id = job_id
                st.rerun()

# 2. RESEARCH LAB (Chat Interface)
elif nav_choice == "Research Lab":
    st.header("Inquiry Lab")
    st.markdown("Engage with your data through natural language search and synthesis.")
    
    ve, ge, orch = initialize_system()
    if ve: ve.load()
    
    if not ve or ve.collection.count() == 0:
        st.warning("System Readiness: Knowledge base is currently empty.")
        st.stop()

    chat_container = st.container(height=500, border=False)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if q := st.chat_input("Input inquiry..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with chat_container:
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text = ""
                hits = []
                stream = orch.query_stream(q, top_k=5)
                for chunk in stream:
                    if chunk["type"] == "sources": hits = chunk["content"]
                    elif chunk["type"] == "answer_chunk":
                        full_text += chunk["content"]
                        placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                st.session_state.messages.append({"role": "assistant", "content": full_text})

# 3. SYSTEM STATUS
elif nav_choice == "System Status":
    st.header("Infrastructure & Health")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Resource Utilization")
        import torch
        st.metric("Computation Engine", "CUDA (GPU)" if torch.cuda.is_available() else "Standard (CPU)")
        ve, _, _ = initialize_system()
        if ve: st.metric("Indexed Knowledge Assets", f"{ve.collection.count()} Chunks")
    with c2:
        st.subheader("Environment Configuration")
        st.code(f"Base Directory: {config.BASE_DIR}\nModel Cache: {config.MODELS_CACHE}")

st.divider()
st.caption(f"Zenith Vault Framework | {datetime.now().strftime('%Y-%m-%d')}")
