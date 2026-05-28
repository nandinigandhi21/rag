import streamlit as st
import os
import threading
import queue
import time
import fitz  # PyMuPDF for PDF preview
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

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
def initialize_system(llm_model: str, embed_model: str, collection_names: List[str]):
    try:
        # Update config temporarily for the engines
        config.OLLAMA_LLM_MODEL = llm_model
        config.OLLAMA_EMBED_MODEL = embed_model
        
        ves = []
        for cname in collection_names:
            ve = VectorEngine(embed_model=embed_model, collection_name=cname)
            ves.append(ve)
        
        ge = GenerationEngine()
        ge.model_name = llm_model # Override
        
        return ves, ge, RAGOrchestrator(ves, ge)
    except Exception as e:
        st.error(f"System Initialization Failure: {e}")
        return [], None, None

# --- BACKGROUND WORKER ---
def background_worker(job_id: str, files: list, config_params: dict):
    jm = get_job_manager()
    target_path = Path(config_params['target_path'])
    
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
            # For folder processing, create a subfolder for each PDF
            if config_params.get('is_folder_mode', False):
                pdf_output_path = target_path / Path(f).stem
            else:
                pdf_output_path = target_path

            res = ie.process(f, output_root=str(pdf_output_path), skip_start=config_params['skip_head'], skip_end=config_params['skip_tail'])
            results_batch.append(res.output_path)
            jm.update_job(job_id, log=f"SUCCESS: Extracted {res.total_chunks} segments from {fname}")
            
            # Indexing (needs engines)
            if config_params['run_indexing']:
                jm.update_job(job_id, status=f"Indexing {fname}...")
                # Ensure we use the selected embed model and collection
                ve = VectorEngine(embed_model=config_params['embed_model'], collection_name=config_params['collection_name'])
                ve.add_processed_folder(res.output_path)
                jm.update_job(job_id, log=f"SUCCESS: Indexed metadata for {fname} into collection '{config_params['collection_name']}'")
                ve.unload()
                
        except Exception as e:
            jm.update_job(job_id, log=f"FAILURE: {fname} - {str(e)}")
            
    jm.update_job(job_id, status="Completed", progress=1.0, result=results_batch)


# --- ARCHITECTURAL STYLING ---
st.set_page_config(
    page_title="Amaya | Enterprise Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS - Minimalist Luxe Palette
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800;900&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-color: #1A1C2C;
        --secondary-color: #424769;
        --accent-color: #7077A1;
        --text-main: #1e1e1e;
        --bg-light: #F8FAFC; 
    }

    .stApp { background-color: var(--bg-light); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-main); }

    /* Branding Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 24px;
        padding: 10px 0;
    }
    .project-title {
        font-family: 'Montserrat', sans-serif; 
        font-weight: 900; 
        font-size: 2.6rem;
        background: linear-gradient(135deg, #1A1C2C 0%, #4A5568 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.06em;
        text-transform: lowercase;
    }
    .project-subtitle {
        color: #94A3B8; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase; margin-top: -22px;
        margin-bottom: 30px; margin-left: 4px;
    }
    
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    
    .citation-card {
        padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0;
        background-color: #ffffff; margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); transition: all 0.3s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border-color: var(--accent-color);
    }
    .citation-header {
        font-size: 0.7rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.1em; color: var(--secondary-color); margin-bottom: 10px;
        display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px;
    }
    .citation-text { font-size: 0.85rem; line-height: 1.7; color: #334155; }
    .source-label { background: #F1F5F9; color: #475569; padding: 3px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.65rem; }
    
    div.stButton > button {
        border-radius: 10px !important; font-weight: 700 !important;
        background-color: var(--primary-color) !important; color: white !important;
        border: none !important; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.02em;
    }
    div.stButton > button:hover {
        background-color: #2D3748 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26, 28, 44, 0.15) !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 32px; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; font-weight: 700; font-size: 0.95rem; color: #64748b; }
    .stTabs [aria-selected="true"] { color: var(--primary-color) !important; }
    [data-testid="stMetricValue"] { color: var(--primary-color); font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- STATE PERSISTENCE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []
if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = "Executive Overview"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    # Minimalist Tech Logo + Amaya Title
    st.markdown("""
    <div class="brand-container">
        <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHg9IjEwIiB5PSIxMCIgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiByeD0iMjAiIGZpbGw9InVybCgjYW1heWFfZ3JhZCkiLz48cGF0aCBkPSJNMzAgNzBMNTAgMzBMNzAgNzAiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PGNpcmNsZSBjeD0iNTAiIGN5PSI1NSIgcj0iNiIgZmlsbD0id2hpdGUiLz48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImFtYXlhX2dyYWQiIHgxPSIwIiB5MT0iMCIgeDI9IjEwMCIgeTI9IjEwMCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIHN0b3AtY29sb3I9IiMxQTFDMkMiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiM0QTU1NjgiLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48L3N2Zz4=" alt="Amaya Logo" style="width: 48px; height: 48px;" />
        <div class="project-title">amaya</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="project-subtitle">Intelligence Refined</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sync radio with session state
    nav_options = ["Executive Overview", "Knowledge Engineering Studio", "Cognitive Research Lab", "Infrastructure Stats"]
    
    def on_nav_change():
        st.session_state.nav_choice = st.session_state.nav_radio

    nav_choice = st.radio(
        "Navigation", 
        nav_options, 
        index=nav_options.index(st.session_state.nav_choice),
        key="nav_radio",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )
    
    st.divider()
    st.subheader("Model Configuration")
    
    # Helper to fetch models for selection
    temp_ge = GenerationEngine()
    available_models = temp_ge.list_models()
    
    if available_models:
        selected_llm = st.selectbox("LLM Inference Model", available_models, help="Select the model for text generation.")
        selected_embed = st.selectbox("Vector Embedding Model", available_models, help="Select the model for document indexing.")
    else:
        st.error("Connection Failure: Ollama server unreachable.")
        selected_llm = config.OLLAMA_LLM_MODEL
        selected_embed = config.OLLAMA_EMBED_MODEL

    st.divider()
    if nav_choice == "Cognitive Research Lab":
        st.subheader("Knowledge Domains")
        temp_ve = VectorEngine(embed_model=selected_embed)
        collections = temp_ve.list_collections()
        if collections:
            all_selected = st.checkbox("Select All Vaults", value=False)
            selected_collections = []
            with st.container(border=True):
                for col in collections:
                    if st.checkbox(col, value=all_selected, key=f"sel_{col}"):
                        selected_collections.append(col)
            
            if not selected_collections:
                st.warning("Please select at least one knowledge domain.")
                selected_collections = ["default_vault"]
        else:
            st.warning("No active collections identified.")
            selected_collections = ["default_vault"]
    else:
        selected_collections = ["default_vault"]

    st.divider()
    st.subheader("Operational Scope")
    run_extraction = st.checkbox("Enable Document Extraction", value=True)
    run_indexing = st.checkbox("Enable Vector Indexing", value=True)
    
    if st.button("Purge Session State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_logs = []
        st.cache_resource.clear()
        st.rerun()

# Initialize engines only when needed for performance
if nav_choice == "Cognitive Research Lab":
    ves, ge, orch = initialize_system(selected_llm, selected_embed, selected_collections)
else:
    ves, ge, orch = [], None, None

# --- MAIN INTERFACE ---

# 0. EXECUTIVE OVERVIEW (Landing Page)
if nav_choice == "Executive Overview":
    st.header("Executive Overview")
    st.markdown("Welcome to **Amaya**, the next-generation Operational Intelligence Platform. Navigate through our core modules to manage your knowledge architecture and extract strategic, hallucination-free insights.")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.subheader("🛠️ Knowledge Engineering")
            st.markdown("""
            **The Foundation of Intelligence.**
            
            Architect your data environment through high-fidelity ingestion, structural decomposition, and semantic indexing.
            
            *   Convert raw PDFs into neural assets.
            *   Configure OCR and Table Vision.
            *   Build custom Knowledge Domains.
            """)
            if st.button("Enter Studio", use_container_width=True, key="btn_eng"):
                st.session_state.nav_choice = "Knowledge Engineering Studio"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("🔬 Cognitive Research")
            st.markdown("""
            **Advanced Insight Synthesis.**
            
            Engage with your knowledge base through state-of-the-art RAG orchestration and cross-domain research.
            
            *   Multi-vault semantic search.
            *   Cited evidence & provenance.
            *   Neural research assistance.
            """)
            if st.button("Initialize Research", use_container_width=True, key="btn_res"):
                st.session_state.nav_choice = "Cognitive Research Lab"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.subheader("📊 Infrastructure Stats")
            st.markdown("""
            **Operational Health & Metrics.**
            
            Monitor compute resources, model health, and repository telemetry for the entire ecosystem.
            
            *   GPU/CPU resource tracking.
            *   Ollama inference monitoring.
            *   Index growth & health analytics.
            """)
            if st.button("View Telemetry", use_container_width=True, key="btn_inf"):
                st.session_state.nav_choice = "Infrastructure Stats"
                st.rerun()

    st.divider()
    with st.expander("System Architecture Overview"):
        st.markdown("""
        **Amaya** utilizes a hybrid-cloud architecture:
        - **Local Ingestion:** IBM Docling powered extraction for 100% data privacy.
        - **Vector Storage:** ChromaDB persistent storage with cosine-similarity indexing.
        - **Inference Runtime:** Local-first Ollama server for LLM and Embedding execution.
        """)

# 1. KNOWLEDGE ENGINEERING STUDIO
elif nav_choice == "Knowledge Engineering Studio":
    st.header("Knowledge Engineering Studio")
    st.markdown("Orchestrate the high-fidelity ingestion, structural analysis, and neural indexing of unstructured information assets.")
    
    jm = get_job_manager()
    
    # Robust Auto-Refresh Monitor
    if "last_job_id" in st.session_state:
        job = jm.get_job(st.session_state.last_job_id)
        if job:
            status_container = st.empty()
            with status_container.container(border=True):
                if job["status"] == "Completed":
                    st.success(f"✅ **Engineering Success:** '{job['name']}' has been integrated into the vault.")
                    st.balloons()
                    if st.button("Acknowledge & Clear"):
                        del st.session_state.last_job_id
                        st.rerun()
                else:
                    st.info(f"⏳ **Knowledge Construction in Progress:** {job['name']}")
                    st.write(f"Studio Phase: `{job['status']}`")
                    st.progress(job["progress"])
                    st.caption(f"Targeting Domain: `{st.session_state.get('target_col_input', 'default')}`")
                    time.sleep(2)
                    st.rerun()

    processing_mode = st.radio("Asset Source Configuration", ["Individual Document", "Directory Batch"], horizontal=True)

    with st.container(border=True):
        if processing_mode == "Individual Document":
            c1, c2 = st.columns([2, 1])
            with c1:
                source_path = st.text_input("Source Document Path (PDF)", placeholder="C:\\Data\\Executive_Report.pdf")
                target_path = st.text_input("Asset Repository Root", value=str(config.OUTPUT_ROOT))
                target_collection = st.text_input("Target Knowledge Domain", value="intel_vault", key="target_col_input")
            with c2:
                s_range = st.expander("Boundary Constraints", expanded=True)
                with s_range:
                    skip_head = st.number_input("Prefix Page Offset", 0, 500, 0)
                    skip_tail = st.number_input("Suffix Page Offset", 0, 500, 0)
            
            # PDF Preview
            if source_path and os.path.exists(source_path) and source_path.lower().endswith(".pdf"):
                try:
                    doc = fitz.open(source_path)
                    total_p = len(doc)
                    st.markdown(f"**Live Document Preview** | {total_p} Pages")
                    page_to_show = st.slider("Page Selection", 1, total_p, 1)
                    page = doc.load_page(page_to_show - 1)
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.6, 0.6))
                    st.image(pix.tobytes(), use_container_width=False)
                    doc.close()
                except Exception as e:
                    st.error(f"Render Error: {e}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                source_path = st.text_input("Source Directory", placeholder="C:\\Archive\\Annual_Reports")
                target_path = st.text_input("Global Repository Root", value=str(config.OUTPUT_ROOT))
            with c2:
                target_collection = st.text_input("Target Knowledge Domain", value="batch_intel", key="target_col_input")
            
            st.info("💡 **Architectural Logic:** Each document will be isolated into a discrete subdirectory within the repository root.")
            skip_head, skip_tail = 0, 0

    st.subheader("Extraction Intelligence")
    t1, t2, t3 = st.tabs(["Vision & Logic", "Chunking Strategy", "Table Reconstruction"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        ocr_enabled = cc1.toggle("Enable OCR Vision (For Scanned Assets)", value=True)
        formula_enabled = cc2.toggle("Extract Mathematical Notation (LaTeX)", value=True)
    with t2:
        seg_strategy = st.selectbox("Semantic Segmentation", ["Hybrid (Context Aware)", "Hierarchical (Structural)"])
        cc1, cc2 = st.columns(2)
        chunk_val = cc1.number_input("Token Window Size", 256, 1024, 512, step=64)
        overlap_val = cc2.number_input("Contextual Overlap", 0, 128, 64, step=8)
    with t3:
        st.markdown("Determine the computational fidelity for tabular data reconstruction.")
        extract_mode = st.selectbox("Fidelity Level", ["accurate", "fast"], index=0, help="Accurate mode utilizes TableFormerV2 models for structural precision.")

    if st.button("Initiate Engineering Workflow", type="primary", use_container_width=True):
        if not source_path or not os.path.exists(source_path):
            st.error("Validation Failure: Valid system path required.")
        else:
            files = [str(f) for f in Path(source_path).glob("*.pdf")] if os.path.isdir(source_path) else ([source_path] if source_path.endswith(".pdf") else [])
            if not files:
                st.warning("Notification: No valid PDF candidates identified.")
            else:
                job_id = jm.create_job(Path(files[0]).name if len(files)==1 else f"Batch ({len(files)} files)")
                config_params = {
                    "target_path": target_path, "use_ocr": ocr_enabled, "use_formula": formula_enabled,
                    "seg_strategy": seg_strategy.split()[0].lower(), "extract_mode": extract_mode,
                    "chunk_val": chunk_val, "skip_head": skip_head, "skip_tail": skip_tail, 
                    "run_indexing": run_indexing, "embed_model": selected_embed,
                    "collection_name": target_collection, "is_folder_mode": processing_mode == "Directory Batch"
                }
                thread = threading.Thread(target=background_worker, args=(job_id, files, config_params))
                thread.start()
                st.session_state.last_job_id = job_id
                st.rerun()

# 2. COGNITIVE RESEARCH LAB (Chat Interface)
elif nav_choice == "Cognitive Research Lab":
    st.header("Cognitive Research Lab")
    
    # Check for indices
    for v_eng in ves: v_eng.load()
    total_chunks = sum(v_eng.collection.count() for v_eng in ves)

    if total_chunks == 0:
        with st.container(border=True):
            st.subheader("🚀 Initializing Strategic Insights")
            st.markdown("""
            The Cognitive Research Lab is ready, but your neural vault is currently empty. Follow these steps to begin:
            
            1. **Navigate** to the **Knowledge Engineering Studio** in the sidebar.
            2. **Connect** a data source (PDF or directory) to the ingestion engine.
            3. **Define** a unique **Knowledge Domain** for your assets.
            4. **Initiate** the workflow. Return here once the knowledge architecture is complete.
            
            *Operational Note: Multi-domain cross-referencing is enabled by default.*
            """)
            st.info("The system will automatically synthesize evidence across all selected knowledge domains.")
        st.stop()

    st.markdown(f"**Active Domains:** `{', '.join(selected_collections)}` | **Inference Runtime:** `{selected_llm}`")
    
    chat_container = st.container(height=520, border=False)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and "sources" in m:
                    with st.expander("Evidence Registry & Provenance"):
                        for i, s in enumerate(m["sources"]):
                            meta = s["metadata"]
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span>DOCUMENT REF {i+1} | {meta.get('pdf_name', 'Unknown')}</span>
                                    <span class="source-label">PG {meta.get('pages', 'N/A')}</span>
                                </div>
                                <div class="citation-text">{s['text']}</div>
                                <div style="font-size: 0.72rem; color: #666; margin-top: 10px; border-top: 1px solid #f8f9fa; padding-top: 6px;">
                                    <b>Vault Domain:</b> {meta.get('source', 'Unknown')} |
                                    <b>Hierarchy:</b> {meta.get('breadcrumb', 'N/A')} | 
                                    <b>Tabular Content:</b> {'Identified' if meta.get('is_table') else 'None'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    if q := st.chat_input("Submit inquiry to cognitive lab..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with chat_container:
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text = ""
                hits = []
                stream = orch.query_stream(q, top_k=5)
                for chunk in stream:
                    if chunk["type"] == "sources": 
                        hits = [s.model_dump() for s in chunk["content"]]
                    elif chunk["type"] == "answer_chunk":
                        full_text += chunk["content"]
                        placeholder.markdown(full_text + "▌")
                
                placeholder.markdown(full_text)
                
                if hits:
                    with st.expander("Evidence Registry & Provenance", expanded=False):
                        for i, s in enumerate(hits):
                            meta = s["metadata"]
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span>DOCUMENT REF {i+1} | {meta.get('pdf_name', 'Unknown')}</span>
                                    <span class="source-label">PG {meta.get('pages', 'N/A')}</span>
                                </div>
                                <div class="citation-text">{s['text']}</div>
                                <div style="font-size: 0.72rem; color: #666; margin-top: 10px; border-top: 1px solid #f8f9fa; padding-top: 6px;">
                                    <b>Vault Domain:</b> {meta.get('source', 'Unknown')} |
                                    <b>Hierarchy:</b> {meta.get('breadcrumb', 'N/A')} | 
                                    <b>Tabular Content:</b> {'Identified' if meta.get('is_table') else 'None'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_text, "sources": hits})


# 3. INFRASTRUCTURE STATS
elif nav_choice == "Infrastructure Stats":
    st.header("Infrastructure & Health Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Compute Resources")
        import torch
        st.metric("Acceleration Engine", "NVIDIA CUDA (GPU)" if torch.cuda.is_available() else "Intel/AMD (CPU)")
        # Total index count across all collections
        temp_ve = VectorEngine(embed_model=selected_embed)
        collections = temp_ve.list_collections()
        total_chunks = 0
        for cname in collections:
            tve = VectorEngine(embed_model=selected_embed, collection_name=cname)
            tve.load()
            total_chunks += tve.collection.count()
        st.metric("Global Semantic Index Count", f"{total_chunks} Chunks")
    with c2:
        st.subheader("Environment Inventory")
        st.code(f"Core Repository: {config.BASE_DIR}\nOllama Protocol: {config.OLLAMA_BASE_URL}\nInference Runtime: {selected_llm}\nEmbedding Runtime: {selected_embed}")

st.divider()
st.caption(f"Amaya Enterprise | Operational Intelligence Platform | {datetime.now().strftime('%Y-%m-%d')}")
