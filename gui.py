import streamlit as st
import os
import sys
from pathlib import Path
import time
import logging
from ingestion_engine import IngestionEngine

# --- CUSTOM LOGGING HANDLER FOR STREAMLIT ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_area):
        super().__init__()
        self.log_area = log_area
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
        # Keep only the last 50 lines for performance
        if len(self.log_buffer) > 50:
            self.log_buffer.pop(0)
        self.log_area.code("\n".join(self.log_buffer))

# --- APP CONFIG ---
st.set_page_config(page_title="Docling Pro Ingestion GUI", layout="wide")

st.title("🚀 Docling Professional Ingestion (Offline)")
st.markdown("Automate PDF parsing and chunking with professional-grade controls.")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Core Configuration")
    chunking_strategy = st.selectbox(
        "Chunking Strategy", 
        ["Hybrid", "Hierarchical"],
        help="Select the method for breaking down document text."
    )
    
    with st.expander("❓ Help: Which strategy to choose?"):
        st.markdown("""
        **Hybrid Chunking:**
        - *Best for:* Certificates, forms, and technical sheets.
        - *Why:* Keeps labels and values together. Handles tables exceptionally well.
        
        **Hierarchical Chunking:**
        - *Best for:* Text-heavy documents, reports, and books.
        - *Why:* Respects document structure (headings, sub-headings) for better semantic context.
        """)

    st.divider()
    st.subheader("Parsing Options")
    use_formula = st.checkbox("Enable Formulas", value=False)
    use_ocr = st.checkbox("Enable OCR (RapidOCR)", value=True)
    
    st.divider()
    st.subheader("Range Settings")
    skip_start = st.number_input("Skip Pages (Start)", min_value=0, value=0)
    skip_end = st.number_input("Skip Pages (End)", min_value=0, value=0)

# --- MAIN INTERFACE: TABS ---
tab1, tab2 = st.tabs(["📥 Ingestion", "📜 Live Logs"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio("Processing Mode", ["Single File", "Batch (Folder)"])
    
    with col2:
        output_root = st.text_input("Output Directory", value=str(Path.home() / "parsing_output"))

    if mode == "Single File":
        pdf_path = st.text_input("Input PDF Path", placeholder="C:\\path\\to\\file.pdf")
    else:
        folder_path = st.text_input("Input Folder Path", placeholder="C:\\path\\to\\pdf_folder")
        pdf_path = None # Will be populated during loop

    st.divider()
    
    # Progress UI
    status_msg = st.empty()
    progress_bar = st.progress(0)
    
    if st.button("▶️ Start Ingestion", type="primary", use_container_width=True):
        # Validation
        inputs_valid = False
        files_to_process = []
        
        if mode == "Single File":
            if pdf_path and os.path.exists(pdf_path):
                files_to_process = [pdf_path]
                inputs_valid = True
            else:
                st.error("Invalid file path.")
        else:
            if folder_path and os.path.exists(folder_path):
                files_to_process = [str(f) for f in Path(folder_path).glob("*.pdf")]
                if files_to_process:
                    inputs_valid = True
                else:
                    st.error("No PDFs found in the selected folder.")
            else:
                st.error("Invalid folder path.")

        if inputs_valid:
            # Setup Logger for tab2
            with tab2:
                st.subheader("Real-time Process Logs")
                log_display = st.empty()
                streamlit_handler = StreamlitLogHandler(log_display)
                streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                
                # Get the logger from ingestion_engine
                engine_logger = logging.getLogger("IngestionEngine")
                engine_logger.addHandler(streamlit_handler)

            total_files = len(files_to_process)
            try:
                engine = IngestionEngine(
                    use_ocr=use_ocr, 
                    use_formula=use_formula, 
                    chunking_strategy=chunking_strategy
                )
                
                for idx, file in enumerate(files_to_process):
                    p_val = int(((idx) / total_files) * 100)
                    progress_bar.progress(p_val)
                    
                    filename = Path(file).name
                    status_msg.info(f"📁 Processing {idx+1}/{total_files}: **{filename}**")
                    
                    # Core process call with callback for status updates
                    def update_status(text):
                        status_msg.markdown(f"**Current Step:** {text} (File {idx+1}/{total_files})")
                    
                    engine.process(
                        file, 
                        output_root, 
                        skip_start=skip_start, 
                        skip_end=skip_end, 
                        status_callback=update_status
                    )
                
                progress_bar.progress(100)
                status_msg.success(f"✅ Successfully processed {total_files} file(s)!")
                st.balloons()
                
            except Exception as e:
                status_msg.error(f"❌ FATAL ERROR: {str(e)}")
                st.exception(e)
            finally:
                # Cleanup logging handler
                engine_logger.removeHandler(streamlit_handler)
                import gc
                gc.collect()

with tab2:
    if 'log_display' not in locals():
        st.info("Logs will appear here once the process starts.")

st.divider()
st.caption("Docling Professional Suite | Offline Mode Enabled")
