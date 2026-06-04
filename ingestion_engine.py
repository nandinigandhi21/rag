import os
import time
import json
import logging
import gc
import fitz
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from config import config
from schema import IngestionResult, DocumentMetadata, Chunk

# Apply environment setup
config.setup_environment()

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableStructureOptions
from docling_core.types.doc.base import ImageRefMode
from docling.chunking import HybridChunker, HierarchicalChunker

logger = logging.getLogger("IngestionEngine")

class IngestionEngine:
    """
    An optimized document ingestion engine specializing in high-fidelity PDF extraction.
    
    This engine leverages the Docling library to convert complex PDFs into structured 
    Markdown, with a specific focus on preserving table structures, mathematical 
    formulas, and spatial hierarchy. It handles the full pipeline from raw PDF 
    to searchable, metadata-rich document chunks.

    Attributes:
        strategy (str): The chunking methodology ('hybrid' or 'hierarchical').
        pipeline_options (PdfPipelineOptions): Configuration for the Docling conversion pipeline.
        converter (DocumentConverter): The core converter instance for PDF processing.
        chunker (BaseChunker): The initialized chunker based on the selected strategy.
    """
    def __init__(self, use_ocr: bool = True, use_formula: bool = False, chunking_strategy: str = "hybrid", table_mode: str = "accurate", **chunker_kwargs):
        """
        Initializes the IngestionEngine with specific extraction and chunking parameters.

        Args:
            use_ocr (bool): Whether to use Optical Character Recognition for scanned content.
            use_formula (bool): Whether to enable specialized formula enrichment.
            chunking_strategy (str): The method for breaking documents into chunks ('hybrid' or 'hierarchical').
            table_mode (str): Extraction mode for tables ('accurate' or 'fast').
            **chunker_kwargs: Additional arguments for chunker configuration (max_tokens, merge_peers).
        """
        self.strategy = chunking_strategy.lower()
        
        # --- ACCURATE TABLE CONFIGURATION ---
        self.pipeline_options = PdfPipelineOptions()
        
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.table_structure_options.mode = table_mode 
        
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_code_enrichment = True
        self.pipeline_options.do_formula_enrichment = use_formula
        
        if use_ocr:
            self.pipeline_options.do_ocr = True
            self.pipeline_options.ocr_options = RapidOcrOptions()
        else:
            self.pipeline_options.do_ocr = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        tokenizer_path = str(config.MODELS_CACHE / "docling-project--CodeFormulaV2")
        max_tokens = chunker_kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS)
        merge_peers = chunker_kwargs.get("merge_peers", config.DEFAULT_MERGE_PEERS)

        if self.strategy == "hierarchical":
            self.chunker = HierarchicalChunker(tokenizer=tokenizer_path, max_tokens=max_tokens)
        else:
            self.chunker = HybridChunker(tokenizer=tokenizer_path, max_tokens=max_tokens, merge_peers=merge_peers)

    def _save_table(self, table, i, table_dir):
        """Exports extracted tables to CSV format."""
        try:
            csv_path = table_dir / f"table_{i+1:03d}.csv"
            df = table.export_to_dataframe()
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save table {i}: {e}")

    def _save_image(self, element, i, img_dir):
        """Saves extracted images to disk."""
        try:
            if element.image:
                img_name = f"image_{i+1:03d}.png"
                element.image.pil_image.save(img_dir / img_name)
                element.image.uri = Path("images") / img_name
        except Exception as e:
            logger.error(f"Failed to save image {i}: {e}")

    def process(self, pdf_path: str, output_root: Optional[str] = None, skip_start: int = 0, skip_end: int = 0, status_callback: Optional[Callable[[str], None]] = None, batch_size: int = 15) -> IngestionResult:
        """
        Executes the ingestion pipeline in batches to prevent memory exhaustion (std::bad_alloc).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

        output_root = Path(output_root or config.OUTPUT_ROOT)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        job_id = f"{pdf_path.stem}_{self.strategy}_{timestamp_str}"
        job_dir = output_root / job_id
        
        img_dir = job_dir / "images"
        table_dir = job_dir / "tables"
        for d in [job_dir, img_dir, table_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing (PDF: {pdf_path.name}) with Batch Size: {batch_size}")
        start_time = time.time()

        all_chunks_data = []
        md_path = job_dir / f"{pdf_path.stem}.md"
        
        # Initialize MD file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Processed Document: {pdf_path.name}\n\n")

        try:
            # 1. Determine Scope
            with fitz.open(pdf_path) as pdf_doc:
                total_pages = len(pdf_doc)
                doc_meta = pdf_doc.metadata
                title = doc_meta.get("title") or pdf_path.stem
                author = doc_meta.get("author") or "Unknown"
            
            start_p = skip_start + 1
            end_p = total_pages - skip_end
            if start_p > end_p: end_p = total_pages 

            # 2. Batch Processing Loop
            for batch_start in range(start_p, end_p + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, end_p)
                
                msg = f"Processing batch: Pages {batch_start}-{batch_end} of {total_pages}..."
                if status_callback: status_callback(msg)
                logger.info(msg)

                # Convert Batch
                conv_res = self.converter.convert(pdf_path, page_range=(batch_start, batch_end))
                doc = conv_res.document
                
                # Assets
                asset_offset = len(all_chunks_data)
                for i, table in enumerate(doc.tables):
                    self._save_table(table, asset_offset + i, table_dir)
                for i, element in enumerate(doc.pictures):
                    self._save_image(element, asset_offset + i, img_dir)

                # Append Markdown
                with open(md_path, "a", encoding="utf-8") as f:
                    f.write(doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED))
                    f.write("\n\n")

                # Chunking Batch
                for i, chunk in enumerate(self.chunker.chunk(dl_doc=doc)):
                    page_nos = set()
                    is_tab = False
                    
                    if hasattr(chunk.meta, 'doc_items'):
                        for item in chunk.meta.doc_items:
                            if hasattr(item, 'label') and item.label == 'table': is_tab = True
                            if hasattr(item, 'prov') and item.prov:
                                for p in item.prov: page_nos.add(p.page_no)
                    
                    headings = getattr(chunk.meta, 'headings', []) or []
                    
                    meta = DocumentMetadata(
                        pages=sorted(list(page_nos)),
                        total_pages=total_pages,
                        headings=headings,
                        breadcrumb=" > ".join(headings) if headings else "General",
                        is_table=is_tab,
                        char_count=len(chunk.text),
                        strategy=self.strategy,
                        source_name=pdf_path.name,
                        doc_title=title,
                        doc_author=author
                    )

                    # --- CONTEXT ENRICHMENT (Context-Augmented RAG) ---
                    # We prepend document metadata directly to the text chunk. This ensures the 
                    # vector engine can search based on document context (like headings) 
                    # and the LLM knows the exact provenance of every piece of info.
                    pages_str = ", ".join(map(str, meta.pages)) if meta.pages else "N/A"
                    context_header = f"[Source: {pdf_path.name} | Section: {meta.breadcrumb} | Page: {pages_str}]\n"
                    enriched_text = context_header + chunk.text
                    
                    all_chunks_data.append(Chunk(
                        chunk_id=len(all_chunks_data)+1, 
                        text=enriched_text, 
                        metadata=meta
                    ).model_dump())

                # Explicit Cleanup
                del conv_res
                del doc
                gc.collect()

            # 3. Final Save
            with open(job_dir / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(all_chunks_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Ingestion complete. Total chunks: {len(all_chunks_data)}")

            return IngestionResult(
                job_id=job_id, pdf_name=pdf_path.name, output_path=str(job_dir),
                total_chunks=len(all_chunks_data), duration_seconds=round(time.time()-start_time, 2)
            )

        except Exception as e:
            logger.error(f"Ingestion process failed: {e}")
            raise
