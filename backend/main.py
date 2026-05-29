"""FastAPI app for FEK OCR + AI structuring pipeline."""

from __future__ import annotations

import asyncio
import fitz
import json
import logging
import os
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from markitdown import MarkItDown
    from backend.ocr.detector import detect_document, extract_images_from_pdf
    from backend.ocr.extractor import extract_document_text_layer
    from backend.ocr.engine import ocr_document, ocr_image
    from backend.ai.structurer import build_outputs
    from backend.ai.model_manager import choose_model, classify_image
    from backend.ai.diff_engine import apply_amendments
    from backend.ai.review_queue import enqueue_review, list_pending_reviews
    from backend.storage.organizer import ensure_dirs, output_paths, write_text
    from backend.storage.versioning import get_current_law_text, snapshot_and_update
    from backend.config import (
        INPUT_DIR,
        OLLAMA_BASE_URL,
        OUTPUT_HTML_DIR,
        OUTPUT_XML_DIR,
        DOCINTEL_ENDPOINT,
        IMAGES_DIR
    )
except ImportError:
    from markitdown import MarkItDown
    from ocr.detector import detect_document, extract_images_from_pdf
    from ocr.extractor import extract_document_text_layer
    from ocr.engine import ocr_document, ocr_image
    from ai.structurer import build_outputs
    from ai.model_manager import choose_model, classify_image
    from ai.diff_engine import apply_amendments
    from ai.review_queue import enqueue_review, list_pending_reviews
    from storage.organizer import ensure_dirs, output_paths, write_text
    from storage.versioning import get_current_law_text, snapshot_and_update
    from config import (
        INPUT_DIR,
        OLLAMA_BASE_URL,
        OUTPUT_HTML_DIR,
        OUTPUT_XML_DIR,
        DOCINTEL_ENDPOINT,
        IMAGES_DIR
    )

app = FastAPI(title="FEK Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

from backend.config import DATA_DIR
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

logger = logging.getLogger("fek_processor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ── WebSocket manager ────────────────────────────────────────────────────────


class WSManager:
    def __init__(self) -> None:
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, event: Dict) -> None:
        dead: List[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(event, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


# ── Event helpers ─────────────────────────────────────────────────────────────


async def _event(stage: str, message: str, **extra) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": message,
    }
    payload.update(extra)
    await ws_manager.broadcast(payload)


async def _log(level: str, source: str, message: str, **extra) -> None:
    await _event("log", message, level=level.upper(), source=source, **extra)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@app.get("/outputs")
async def list_outputs():
    """List all processed output files."""
    ensure_dirs()
    results = []
    html_dir = OUTPUT_HTML_DIR
    xml_dir = OUTPUT_XML_DIR

    if os.path.isdir(html_dir):
        for name in sorted(os.listdir(html_dir)):
            if name.endswith(".html"):
                base = name[:-5]
                xml_path = os.path.join(xml_dir, f"{base}.xml")
                results.append(
                    {
                        "id": base,
                        "html_url": f"/outputs/html/{name}",
                        "xml_url": (
                            f"/outputs/xml/{base}.xml"
                            if os.path.exists(xml_path)
                            else None
                        ),
                    }
                )
    return JSONResponse({"outputs": results})


@app.get("/outputs/html/{filename}")
async def get_html_output(filename: str):
    path = os.path.join(OUTPUT_HTML_DIR, filename)
    if not os.path.exists(path) or not filename.endswith(".html"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/outputs/xml/{filename}")
async def get_xml_output(filename: str):
    path = os.path.join(OUTPUT_XML_DIR, filename)
    if not os.path.exists(path) or not filename.endswith(".xml"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="application/xml")


@app.get("/review-queue")
async def get_review_queue():
    """Return all pending items in the human review queue."""
    items = list_pending_reviews()
    return JSONResponse({"pending": len(items), "items": items})


# ── PDF processing pipeline ───────────────────────────────────────────────────


async def process_pdf(pdf_path: str) -> Dict:
    filename = os.path.basename(pdf_path)
    logger.info("process_pdf started file=%s", filename)
    await _log("info", "process_pdf", f"started file={filename}")
    await _event("start", f"Processing {filename}")

    # ── Extract Images ──────────────────────────────────────
    await _event("extract_images", "Extracting images from PDF")
    extracted_images = extract_images_from_pdf(pdf_path)
    
    loop = asyncio.get_running_loop()
    
    # ── Classify and OCR Images ─────────────────────────────
    image_processed_content = {}  # xref -> text or marker

    for i, img_info in enumerate(extracted_images):
        await _event("ocr", f"Processing image {i+1}/{len(extracted_images)}", done=i, total=len(extracted_images))

        category = await asyncio.to_thread(classify_image, img_info["image_path"])
        if category == "pure_text":
            text = await asyncio.to_thread(ocr_image, img_info["image_path"])
            image_processed_content[img_info["xref"]] = text
        else:
            # Keep as image marker for later replacement in markdown
            rel_path = os.path.relpath(img_info["image_path"], os.path.dirname(os.path.dirname(__file__)))
            image_processed_content[img_info["xref"]] = f"![{category}](/{rel_path})"

    # ── MarkItDown Conversion (Page by Page) ────────────────
    await _event("markitdown", "Converting PDF to Markdown using MarkItDown (page by page)")
    if DOCINTEL_ENDPOINT:
        md_converter = MarkItDown(docintel_endpoint=DOCINTEL_ENDPOINT)
    else:
        md_converter = MarkItDown()

    doc = fitz.open(pdf_path)
    merged_md_parts = []

    for page_num in range(1, len(doc) + 1):
        # Create a 1-page temporary PDF for MarkItDown
        temp_page_path = f"{pdf_path}_page_{page_num}.pdf"
        temp_doc = fitz.open()
        temp_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        temp_doc.save(temp_page_path)
        temp_doc.close()

        try:
            page_result = await asyncio.to_thread(md_converter.convert, temp_page_path)
            page_md = page_result.text_content

            # Find images for this page
            page_images = [img for img in extracted_images if img["page_num"] == page_num]
            for img in page_images:
                processed = image_processed_content.get(img["xref"], "")
                if "![" in processed: # it's a photo/table/chart
                    # Try to insert marker or append if not found
                    page_md += f"\n\n{processed}\n\n"
                elif processed: # it's pure_text
                    # Check if already there, otherwise append
                    if processed[:50] not in page_md:
                        page_md += f"\n\n[OCR TEXT]:\n{processed}\n\n"

            merged_md_parts.append(f"[PAGE {page_num}]\n\n{page_md}")
        finally:
            if os.path.exists(temp_page_path):
                os.remove(temp_page_path)

    doc.close()
    raw_text = "\n\n".join(merged_md_parts)

    await _event("merge", "Markdown content ready", chars=len(raw_text))

    # ── AI structuring ───────────────────────────────────────
    def ai_log(source: str, message: str) -> None:
        logger.info("%s %s", source, message)
        asyncio.run_coroutine_threadsafe(_log("info", source, message), loop)

    decision = await asyncio.to_thread(choose_model, None, ai_log)
    await _event(
        "ai_start",
        "AI structuring started",
        model=decision.model,
        free_ram_gb=decision.free_ram_gb,
        model_reason=decision.reason,
        raw_chars=len(raw_text),
    )

    structured = await asyncio.to_thread(build_outputs, raw_text, ai_log)
    chunks_processed = structured.data.get("metadata", {}).get("chunks_processed", 1)
    await _event(
        "ai",
        "AI structuring completed",
        document_id=structured.data.get("document_id"),
        title=structured.data.get("title"),
        ai_mode=structured.data.get("metadata", {}).get("ai_mode"),
        model=structured.data.get("metadata", {}).get("model"),
        chunks_processed=chunks_processed,
        ai_preview=structured.html_text[:1500],
    )

    # ── Save outputs ─────────────────────────────────────────
    doc_id = structured.data.get("document_id") or filename
    paths = output_paths(doc_id)
    write_text(paths["xml"], structured.xml_text)
    write_text(paths["html"], structured.html_text)
    await _event(
        "save",
        "XML/HTML saved",
        xml=paths["xml"],
        html=paths["html"],
        html_url=f"/outputs/html/{os.path.basename(paths['html'])}",
        xml_url=f"/outputs/xml/{os.path.basename(paths['xml'])}",
    )

    # ── Apply amendments ─────────────────────────────────────
    amendment_notes: List[str] = []
    for amend in structured.data.get("amendments", []):
        law_id = amend.get("target_law_id")
        if not law_id:
            continue

        current = get_current_law_text(law_id)
        result = apply_amendments(current, [amend])
        amendment_notes.extend(result.notes)
        await _log(
            "info",
            "amend",
            f"target={law_id} changed={result.changed} notes={len(result.notes)}",
        )

        if result.changed:
            amend["diff"] = result.diff  # Store diff in the amendment object
            current_path, version_path = snapshot_and_update(
                law_id=law_id,
                new_text=result.new_law_text,
                effective_date=structured.data.get("publication_date"),
                diff=result.diff
            )
            await _event(
                "amend",
                f"Applied amendment to {law_id}",
                current=current_path,
                version=version_path,
            )

    # ── Review queue ─────────────────────────────────────────
    if structured.needs_review or amendment_notes:
        review_path = enqueue_review(
            {
                "document_id": doc_id,
                "title": structured.data.get("title"),
                "questions": structured.review_questions + amendment_notes,
                "raw_excerpt": raw_text[:3000],
                "metadata": structured.data.get("metadata", {}),
            }
        )
        await _event(
            "review",
            "Queued for human review",
            review_file=review_path,
            question_count=len(structured.review_questions) + len(amendment_notes),
        )

    await _event(
        "done",
        f"Finished {filename}",
        document_id=doc_id,
        html_url=f"/outputs/html/{os.path.basename(paths['html'])}",
    )
    return {
        "document_id": doc_id,
        "title": structured.data.get("title"),
        "xml": paths["xml"],
        "html": paths["html"],
        "html_url": f"/outputs/html/{os.path.basename(paths['html'])}",
        "xml_url": f"/outputs/xml/{os.path.basename(paths['xml'])}",
        "needs_review": structured.needs_review or bool(amendment_notes),
    }


@app.post("/process")
async def process(files: List[UploadFile] = File(...)):
    ensure_dirs()
    os.makedirs(INPUT_DIR, exist_ok=True)

    results = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            await _event("error", f"Skipped non-PDF: {f.filename}")
            continue

        path = os.path.join(INPUT_DIR, f.filename)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)

        try:
            result = await process_pdf(path)
            results.append(result)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("failed file=%s", f.filename)
            await _log(
                "error",
                "process",
                f"failed file={f.filename} error={exc}",
                traceback=tb,
            )
            await _event("error", f"Failed {f.filename}", error=str(exc))

    return JSONResponse({"ok": True, "count": len(results), "results": results})


# ── AI purge ──────────────────────────────────────────────────────────────────


@app.post("/ai/purge")
async def ai_purge():
    """Gracefully stop Ollama model sessions, remove local models, kill the server."""
    import ollama as _ollama

    removed: List[str] = []
    errors: List[str] = []

    await _log("warning", "ai_purge", "purge requested")

    try:
        client = _ollama.Client(host=OLLAMA_BASE_URL, timeout=10)

        # Stop running models
        try:
            ps_resp = client.ps()
            active_items = (
                ps_resp.get("models", [])
                if isinstance(ps_resp, dict)
                else list(getattr(ps_resp, "models", None) or [])
            )
            for item in active_items:
                name = (
                    item.get("model") or item.get("name")
                    if isinstance(item, dict)
                    else getattr(item, "model", None) or getattr(item, "name", None)
                )
                if not name:
                    continue
                try:
                    stop_fn = getattr(client, "stop", None)
                    if stop_fn:
                        stop_fn(name)
                    await _log("info", "ai_purge", f"stopped model={name}")
                except Exception as exc:
                    errors.append(f"stop {name}: {exc}")
        except Exception as exc:
            await _log("warning", "ai_purge", f"ps() unavailable: {exc}")

        # Delete local models
        try:
            list_resp = client.list()
            model_items = (
                list_resp.get("models", [])
                if isinstance(list_resp, dict)
                else list(getattr(list_resp, "models", None) or [])
            )
            for item in model_items:
                name = (
                    item.get("model") or item.get("name")
                    if isinstance(item, dict)
                    else getattr(item, "model", None) or getattr(item, "name", None)
                )
                if not name:
                    continue
                try:
                    client.delete(name)
                    removed.append(name)
                    await _log("info", "ai_purge", f"removed model={name}")
                except Exception as exc:
                    errors.append(f"delete {name}: {exc}")
        except Exception as exc:
            errors.append(f"list: {exc}")
            await _log("error", "ai_purge", f"failed listing models: {exc}")

    except Exception as exc:
        errors.append(f"client init: {exc}")
        await _log("warning", "ai_purge", f"could not connect to Ollama: {exc}")

    # Kill ollama serve
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], check=False)
        await _log("info", "ai_purge", "ollama serve process terminated")
    except Exception as exc:
        errors.append(f"pkill: {exc}")

    await _event(
        "ai_purge",
        "AI purge completed",
        removed_models=removed,
        errors=errors,
    )
    return JSONResponse(
        {"ok": len(errors) == 0, "removed_models": removed, "errors": errors}
    )
