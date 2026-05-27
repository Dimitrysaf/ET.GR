"""FastAPI app for FEK OCR + AI structuring pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import logging
import traceback
import subprocess
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket, WebSocketDisconnect
import ollama

try:
    from backend.ocr.detector import detect_document
    from backend.ocr.extractor import extract_document_text_layer
    from backend.ocr.engine import ocr_document
    from backend.ai.structurer import build_outputs
    from backend.ai.model_manager import choose_model
    from backend.ai.diff_engine import apply_amendments
    from backend.ai.review_queue import enqueue_review
    from backend.storage.organizer import ensure_dirs, output_paths, write_text
    from backend.storage.versioning import get_current_law_text, snapshot_and_update
    from backend.config import INPUT_DIR
except ImportError:
    from ocr.detector import detect_document
    from ocr.extractor import extract_document_text_layer
    from ocr.engine import ocr_document
    from ai.structurer import build_outputs
    from ai.model_manager import choose_model
    from ai.diff_engine import apply_amendments
    from ai.review_queue import enqueue_review
    from storage.organizer import ensure_dirs, output_paths, write_text
    from storage.versioning import get_current_law_text, snapshot_and_update
    from config import INPUT_DIR

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
logger = logging.getLogger("fek_processor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


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


async def process_pdf(pdf_path: str) -> Dict:
    filename = os.path.basename(pdf_path)
    logger.info("process_pdf started file=%s", filename)
    await _log("info", "process_pdf", f"started file={filename}")
    await _event("start", f"Processing {filename}")

    page_infos = detect_document(pdf_path)
    logger.info("detect complete file=%s pages=%s", filename, len(page_infos))
    await _log("info", "detect", f"pages={len(page_infos)}")
    await _event("detect", "Page detection finished", pages=len(page_infos))

    text_layer = extract_document_text_layer(pdf_path, page_infos)
    logger.info("extract complete file=%s text_layer_pages=%s", filename, len(text_layer))
    await _log("info", "extract", f"text_layer_pages={len(text_layer)}")
    await _event("extract", "Text-layer extraction finished")

    def progress(done: int, total: int, page_num: int):
        logger.info("ocr progress file=%s done=%s total=%s page=%s", filename, done, total, page_num)
        asyncio.create_task(_event("ocr", f"OCR page {page_num}", done=done, total=total))
        asyncio.create_task(_log("info", "ocr", f"done={done}/{total} page={page_num}"))

    ocr_pages = ocr_document(pdf_path, page_infos, progress_callback=progress)
    ocr_map = {item["page_num"]: item["text"] for item in ocr_pages}

    merged_pages = []
    for item in text_layer:
        p = item["page_num"]
        txt = item["text"] if item["text"] else ocr_map.get(p, "")
        merged_pages.append(f"\n\n[PAGE {p}]\n\n{txt}")

    raw_text = "".join(merged_pages).strip()
    logger.info("merge complete file=%s raw_chars=%s", filename, len(raw_text))
    await _log("info", "merge", f"raw_chars={len(raw_text)}")
    await _event("merge", "Raw text merged", chars=len(raw_text))

    loop = asyncio.get_running_loop()

    def ai_log(source: str, message: str) -> None:
        logger.info("%s %s", source, message)
        asyncio.run_coroutine_threadsafe(_log("info", source, message), loop)

    decision = await asyncio.to_thread(choose_model, None, ai_log)
    logger.info("ai_start model=%s ram_gb=%s reason=%s", decision.model, decision.free_ram_gb, decision.reason)
    await _log("info", "ai", f"model_select model={decision.model} free_ram_gb={decision.free_ram_gb} reason={decision.reason}")
    await _event(
        "ai_start",
        "AI structuring started",
        model=decision.model,
        free_ram_gb=decision.free_ram_gb,
        model_reason=decision.reason,
        raw_chars=len(raw_text),
    )

    structured = await asyncio.to_thread(build_outputs, raw_text, ai_log)
    logger.info("ai_complete model=%s doc_id=%s title=%s", structured.data.get("metadata", {}).get("model"), structured.data.get("document_id"), structured.data.get("title"))
    await _log("info", "ai", f"completed model={structured.data.get('metadata', {}).get('model')} doc_id={structured.data.get('document_id')}")
    await _event(
        "ai",
        "AI structuring completed",
        document_id=structured.data.get("document_id"),
        title=structured.data.get("title"),
        ai_mode=structured.data.get("metadata", {}).get("ai_mode"),
        model=structured.data.get("metadata", {}).get("model"),
        llm_error=structured.data.get("metadata", {}).get("llm_error"),
        ai_preview=structured.html_text[:1500],
    )

    doc_id = structured.data.get("document_id") or filename
    paths = output_paths(doc_id)
    write_text(paths["xml"], structured.xml_text)
    write_text(paths["html"], structured.html_text)
    logger.info("save complete xml=%s html=%s", paths["xml"], paths["html"])
    await _log("info", "save", f"xml={paths['xml']} html={paths['html']}")
    await _event("save", "XML/HTML saved", xml=paths["xml"], html=paths["html"])

    amendment_notes = []
    for amend in structured.data.get("amendments", []):
        law_id = amend.get("target_law_id")
        if not law_id:
            continue

        current = get_current_law_text(law_id)
        result = apply_amendments(current, [amend])
        amendment_notes.extend(result.notes)
        await _log("info", "amend", f"target_law_id={law_id} changed={result.changed} notes={len(result.notes)}")

        if result.changed:
            current_path, version_path = snapshot_and_update(
                law_id=law_id,
                new_text=result.new_law_text,
                effective_date=structured.data.get("publication_date"),
            )
            await _event(
                "amend",
                f"Applied amendment to {law_id}",
                current=current_path,
                version=version_path,
            )

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
        await _event("review", "Queued for human review", review_file=review_path)
        await _log("warning", "review", f"queued file={review_path} questions={len(structured.review_questions) + len(amendment_notes)}")

    await _event("done", f"Finished {filename}", document_id=doc_id)
    await _log("info", "process_pdf", f"done file={filename} document_id={doc_id}")

    return {
        "document_id": doc_id,
        "title": structured.data.get("title"),
        "xml": paths["xml"],
        "html": paths["html"],
        "needs_review": structured.needs_review or bool(amendment_notes),
    }


@app.post("/process")
async def process(files: List[UploadFile] = File(...)):
    ensure_dirs()
    os.makedirs(INPUT_DIR, exist_ok=True)

    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            logger.warning("skipped non-pdf file=%s", f.filename)
            await _log("warning", "process", f"skipped_non_pdf file={f.filename}")
            await _event("error", f"Skipped non-PDF file: {f.filename}")
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
            await _log("error", "process", f"failed file={f.filename} error={exc}", traceback=tb)
            await _event("error", f"Failed {f.filename}", error=str(exc))

    return JSONResponse({"ok": True, "count": len(results), "results": results})


@app.post("/ai/purge")
async def ai_purge():
    """
    Safely stop active Ollama model sessions, remove local models, and stop ollama serve.
    """
    removed: List[str] = []
    errors: List[str] = []

    await _log("warning", "ai_purge", "requested purge of local AI runtime and models")

    client = ollama.Client(host="http://127.0.0.1:11434", timeout=10)

    # Stop all loaded models first.
    try:
        ps_resp = client.ps()
        active = ps_resp.get("models", []) if isinstance(ps_resp, dict) else []
        for item in active:
            name = item.get("model") or item.get("name")
            if not name:
                continue
            try:
                client.stop(name)
                await _log("info", "ai_purge", f"stopped active model={name}")
            except Exception as exc:
                errors.append(f"stop {name}: {exc}")
                await _log("error", "ai_purge", f"failed stopping model={name} error={exc}")
    except Exception as exc:
        errors.append(f"ps: {exc}")
        await _log("error", "ai_purge", f"failed listing active sessions error={exc}")

    # Remove local models.
    try:
        list_resp = client.list()
        models = list_resp.get("models", []) if isinstance(list_resp, dict) else []
        for item in models:
            name = item.get("model") or item.get("name")
            if not name:
                continue
            try:
                client.delete(name)
                removed.append(name)
                await _log("info", "ai_purge", f"removed model={name}")
            except Exception as exc:
                errors.append(f"delete {name}: {exc}")
                await _log("error", "ai_purge", f"failed deleting model={name} error={exc}")
    except Exception as exc:
        errors.append(f"list: {exc}")
        await _log("error", "ai_purge", f"failed listing local models error={exc}")

    # Stop ollama server process.
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], check=False)
        await _log("info", "ai_purge", "terminated ollama serve process")
    except Exception as exc:
        errors.append(f"pkill: {exc}")
        await _log("error", "ai_purge", f"failed to terminate ollama serve error={exc}")

    await _event(
        "ai_purge",
        "AI purge completed",
        removed_models=removed,
        errors=errors,
    )

    return JSONResponse(
        {
            "ok": len(errors) == 0,
            "removed_models": removed,
            "errors": errors,
        }
    )
