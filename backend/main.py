"""FastAPI app for FEK OCR + AI structuring pipeline."""

from __future__ import annotations

import asyncio
try:
    import fitz
    _MISSING_PYMUPDF = False
except Exception:  # pragma: no cover - allow importing without PyMuPDF
    fitz = None  # type: ignore
    _MISSING_PYMUPDF = True
import json
import logging
import os
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    _MISSING_FASTAPI = False
except Exception:  # pragma: no cover - allow importing without deps
    _MISSING_FASTAPI = True

    # Lightweight stubs so importing this module outside the project's venv
    # doesn't raise ImportError. These are minimal and only exist to make
    # `import backend.main` succeed for tooling or tests that do not run the
    # server.
    class _StubRoute:
        def __call__(self, *a, **kw):
            def _decorator(func):
                return func

            return _decorator

    class FastAPI:  # type: ignore
        def __init__(self, *a, **kw):
            pass

        def add_middleware(self, *a, **kw):
            pass

        def mount(self, *a, **kw):
            pass

        get = _StubRoute()
        post = _StubRoute()
        websocket = _StubRoute()

    class File:  # type: ignore
        def __init__(self, *a, **kw):
            pass

    class UploadFile:  # type: ignore
        def __init__(self, *a, **kw):
            pass

    class WebSocket:  # type: ignore
        pass

    class WebSocketDisconnect(Exception):  # type: ignore
        pass

    class CORSMiddleware:  # type: ignore
        pass

    class FileResponse:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("FileResponse not available: install FastAPI")

    class JSONResponse:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("JSONResponse not available: install FastAPI")

    class StaticFiles:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("StaticFiles not available: install FastAPI")

# Attempt to import optional/third-party components. If those imports fail
# (e.g. when running outside the project's venv), provide safe fallbacks so
# the module can still be imported for static analysis or simple tooling.
try:
    from markitdown import MarkItDown
    from backend.ocr.detector import detect_document, extract_images_from_pdf
    from backend.ocr.extractor import extract_document_text_layer
    from backend.ocr.engine import ocr_document, ocr_image
    from backend.ai.structurer import build_outputs
    from backend.ai.model_manager import choose_model, classify_image, stop_active_models
    from backend.ai.diff_engine import apply_amendments
    from backend.ai.review_queue import enqueue_review, list_pending_reviews
    from backend.fek.raw_extractor import (
        extract_raw_text,
        extract_metadata,
        build_archive_xml,
        build_archive_html,
    )
    from backend.storage.organizer import ensure_dirs, output_paths, write_text
    from backend.storage.versioning import get_current_law_text, snapshot_and_update
    from backend.codification.codifier import codify
    from backend.config import (
        INPUT_DIR,
        OLLAMA_BASE_URL,
        OUTPUT_HTML_DIR,
        OUTPUT_XML_DIR,
        DOCINTEL_ENDPOINT,
        IMAGES_DIR,
    )
except Exception:  # pragma: no cover - careful fallback for missing deps
    try:
        # Try local (package-relative) imports if package layout differs
        from markitdown import MarkItDown
        from ocr.detector import detect_document, extract_images_from_pdf
        from ocr.extractor import extract_document_text_layer
        from ocr.engine import ocr_document, ocr_image
        from ai.structurer import build_outputs
        from ai.model_manager import choose_model, classify_image, stop_active_models
        from ai.diff_engine import apply_amendments
        from ai.review_queue import enqueue_review, list_pending_reviews
        from fek.raw_extractor import (
            extract_raw_text,
            extract_metadata,
            build_archive_xml,
            build_archive_html,
        )
        from storage.organizer import ensure_dirs, output_paths, write_text
        from storage.versioning import get_current_law_text, snapshot_and_update
        from codification.codifier import codify
        from config import (
            INPUT_DIR,
            OLLAMA_BASE_URL,
            OUTPUT_HTML_DIR,
            OUTPUT_XML_DIR,
            DOCINTEL_ENDPOINT,
            IMAGES_DIR,
        )
    except Exception:
        # Provide minimal stubs so the module can be imported. Each stub will
        # raise a helpful RuntimeError if called, except for simple helpers
        # like `list_pending_reviews` and `ensure_dirs` which are safe no-ops.
        def _missing(name):
            def _fn(*a, **kw):
                raise RuntimeError(f"Missing runtime dependency for {name}; install project dependencies to use this feature")

            return _fn

        class MarkItDown:  # minimal wrapper used only for imports
            def __init__(self, *a, **kw):
                pass

            def convert(self, *a, **kw):
                class R:
                    text_content = ""

                return R()

        detect_document = _missing("detect_document")
        extract_images_from_pdf = _missing("extract_images_from_pdf")
        extract_document_text_layer = _missing("extract_document_text_layer")
        ocr_document = _missing("ocr_document")
        ocr_image = _missing("ocr_image")
        build_outputs = _missing("build_outputs")
        choose_model = _missing("choose_model")
        classify_image = _missing("classify_image")
        apply_amendments = _missing("apply_amendments")
        enqueue_review = _missing("enqueue_review")
        list_pending_reviews = lambda: []
        ensure_dirs = lambda: None

        def output_paths(doc_id: str):
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")
            return {"xml": os.path.join(base, "xml", f"{doc_id}.xml"), "html": os.path.join(base, "html", f"{doc_id}.html")}

        write_text = lambda path, text: None
        get_current_law_text = _missing("get_current_law_text")
        snapshot_and_update = _missing("snapshot_and_update")
        codify = _missing("codify")

        # Conservative defaults for config values used at import time
        SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
        INPUT_DIR = os.path.join(SCRIPT_DIR, "data", "input")
        OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        OUTPUT_HTML_DIR = os.path.join(SCRIPT_DIR, "data", "output", "html")
        OUTPUT_XML_DIR = os.path.join(SCRIPT_DIR, "data", "output", "xml")
        DOCINTEL_ENDPOINT = None
        IMAGES_DIR = os.path.join(SCRIPT_DIR, "data", "output", "images")

app = FastAPI(title="FEK Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if not _MISSING_FASTAPI:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

from backend.config import DATA_DIR
if not _MISSING_FASTAPI:
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

    loop = asyncio.get_running_loop()

    # ── Faithful raw extraction (NO AI restructuring) ──────────
    # Επίπεδο 1: εξάγουμε το ΦΕΚ αυτούσιο (verbatim) με τα υπάρχοντα
    # detector + extractor + OCR engine. Κανένα AI restructuring.
    def extract_log(source: str, message: str) -> None:
        logger.info("%s %s", source, message)
        asyncio.run_coroutine_threadsafe(_log("info", source, message), loop)

    await _event("extract", "Extracting raw text faithfully (column-aware + OCR)")
    extraction = await asyncio.to_thread(extract_raw_text, pdf_path, extract_log)
    raw_text = extraction["raw_text"]
    await _event(
        "extract",
        "Raw text extracted",
        chars=len(raw_text),
        page_count=extraction["page_count"],
    )

    # ── Regex-based metadata (NO AI) ────────────────────────────
    metadata = extract_metadata(raw_text)

    # Build doc_id from metadata, fall back to filename.
    teychos = metadata.get("teychos")
    arithmos = metadata.get("arithmos_fyllou")
    hmerominia = metadata.get("hmerominia")
    year = hmerominia.split("-")[0] if hmerominia else None
    if teychos and arithmos and year:
        doc_id = f"ΦΕΚ {teychos} {arithmos}-{year}"
    else:
        doc_id = os.path.splitext(filename)[0]
    title = metadata.get("titlos") or doc_id

    # ── Build light archival XML + HTML ─────────────────────────
    xml_text = build_archive_xml(metadata, raw_text)
    html_text = build_archive_html(metadata, raw_text, images_url_prefix="/data/output/images")

    # ── Save outputs ────────────────────────────────────────────
    paths = output_paths(doc_id)
    write_text(paths["xml"], xml_text)
    write_text(paths["html"], html_text)
    await _event(
        "save",
        "XML/HTML saved",
        xml=paths["xml"],
        html=paths["html"],
        html_url=f"/outputs/html/{os.path.basename(paths['html'])}",
        xml_url=f"/outputs/xml/{os.path.basename(paths['xml'])}",
    )

    await _event(
        "done",
        f"Finished {filename}",
        document_id=doc_id,
        html_url=f"/outputs/html/{os.path.basename(paths['html'])}",
    )
    return {
        "document_id": doc_id,
        "title": title,
        "xml": paths["xml"],
        "html": paths["html"],
        "html_url": f"/outputs/html/{os.path.basename(paths['html'])}",
        "xml_url": f"/outputs/xml/{os.path.basename(paths['xml'])}",
        "needs_review": False,
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


# ── Codification (Level 2) ──────────────────────────────────────────────────


def _read_archive_keimeno(doc_id: str):
    """Διάβασε <Keimeno> + <Hmerominia> από το αποθηκευμένο αρχειακό XML.

    Reads the saved archive XML for doc_id from OUTPUT_XML_DIR and returns
    (raw_text, hmerominia). Returns (None, None) if not found.
    """
    paths = output_paths(doc_id)
    xml_path = paths["xml"]
    if not os.path.exists(xml_path):
        return None, None
    try:
        from lxml import etree as ET  # local import; lxml is a project dep
        tree = ET.parse(xml_path)
        root = tree.getroot()
        keimeno_el = root.find(".//Keimeno")
        hmer_el = root.find(".//Hmerominia")
        raw_text = keimeno_el.text if keimeno_el is not None else None
        hmerominia = hmer_el.text if hmer_el is not None else None
        return raw_text, hmerominia
    except Exception as exc:
        logger.warning("could not parse archive XML for %s: %s", doc_id, exc)
        return None, None


@app.post("/codify")
async def codify_endpoint(payload: Dict):
    """Κωδικοποίηση νόμων από τροποποιήσεις ΦΕΚ (explicit, ξεχωριστό βήμα).

    Accepts JSON either:
      {"raw_text": "...", "effective_date": "YYYY-MM-DD"?, "use_claude": false}
    OR
      {"doc_id": "..."}  → reads <Keimeno>/<Hmerominia> from the saved archive XML.
    Returns {"ok": true, "results": [CodificationResult, ...]}.
    """
    raw_text = (payload or {}).get("raw_text")
    effective_date = (payload or {}).get("effective_date")
    use_claude = bool((payload or {}).get("use_claude", False))
    doc_id = (payload or {}).get("doc_id")

    if doc_id and not raw_text:
        raw_text, hmerominia = _read_archive_keimeno(doc_id)
        if raw_text is None:
            return JSONResponse(
                {"ok": False, "error": f"archive not found for doc_id={doc_id}"},
                status_code=404,
            )
        if not effective_date:
            effective_date = hmerominia

    if not raw_text:
        return JSONResponse(
            {"ok": False, "error": "raw_text or doc_id required"}, status_code=400
        )

    await _event("codify_start", "Codification started", doc_id=doc_id)

    loop = asyncio.get_running_loop()

    def codify_log(source: str, message: str) -> None:
        logger.info("%s %s", source, message)
        asyncio.run_coroutine_threadsafe(_log("info", source, message), loop)

    try:
        results = await asyncio.to_thread(
            codify, raw_text, effective_date, codify_log, use_claude
        )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("codify failed")
        await _log("error", "codify", f"failed error={exc}", traceback=tb)
        await _event("codify_done", "Codification failed", error=str(exc))
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    out = [
        {
            "law_id": r.law_id,
            "applied": r.applied,
            "queued_for_review": r.queued_for_review,
            "diff": r.diff,
            "current_path": r.current_path,
            "version_path": r.version_path,
            "notes": r.notes,
        }
        for r in results
    ]
    for r in out:
        await _event(
            "codify",
            f"Codified {r['law_id']}: applied={r['applied']} queued={r['queued_for_review']}",
            **r,
        )

    await _event("codify_done", "Codification finished", count=len(out))
    return JSONResponse({"ok": True, "count": len(out), "results": out})


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
