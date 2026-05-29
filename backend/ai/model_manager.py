"""Model manager for local Ollama models with dynamic fallback based on free RAM."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import ollama
import psutil

try:
    from backend.config import (
        OLLAMA_MODELS,
        OLLAMA_BASE_URL,
        OLLAMA_TIMEOUT,
        RAM_DANGER_GB,
        OLLAMA_AUTO_START,
        OLLAMA_AUTO_PULL,
        OLLAMA_NUM_CTX,
        OLLAMA_MAX_OUTPUT_TOKENS,
    )
except ImportError:
    from config import (
        OLLAMA_MODELS,
        OLLAMA_BASE_URL,
        OLLAMA_TIMEOUT,
        RAM_DANGER_GB,
        OLLAMA_AUTO_START,
        OLLAMA_AUTO_PULL,
        OLLAMA_NUM_CTX,
        OLLAMA_MAX_OUTPUT_TOKENS,
    )


@dataclass
class ModelDecision:
    model: str
    free_ram_gb: float
    downgraded: bool
    reason: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def _emit(
    log_hook: Optional[Callable[[str, str], None]], source: str, message: str
) -> None:
    if log_hook:
        try:
            log_hook(source, message)
        except Exception:
            pass


def _coerce_list(obj) -> list:
    """Return obj as a plain list regardless of whether it is a dict, list, or Pydantic model."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("models", [])
    # Pydantic / dataclass object (ollama >= 0.2)
    return list(getattr(obj, "models", None) or [])


def _model_name(item) -> Optional[str]:
    """Extract model name from a list-response item (dict or object)."""
    if isinstance(item, dict):
        return item.get("model") or item.get("name")
    return getattr(item, "model", None) or getattr(item, "name", None)


def _response_content(response) -> str:
    """
    Extract the assistant message content from an ollama chat response.

    ollama < 0.2  → dict:   response["message"]["content"]
    ollama >= 0.2 → object: response.message.content
    """
    if isinstance(response, dict):
        msg = response.get("message", {})
        if isinstance(msg, dict):
            return msg.get("content", "{}")
        return getattr(msg, "content", "{}")

    # Pydantic ChatResponse
    msg = getattr(response, "message", None)
    if msg is None:
        return "{}"
    if isinstance(msg, dict):
        return msg.get("content", "{}")
    return getattr(msg, "content", "{}")


# ── Ollama server management ─────────────────────────────────────────────────


def _start_ollama_server_if_needed(
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> None:
    if not OLLAMA_AUTO_START:
        return

    parsed = urlparse(OLLAMA_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434

    try:
        probe = ollama.Client(host=OLLAMA_BASE_URL, timeout=2)
        probe.list()
        _emit(log_hook, "ollama", f"server reachable at {OLLAMA_BASE_URL}")
        return
    except Exception:
        _emit(log_hook, "ollama", f"server not reachable – attempting auto-start")

    env = dict(os.environ)
    env.setdefault("OLLAMA_HOST", f"http://{host}:{port}")

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    _emit(log_hook, "ollama", "auto-start command executed: ollama serve")

    deadline = time.time() + 20
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            probe = ollama.Client(host=OLLAMA_BASE_URL, timeout=2)
            probe.list()
            _emit(log_hook, "ollama", "auto-start successful")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)

    raise RuntimeError(
        f"Failed to auto-start Ollama server at {OLLAMA_BASE_URL}: {last_err}"
    )


def _list_available_models(client: ollama.Client) -> List[str]:
    response = client.list()
    items = _coerce_list(response)
    return [n for n in (_model_name(i) for i in items) if n]


def _ensure_local_model(
    client: ollama.Client,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> None:
    if not OLLAMA_AUTO_PULL:
        return

    available = _list_available_models(client)
    _emit(
        log_hook,
        "ollama",
        f"local models: {', '.join(available) if available else '[none]'}",
    )
    if available:
        return

    preferred = [cfg["name"] for cfg in OLLAMA_MODELS]
    pull_errors: List[str] = []

    parsed = urlparse(OLLAMA_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"http://{host}:{port}"

    for model_name in preferred:
        try:
            _emit(log_hook, "ollama", f"auto-pull starting: {model_name}")
            proc = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pull rc={proc.returncode} stderr={proc.stderr.strip()[:300]}"
                )
            _emit(log_hook, "ollama", f"auto-pull completed: {model_name}")
            refreshed = _list_available_models(client)
            if refreshed:
                _emit(log_hook, "ollama", "at least one model is now available")
                return
        except Exception as exc:
            _emit(log_hook, "ollama", f"auto-pull failed: {model_name} – {exc}")
            pull_errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "No local Ollama models and auto-pull failed: " + " | ".join(pull_errors)
    )


# ── Model selection ──────────────────────────────────────────────────────────


def _candidate_models(client: ollama.Client) -> List[str]:
    """Return available models sorted by preference (RAM-aware)."""
    free_ram = get_free_ram_gb()
    available = _list_available_models(client)
    available_set = set(available)

    # 1. Models that fit preferred list AND RAM requirement
    fits: List[str] = []
    for cfg in OLLAMA_MODELS:
        name = cfg["name"]
        if name in available_set and free_ram >= float(cfg["min_free_ram_gb"]):
            fits.append(name)

    # 2. Preferred models that don't meet RAM (still usable as fallback)
    for cfg in OLLAMA_MODELS:
        name = cfg["name"]
        if name in available_set and name not in fits:
            fits.append(name)

    # 3. Any remaining local models
    for name in available:
        if name not in fits:
            fits.append(name)

    return fits


def choose_model(
    client: Optional[ollama.Client] = None,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> ModelDecision:
    _start_ollama_server_if_needed(log_hook=log_hook)
    client = client or ollama.Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
    _ensure_local_model(client, log_hook=log_hook)

    free_ram = get_free_ram_gb()
    candidates = _candidate_models(client)

    if not candidates:
        return ModelDecision(
            model="",
            free_ram_gb=round(free_ram, 2),
            downgraded=True,
            reason="no local Ollama models installed",
        )

    selected = candidates[0]
    top_config = OLLAMA_MODELS[0]["name"] if OLLAMA_MODELS else selected
    downgraded = selected != top_config
    return ModelDecision(
        model=selected,
        free_ram_gb=round(free_ram, 2),
        downgraded=downgraded,
        reason=f"selected {selected} from {', '.join(candidates)}",
    )


def should_pause_for_memory() -> bool:
    return get_free_ram_gb() < RAM_DANGER_GB


def classify_image(
    image_path: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> str:
    """
    Classifies an image using a local multimodal model (e.g., llava).
    Returns "pure_text", "table", "chart", or "photo".
    """
    # Prefer a multimodal model if available, otherwise use default
    _start_ollama_server_if_needed(log_hook=log_hook)
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
    
    available = _list_available_models(client)
    # Heuristic: look for 'llava' or 'bakllava' or 'moondream'
    model = "llava"
    if "llava" not in available:
        if available:
            model = available[0]
        else:
            _ensure_local_model(client, log_hook=log_hook)
            model = _list_available_models(client)[0]

    prompt = (
        "Classify this image into one of these categories: 'pure_text', 'table', 'chart', 'photo'. "
        "Reply with ONLY the category name."
    )

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            images=[image_path],
            stream=False
        )
        category = response.get("response", "photo").strip().lower()
        # Clean up category in case the model ignored "ONLY"
        for candidate in ["pure_text", "table", "chart", "photo"]:
            if candidate in category:
                return candidate
        return "photo"
    except Exception as exc:
        _emit(log_hook, "ai", f"classify_image failed: {exc}")
        return "photo"


# ── Main chat entry point ────────────────────────────────────────────────────


def chat_json(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.0,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """
    Run a chat completion against the best available local model and return
    a dict with keys: model, free_ram_gb, downgraded, reason, raw (str).

    Tries two profiles per model (default → cpu_safe) and iterates over all
    candidate models before giving up.
    """
    _start_ollama_server_if_needed(log_hook=log_hook)
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
    _ensure_local_model(client, log_hook=log_hook)

    free_ram = round(get_free_ram_gb(), 2)
    candidates = _candidate_models(client)

    if not candidates:
        raise RuntimeError("No local Ollama models available after auto-setup.")

    errors: List[str] = []

    for model_name in candidates:
        profiles = [
            (
                "default",
                {
                    "temperature": temperature,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_MAX_OUTPUT_TOKENS,
                },
            ),
            (
                "cpu_safe",
                {
                    "temperature": 0,
                    "num_ctx": min(2048, OLLAMA_NUM_CTX),
                    "num_predict": min(1024, OLLAMA_MAX_OUTPUT_TOKENS),
                    "num_gpu": 0,
                },
            ),
        ]

        for profile_name, opts in profiles:
            try:
                _emit(
                    log_hook,
                    "ai",
                    f"chat attempt model={model_name} profile={profile_name}",
                )
                response = client.chat(
                    model=model_name,
                    format="json",
                    options=opts,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = _response_content(response)
                return {
                    "model": model_name,
                    "free_ram_gb": free_ram,
                    "downgraded": model_name != candidates[0],
                    "reason": f"used {model_name}({profile_name})",
                    "raw": content,
                }
            except Exception as exc:
                _emit(
                    log_hook,
                    "ai",
                    f"chat failed model={model_name} profile={profile_name} error={exc}",
                )

        errors.append(f"{model_name}: last error logged above")

    raise RuntimeError("All candidate models failed: " + " | ".join(errors))
