"""Model manager for local Ollama models with dynamic fallback based on free RAM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import subprocess
import time
from urllib.parse import urlparse
import os

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


def get_free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _emit(log_hook: Optional[Callable[[str, str], None]], source: str, message: str) -> None:
    if log_hook:
        try:
            log_hook(source, message)
        except Exception:
            pass


def _list_available_models(client: ollama.Client) -> List[str]:
    response = client.list()
    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", []) or []
    names = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("model") or item.get("name")
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if name:
            names.append(name)
    return names


def _start_ollama_server_if_needed(log_hook: Optional[Callable[[str, str], None]] = None) -> None:
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
        _emit(log_hook, "ollama", f"server not reachable at {OLLAMA_BASE_URL}, attempting auto-start")

    env = dict()
    # inherit caller env safely
    import os
    env.update(os.environ)
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
    last_err = None
    while time.time() < deadline:
        try:
            probe = ollama.Client(host=OLLAMA_BASE_URL, timeout=2)
            probe.list()
            _emit(log_hook, "ollama", "auto-start successful")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)

    raise RuntimeError(f"Failed to auto-start Ollama server at {OLLAMA_BASE_URL}: {last_err}")


def _ensure_local_model(client: ollama.Client, log_hook: Optional[Callable[[str, str], None]] = None) -> None:
    if not OLLAMA_AUTO_PULL:
        return

    available = _list_available_models(client)
    _emit(log_hook, "ollama", f"local models detected: {', '.join(available) if available else '[none]'}")
    if available:
        return

    preferred = [cfg["name"] for cfg in OLLAMA_MODELS]
    pull_errors = []
    pull_no_effect = []
    for model_name in preferred:
        try:
            _emit(log_hook, "ollama", f"auto-pull starting model={model_name}")
            # Use CLI pull as a blocking, reliable path across ollama-python versions.
            cmd = ["ollama", "pull", model_name]
            env = dict(os.environ)
            parsed = urlparse(OLLAMA_BASE_URL)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
            env["OLLAMA_HOST"] = f"http://{host}:{port}"
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pull command failed rc={proc.returncode} stderr={proc.stderr.strip()[:300]}"
                )
            _emit(log_hook, "ollama", f"auto-pull completed model={model_name}")
            # refresh after pull attempt
            refreshed = _list_available_models(client)
            _emit(
                log_hook,
                "ollama",
                f"local models after pull: {', '.join(refreshed) if refreshed else '[none]'}",
            )
            if refreshed:
                _emit(log_hook, "ollama", "at least one local model is now available")
                return
            pull_no_effect.append(model_name)
        except Exception as exc:
            _emit(log_hook, "ollama", f"auto-pull failed model={model_name} error={exc}")
            pull_errors.append(f"{model_name}: {exc}")

    details = []
    if pull_errors:
        details.append("errors=" + " | ".join(pull_errors))
    if pull_no_effect:
        details.append("no_effect_models=" + ", ".join(pull_no_effect))
    detail_suffix = (" (" + "; ".join(details) + ")") if details else ""
    raise RuntimeError(
        "No local Ollama models and auto-pull failed" + detail_suffix
    )


def _candidate_models(client: ollama.Client) -> List[str]:
    free_ram = get_free_ram_gb()
    available = _list_available_models(client)
    available_set = set(available)

    preferred = []
    for cfg in OLLAMA_MODELS:
        name = cfg["name"]
        min_ram = float(cfg["min_free_ram_gb"])
        if name in available_set and free_ram >= min_ram:
            preferred.append(name)

    for cfg in OLLAMA_MODELS:
        name = cfg["name"]
        if name in available_set and name not in preferred:
            preferred.append(name)

    for name in available:
        if name not in preferred:
            preferred.append(name)

    return preferred


def choose_model(client: Optional[ollama.Client] = None, log_hook: Optional[Callable[[str, str], None]] = None) -> ModelDecision:
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
        reason=f"selected {selected} from local candidates: {', '.join(candidates)}",
    )


def should_pause_for_memory() -> bool:
    return get_free_ram_gb() < RAM_DANGER_GB


def chat_json(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.0,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict:
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
                    "num_ctx": min(1024, OLLAMA_NUM_CTX),
                    "num_predict": min(350, OLLAMA_MAX_OUTPUT_TOKENS),
                    "num_gpu": 0,
                },
            ),
        ]

        last_exc = None
        for profile_name, opts in profiles:
            try:
                _emit(log_hook, "ai", f"chat attempt model={model_name} profile={profile_name} opts={opts}")
                response = client.chat(
                    model=model_name,
                    format="json",
                    options=opts,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.get("message", {}).get("content", "{}")
                return {
                    "model": model_name,
                    "free_ram_gb": free_ram,
                    "downgraded": model_name != candidates[0],
                    "reason": f"used {model_name}({profile_name}); candidates were: {', '.join(candidates)}",
                    "raw": content,
                }
            except Exception as exc:
                last_exc = exc
                _emit(log_hook, "ai", f"chat failed model={model_name} profile={profile_name} error={exc}")
        errors.append(f"{model_name}: {last_exc}")

    raise RuntimeError("All candidate models failed: " + " | ".join(errors))
