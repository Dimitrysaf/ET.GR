"""Model manager for local Ollama models with dynamic fallback based on free RAM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import ollama
import psutil

from config import OLLAMA_MODELS, OLLAMA_BASE_URL, RAM_DANGER_GB


@dataclass
class ModelDecision:
    model: str
    free_ram_gb: float
    downgraded: bool
    reason: str


def get_free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _list_available_models(client: ollama.Client) -> List[str]:
    try:
        response = client.list()
        models = response.get("models", []) if isinstance(response, dict) else []
        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
                if name:
                    names.append(name)
        return names
    except Exception:
        return []


def choose_model(client: Optional[ollama.Client] = None) -> ModelDecision:
    client = client or ollama.Client(host=OLLAMA_BASE_URL)
    free_ram = get_free_ram_gb()
    available = set(_list_available_models(client))

    for cfg in OLLAMA_MODELS:
        model_name = cfg["name"]
        min_ram = float(cfg["min_free_ram_gb"])
        if free_ram >= min_ram and (not available or model_name in available):
            return ModelDecision(
                model=model_name,
                free_ram_gb=round(free_ram, 2),
                downgraded=False,
                reason=f"selected {model_name} for free RAM {free_ram:.2f} GB",
            )

    fallback = OLLAMA_MODELS[-1]["name"]
    return ModelDecision(
        model=fallback,
        free_ram_gb=round(free_ram, 2),
        downgraded=True,
        reason=f"fallback to {fallback} (low RAM or unavailable higher-tier models)",
    )


def should_pause_for_memory() -> bool:
    return get_free_ram_gb() < RAM_DANGER_GB


def chat_json(prompt: str, system_prompt: str, temperature: float = 0.0) -> Dict:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    decision = choose_model(client)

    response = client.chat(
        model=decision.model,
        format="json",
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.get("message", {}).get("content", "{}")
    return {
        "model": decision.model,
        "free_ram_gb": decision.free_ram_gb,
        "downgraded": decision.downgraded,
        "reason": decision.reason,
        "raw": content,
    }
