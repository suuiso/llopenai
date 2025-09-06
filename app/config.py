import os
from typing import Dict, Set


def _get_int(name: str, default_value: int) -> int:
    try:
        return int(os.getenv(name, str(default_value)))
    except (TypeError, ValueError):
        return default_value


def _get_allowed_extensions() -> Set[str]:
    exts = os.getenv("ALLOWED_EXTENSIONS", "txt")
    parts = [p.strip().lower().lstrip(".") for p in exts.split(",") if p.strip()]
    return set(parts or ["txt"])


def get_config() -> Dict[str, object]:
    max_mb = _get_int("MAX_CONTENT_LENGTH_MB", 2)
    return {
        "MAX_CONTENT_LENGTH": max_mb * 1024 * 1024,  # bytes
        "ALLOWED_EXTENSIONS": _get_allowed_extensions(),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        "PREFERRED_URL_SCHEME": "https",
    }

