from __future__ import annotations

import io
from typing import List, Tuple

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
)

from .chunking import chunk_by_characters, chunk_by_paragraphs
from .gemini_client import GeminiClient


bp = Blueprint("main", __name__)


def _allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in current_app.config.get("ALLOWED_EXTENSIONS", {"txt"})


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/process")
def process():
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", error="Selecciona un archivo .txt"), 400

    if not _allowed_file(file.filename):
        return (
            render_template(
                "index.html", error="Extensión de archivo no permitida (usa .txt)"
            ),
            400,
        )

    # Read file safely into memory
    try:
        raw_bytes = file.read()
        if raw_bytes is None:
            raw_bytes = b""
        text_content = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return (
            render_template(
                "index.html",
                error="No se pudo leer/decodificar el archivo. Usa UTF-8.",
            ),
            400,
        )

    strategy = (request.form.get("strategy") or "paragraphs").strip()
    model_name = current_app.config.get("GEMINI_MODEL", "gemini-2.5-pro")
    api_key = current_app.config.get("GEMINI_API_KEY")

    # Parse chunking parameters
    try:
        if strategy == "characters":
            chunk_size = int(request.form.get("chunk_size", 1000))
            overlap = int(request.form.get("overlap", 120))
            chunk_size = max(200, min(chunk_size, 4000))
            overlap = max(0, min(overlap, chunk_size - 1))
            chunks = chunk_by_characters(text_content, chunk_size, overlap)
        else:
            max_chars = int(request.form.get("max_chars", 1200))
            max_chars = max(300, min(max_chars, 4000))
            chunks = chunk_by_paragraphs(text_content, max_chars)
    except Exception:
        return (
            render_template(
                "index.html", error="Parámetros de chunking inválidos."
            ),
            400,
        )

    if not chunks:
        return render_template("index.html", error="El archivo está vacío."), 400

    # Call Gemini per chunk
    try:
        client = GeminiClient(api_key=api_key, model_name=model_name)
    except Exception as exc:
        return (
            render_template(
                "index.html",
                error=f"Error inicializando cliente de Gemini: {exc}",
            ),
            500,
        )

    results: List[Tuple[str, str]] = []  # (chunk_text, gemini_output)
    for chunk in chunks:
        output = client.summarize_chunk(chunk, language="es")
        results.append((chunk, output))

    return render_template(
        "result.html",
        filename=file.filename,
        strategy=strategy,
        num_chunks=len(chunks),
        results=results,
    )

