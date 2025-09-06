from typing import Optional

import google.generativeai as genai


class GeminiClient:
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está configurada.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def summarize_chunk(self, chunk_text: str, language: str = "es") -> str:
        prompt = (
            f"Resume el siguiente texto en 1–2 oraciones claras y útiles en {language}.\n"
            f"Texto:\n" + chunk_text
        )
        try:
            response = self.model.generate_content(prompt)
            # google-generativeai returns .text for single response
            return (response.text or "").strip() or "(Sin respuesta)"
        except Exception as exc:  # broad catch to surface error per chunk
            return f"[Error procesando el chunk: {exc}]"

