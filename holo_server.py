r"""
SCRIPT: holo_server.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Servidor API OpenAI-compatible con soporte CORS y frontend WebGL 3D integrado.
"""

import os
import time
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import List, Optional

from src.engine.holo_engine import HoloInferenceEngine

app = FastAPI(title="HoloLLM API Server (O(1) Memory)")

# Habilitar CORS para permitir peticiones web locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando Motor HoloLLM en memoria local (CPU)...")
engine = HoloInferenceEngine(device="cpu")
print("✔ HoloLLM listo.")


class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "holollm-70m"
    messages: List[Message]
    stream: bool = False
    max_tokens: int = 200
    temperature: float = 0.2


@app.get("/", response_class=HTMLResponse)
async def serve_cosmos_ui():
    """Sirve la interfaz gráfica WebGL 3D directamente en la raíz."""
    ui_path = "tools/visualization/holo_cosmos_chat.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>HoloLLM API activa. Interfaz no encontrada.</h1>"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "holollm-70m", "object": "model", "created": int(time.time()), "owned_by": "MRCSIBR/HoloLLM"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_message = request.messages[-1].content

    bragg_map = {
        "fibonacci": "fibonacci with memoization",
        "factorial": "factorial",
        "binary search": "binary search",
        "reverse list": "reverse linked list",
        "bfs": "breadth first search bfs",
        "kadane": "maximum subarray kadane"
    }
    query_clean = user_message.strip().lower()
    task_str = bragg_map.get(query_clean, user_message)

    prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task_str}<|im_end|>\n<|im_start|>assistant\n```python\n"

    if request.stream:
        async def event_generator():
            response_id = f"chatcmpl-holo-{int(time.time())}"
            
            yield {"data": json.dumps({
                "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request.model, "choices": [{"index": 0, "delta": {"content": "```python\n"}, "finish_reason": None}]
            })}
            
            for chunk_text, _ in engine.generate_stream(prompt, max_new_tokens=request.max_tokens):
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield {"data": json.dumps(chunk_data)}
            
            final_data = {
                "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield {"data": json.dumps(final_data)}
            yield {"data": "[DONE]"}
            
        return EventSourceResponse(event_generator())

    else:
        full_text = "```python\n"
        for chunk_text, _ in engine.generate_stream(prompt, max_new_tokens=request.max_tokens):
            full_text += chunk_text

        return JSONResponse({
            "id": f"chatcmpl-holo-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}]
        })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
